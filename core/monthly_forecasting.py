# core/monthly_forecasting.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm
from datetime import timedelta
import logging
import traceback
import time
import warnings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information was")
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ValueWarning) # لتجاهل تحذيرات التقارب

# --- الثوابت الشهرية ---
TARGET_COLUMN_MONTHLY = 'avg_invoice_value' # الهدف هو متوسط الفاتورة
DATE_COLUMN_OUTPUT = 'date' # اسم العمود القياسي للناتج (موحد)
DEFAULT_SEASONAL_PERIOD_MONTHLY = 12 # موسمية سنوية
AUTO_ARIMA_MAX_P_M = 3
AUTO_ARIMA_MAX_Q_M = 3
AUTO_ARIMA_MAX_P_SEASONAL_M = 2
AUTO_ARIMA_MAX_Q_SEASONAL_M = 1
CONFIDENCE_ALPHA = 0.05 # 95% CI
MIN_OBSERVATIONS_MONTHLY = max(2 * DEFAULT_SEASONAL_PERIOD_MONTHLY, 24) # سنتين على الأقل

# --- الدوال المساعدة الشهرية (مُكيفة) ---

def _prepare_monthly_forecasting_data(monthly_features_df, target_col):
    """ تحضير البيانات الشهرية من DataFrame الميزات. """
    logger.info(f"بدء تحضير بيانات التنبؤ الشهرية من DataFrame بالشكل: {monthly_features_df.shape}")
    start_time = time.time()
    try:
        if not isinstance(monthly_features_df, pd.DataFrame) or monthly_features_df.empty:
            raise ValueError("DataFrame الميزات الشهري المدخل فارغ أو غير صالح.")

        df = monthly_features_df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("فهرس DataFrame الميزات الشهري ليس DatetimeIndex.")
        if target_col not in df.columns:
            raise ValueError(f"عمود الهدف الشهري '{target_col}' مفقود.")

        # التأكد من التردد الشهري (بداية الشهر)
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq != 'MS':
                logger.warning(f"التردد الشهري المستنتج '{inferred_freq}' ليس 'MS'. محاولة فرض 'MS'.")
                df = df.asfreq('MS') # فرض بداية الشهر
                if df.index.freqstr != 'MS': # تحقق مرة أخرى
                     raise ValueError("فشل فرض التردد الشهري 'MS'.")
                logger.info("تم فرض التردد الشهري 'MS' بنجاح.")
        except Exception as e_freq:
             logger.error(f"خطأ في التعامل مع التردد الشهري: {e_freq}. المتابعة بحذر.")


        df = df.sort_index() # التأكد من الفرز
        min_date, max_date = df.index.min(), df.index.max()
        logger.info(f"فترة البيانات التاريخية الشهرية المستخدمة: من {min_date.strftime('%Y-%m')} إلى {max_date.strftime('%Y-%m')}")

        # معالجة الهدف (endog)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isnull().any():
            nan_count = df[target_col].isnull().sum()
            logger.warning(f"العثور على {nan_count} NaN في الهدف الشهري '{target_col}'. ملء بـ ffill->bfill->0.")
            df[target_col] = df[target_col].ffill().bfill().fillna(0)

        endog = df[target_col].astype('float32')

        # معالجة المتغيرات الخارجية (exog)
        exog_cols = df.columns.drop(target_col, errors='ignore').tolist()
        exog = None
        exog_cols_list = []

        if exog_cols:
            exog = df[exog_cols].copy()
            logger.info(f"تم تحديد {len(exog_cols)} متغير خارجي (Exog) شهري مبدئي: {exog_cols}")

            cols_to_fill = []
            numeric_exog_cols = []
            for col in exog.columns:
                # استبدال inf بـ NaN
                if np.isinf(exog[col]).any():
                     inf_count = np.isinf(exog[col]).sum()
                     logger.warning(f"العمود الشهري '{col}' في exog يحتوي على {inf_count} قيم inf. استبدال بـ NaN.")
                     exog[col] = exog[col].replace([np.inf, -np.inf], np.nan)

                # محاولة التحويل إلى رقمي
                if not pd.api.types.is_numeric_dtype(exog[col]):
                    original_dtype = exog[col].dtype
                    exog[col] = pd.to_numeric(exog[col], errors='coerce')
                    if exog[col].isnull().all() and original_dtype != 'object':
                        logger.warning(f"العمود الشهري '{col}' أصبح كله NaN بعد محاولة التحويل الرقمي. قد يكون غير مفيد.")
                    elif exog[col].isnull().any():
                        logger.warning(f"العمود الشهري '{col}' يحتوي على NaN بعد التحويل الرقمي.")
                        cols_to_fill.append(col)
                elif exog[col].isnull().any():
                    cols_to_fill.append(col) # العمود رقمي بالفعل ولكنه يحتوي على NaN

                # الاحتفاظ فقط بالأعمدة الرقمية
                if pd.api.types.is_numeric_dtype(exog[col]):
                    numeric_exog_cols.append(col)
                else:
                    logger.warning(f"تجاهل العمود الشهري غير الرقمي '{col}' من exog.")

            # إعادة تعيين exog ليحتوي فقط على الأعمدة الرقمية
            exog = exog[numeric_exog_cols]

            # ملء القيم المفقودة في الأعمدة الرقمية المتبقية
            cols_to_fill_numeric = [col for col in cols_to_fill if col in exog.columns]
            if cols_to_fill_numeric:
                logger.warning(f"ملء NaN في أعمدة exog الشهرية الرقمية التالية بـ 0: {cols_to_fill_numeric}")
                exog[cols_to_fill_numeric] = exog[cols_to_fill_numeric].fillna(0)

            # التحقق النهائي من exog
            if not exog.empty:
                if exog.isnull().values.any():
                    nan_cols_final = exog.columns[exog.isnull().any()].tolist()
                    raise ValueError(f"NaN متبقية في exog الشهري بعد المعالجة: {nan_cols_final}")
                if np.isinf(exog.values).any(): # يجب ألا يحدث هذا بعد الاستبدال أعلاه، لكن للتحقق
                     raise ValueError("قيم لانهائية (inf) متبقية في exog الشهري بعد المعالجة.")

                exog = exog.astype('float32')
                exog_cols_list = exog.columns.tolist()
                logger.info(f"تم تحديد {len(exog_cols_list)} متغير خارجي شهري رقمي نهائي.")
            else:
                logger.warning("لم يتبق متغيرات خارجية شهرية رقمية بعد المعالجة.")
                exog = None
        else:
            logger.warning("لم يتم العثور على متغيرات خارجية شهرية محتملة.")

        last_known_date = df.index.max()
        logger.info(f"اكتمل تحضير البيانات الشهرية ({time.time() - start_time:.2f} ث). Endog: {endog.shape}, Exog: {exog.shape if exog is not None else 'None'}")
        return endog, exog, last_known_date, exog_cols_list

    except ValueError as ve:
        logger.error(f"خطأ في بيانات التنبؤ الشهرية: {ve}", exc_info=True)
        return None, None, None, None
    except Exception as e:
        logger.error(f"خطأ غير متوقع أثناء تحضير بيانات التنبؤ الشهرية: {e}", exc_info=True)
        return None, None, None, None

def _create_future_monthly_exog(last_known_date, horizon_months, exog_historical_cols, endog_historical):
    """ إنشاء DataFrame للمتغيرات الخارجية الشهرية المستقبلية. """
    logger.info(f"إنشاء متغيرات خارجية شهرية مستقبلية لـ {horizon_months} أشهر بعد {last_known_date.strftime('%Y-%m')}...")
    if not exog_historical_cols:
        logger.info("لا توجد أعمدة exog تاريخية شهرية، لا حاجة لإنشاء exog مستقبلي.")
        return None

    future_dates = pd.date_range(start=last_known_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq='MS')
    future_exog_df = pd.DataFrame(index=future_dates)

    # --- إنشاء الميزات الزمنية المستقبلية ---
    time_features_created = []
    known_time_features = ['month', 'quarter', 'year'] #, 'month_sin', 'month_cos']
    for feature in known_time_features:
        if feature in exog_historical_cols:
            try:
                if feature == 'month': future_exog_df[feature] = future_exog_df.index.month.astype('int8')
                elif feature == 'quarter': future_exog_df[feature] = future_exog_df.index.quarter.astype('int8')
                elif feature == 'year': future_exog_df[feature] = future_exog_df.index.year.astype('int16')
                # elif feature == 'month_sin': future_exog_df[feature] = np.sin(2 * np.pi * future_exog_df.index.month / 12)
                # elif feature == 'month_cos': future_exog_df[feature] = np.cos(2 * np.pi * future_exog_df.index.month / 12)
                time_features_created.append(feature)
            except Exception as e_time_feat:
                 logger.error(f"خطأ إنشاء ميزة الوقت الشهرية المستقبلية '{feature}': {e_time_feat}")
    logger.info(f"تم إنشاء ميزات الوقت الشهرية المستقبلية: {time_features_created}")

    # --- إنشاء ميزات اللاج المستقبلية ---
    lag_cols = [col for col in exog_historical_cols if col.startswith('avg_invoice_lag_')]
    if lag_cols:
        logger.info("إنشاء ميزات اللاج الشهرية المستقبلية...")
        if endog_historical is None or endog_historical.empty:
            logger.error("endog التاريخي الشهري فارغ، لا يمكن إنشاء لاجات مستقبلية. ملء بـ 0.")
            for col_name in lag_cols: future_exog_df[col_name] = 0.0
        else:
            logger.warning("استخدام آخر قيمة تاريخية فقط لتقدير اللاجات المستقبلية (للتبسيط).")
            endog_hist_sorted = endog_historical.sort_index()
            max_lag_needed = 0
            try: max_lag_needed = max(int(col.split('_lag_')[-1]) for col in lag_cols)
            except ValueError: logger.warning("خطأ في استخلاص أرقام اللاج الشهرية.")

            if len(endog_hist_sorted) < max_lag_needed:
                 logger.warning(f"طول endog الشهري ({len(endog_hist_sorted)}) أصغر من أقصى لاج شهري مطلوب ({max_lag_needed}). قد تكون اللاجات غير دقيقة.")

            for col_name in lag_cols:
                try:
                    lag_num = int(col_name.split('_lag_')[-1])
                    if lag_num <= 0: continue
                    # استخدام آخر قيمة متاحة من السلسلة التاريخية
                    if lag_num <= len(endog_hist_sorted):
                        future_exog_df[col_name] = endog_hist_sorted.iloc[-lag_num]
                    else:
                        logger.warning(f"لا يمكن الحصول على قيمة تاريخية لللاج الشهري '{col_name}' (لاج={lag_num}). ملء بـ 0.")
                        future_exog_df[col_name] = 0.0
                except (ValueError, IndexError, TypeError) as e_lag:
                    logger.warning(f"خطأ معالجة اللاج الشهري '{col_name}': {e_lag}. ملء بـ 0.")
                    future_exog_df[col_name] = 0.0

    # --- التحقق النهائي وملء المفقود ---
    missing_cols = [col for col in exog_historical_cols if col not in future_exog_df.columns]
    if missing_cols:
        logger.warning(f"أعمدة exog شهرية مفقودة في المستقبل (ملء بـ 0): {missing_cols}")
        for col in missing_cols: future_exog_df[col] = 0.0

    try:
        # إعادة الترتيب لضمان تطابق الأعمدة مع النموذج
        future_exog_df = future_exog_df.reindex(columns=exog_historical_cols, fill_value=0.0)
        future_exog_df = future_exog_df.astype('float32')
        if future_exog_df.isnull().values.any():
            nan_cols_final = future_exog_df.columns[future_exog_df.isnull().any()].tolist()
            logger.error(f"NaN متبقية في future_exog الشهري النهائي. ملء بـ 0. الأعمدة: {nan_cols_final}")
            future_exog_df = future_exog_df.fillna(0.0)
        if np.isinf(future_exog_df.values).any():
             inf_cols_final = future_exog_df.columns[np.isinf(future_exog_df).any()].tolist()
             logger.error(f"قيم لانهائية (inf) متبقية في future_exog الشهري النهائي. ملء بـ 0. الأعمدة: {inf_cols_final}")
             future_exog_df = future_exog_df.replace([np.inf, -np.inf], 0.0)
    except Exception as e_reindex:
        logger.error(f"خطأ إعادة فهرسة/تحويل future_exog الشهري: {e_reindex}", exc_info=True)
        return None

    logger.info(f"تم إنشاء DataFrame للمتغيرات الخارجية الشهرية المستقبلية بالشكل: {future_exog_df.shape}")
    return future_exog_df

def _find_best_monthly_sarimax_model(endog_hist, exog_hist, seasonal_period):
    """ البحث عن أفضل معاملات SARIMAX شهرية. """
    logger.info(f"بدء البحث عن أفضل نموذج SARIMAX شهري (m={seasonal_period})...")
    start_time = time.time()

    if endog_hist is None or len(endog_hist) < MIN_OBSERVATIONS_MONTHLY:
        logger.error(f"السلسلة endog الشهرية قصيرة جدًا ({len(endog_hist) if endog_hist is not None else 0} < {MIN_OBSERVATIONS_MONTHLY}) للبحث الموسمي.")
        return None, None

    exog_for_search = None
    if exog_hist is not None:
        if isinstance(exog_hist, pd.DataFrame) and not exog_hist.empty and endog_hist.index.equals(exog_hist.index) and not exog_hist.isnull().values.any() and not np.isinf(exog_hist.values).any():
            exog_for_search = exog_hist.astype('float32')
            logger.info(f"استخدام {exog_for_search.shape[1]} exog في البحث الشهري.")
        else:
            logger.warning("تجاهل exog في البحث الشهري بسبب مشاكل في البيانات أو عدم التطابق.")

    try:
        # التأكد أن endog مفهرسة بشكل صحيح
        if not isinstance(endog_hist.index, pd.DatetimeIndex) or endog_hist.index.freqstr != 'MS':
             logger.warning("إعادة فهرسة endog_hist بـ 'MS' قبل auto_arima.")
             endog_hist = endog_hist.asfreq('MS') # قد ينتج NaN إذا كان هناك فجوات غير متوقعة
             endog_hist = endog_hist.ffill().bfill().fillna(0) # ملء أي NaN ناتج

        auto_model = pm.auto_arima(y=endog_hist,
                                   exogenous=exog_for_search,
                                   start_p=1, start_q=1,
                                   max_p=AUTO_ARIMA_MAX_P_M, max_q=AUTO_ARIMA_MAX_Q_M,
                                   m=seasonal_period,
                                   start_P=0, start_Q=0,
                                   max_P=AUTO_ARIMA_MAX_P_SEASONAL_M, max_Q=AUTO_ARIMA_MAX_Q_SEASONAL_M,
                                   seasonal=True,
                                   d=None, D=None, # اسمح لـ auto_arima بتحديد الفروقات
                                   test='adf', # اختبار الوحدة
                                   seasonal_test='ocsb', # اختبار الموسمية
                                   trace=False, # تعطيل الطباعة التفصيلية للبحث
                                   error_action='ignore', # تجاهل النماذج التي تفشل
                                   suppress_warnings=True, # كبت تحذيرات التقارب وغيرها
                                   stepwise=True, # بحث فعال
                                   n_jobs=-1, # استخدام كل المعالجات المتاحة
                                   maxiter=150, # السماح بمزيد من التكرارات للتقارب
                                   information_criterion='aic') # معيار الاختيار

        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        logger.info(f"auto_arima الشهري انتهى ({time.time() - start_time:.2f} ث).")
        logger.info(f"أفضل المعاملات الشهرية: Order={best_order}, Seasonal={best_seasonal_order}")
        return best_order, best_seasonal_order

    except Exception as e:
        logger.error(f"خطأ auto_arima الشهري: {e}", exc_info=True)
        logger.warning(f"فشل البحث، استخدام معاملات شهرية افتراضية: (1,1,1)(1,0,0,{seasonal_period}).")
        # معاملات افتراضية بسيطة
        return (1, 1, 1), (1, 0, 0, seasonal_period)

def _train_final_monthly_sarimax_model(endog_hist, exog_hist, order, seasonal_order):
    """ تدريب نموذج SARIMAX الشهري النهائي. """
    logger.info(f"بدء تدريب النموذج الشهري النهائي بـ Order={order}, Seasonal={seasonal_order}...")
    start_time = time.time()
    if order is None or seasonal_order is None:
        logger.error("المعاملات الشهرية غير متوفرة للتدريب.")
        return None

    exog_for_train = None
    if exog_hist is not None:
        if isinstance(exog_hist, pd.DataFrame) and not exog_hist.empty and endog_hist.index.equals(exog_hist.index) and not exog_hist.isnull().values.any() and not np.isinf(exog_hist.values).any():
            exog_for_train = exog_hist.astype('float32')
            logger.info(f"استخدام {exog_for_train.shape[1]} exog في التدريب الشهري.")
        else:
             logger.warning("تجاهل exog في التدريب الشهري بسبب مشاكل بيانات أو عدم تطابق.")

    try:
        # التأكد من التردد قبل التدريب
        endog_train = endog_hist.astype('float32').asfreq('MS')
        if exog_for_train is not None:
             exog_for_train = exog_for_train.asfreq('MS')
             # التأكد من عدم وجود NaN بعد الفرض
             if exog_for_train.isnull().values.any():
                  logger.warning("ملء NaN في exog_for_train بعد فرض التردد.")
                  exog_for_train = exog_for_train.fillna(0) # أو ffill/bfill

        model = sm.tsa.SARIMAX(endog=endog_train,
                               exog=exog_for_train,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False, # السماح بعدم الاستقرارية
                               enforce_invertibility=False) # السماح بعدم الانعكاسية
        # زيادة maxiter وتجربة طرق حل مختلفة إذا لزم الأمر
        results = model.fit(disp=False, maxiter=250, method='lbfgs') # 'powell', 'cg', 'ncg'
        logger.info(f"تم تدريب النموذج الشهري بنجاح ({time.time() - start_time:.2f} ث). AIC: {results.aic:.2f}")
        return results
    except np.linalg.LinAlgError as lae:
         logger.error(f"خطأ جبر خطي أثناء التدريب الشهري: {lae}. قد تحتاج البيانات لمزيد من الفروقات أو تبسيط النموذج.", exc_info=True)
         return None
    except ValueError as ve:
         # قد يحدث بسبب NaN أو قيم غير صالحة أخرى
         logger.error(f"خطأ قيمة أثناء التدريب الشهري: {ve}. تحقق من البيانات المدخلة.", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"خطأ غير متوقع أثناء تدريب النموذج الشهري النهائي: {e}", exc_info=True)
        return None

def _generate_monthly_future_forecast(model_results, steps, future_exog):
    """ توليد التنبؤات الشهرية المستقبلية وفترات الثقة. """
    if model_results is None:
        logger.error("النموذج الشهري المدرب غير متوفر للتنبؤ.")
        return None

    logger.info(f"بدء توليد التنبؤات الشهرية لـ {steps} خطوات...")
    start_time = time.time()

    final_future_exog = None
    model_needs_exog = model_results.model.exog is not None
    model_exog_names = getattr(model_results.model, 'exog_names', [])

    if model_needs_exog:
        logger.info("النموذج الشهري يتطلب exog للتنبؤ.")
        if future_exog is None or not isinstance(future_exog, pd.DataFrame) or future_exog.empty:
             logger.error("future_exog الشهري مطلوب وغير متوفر/صالح.")
             return None
        if len(future_exog) != steps:
             logger.error(f"طول future_exog الشهري ({len(future_exog)}) لا يطابق steps ({steps}).")
             return None
        if future_exog.isnull().values.any() or np.isinf(future_exog.values).any():
             logger.error("future_exog الشهري يحتوي على NaN أو inf.")
             return None
        try:
             # التأكد من أن أعمدة future_exog مطابقة للنموذج
             if set(future_exog.columns) != set(model_exog_names):
                 logger.warning(f"أعمدة future_exog الشهرية تختلف عن النموذج. محاولة إعادة الترتيب...")
                 # إعادة الترتيب وملء أي أعمدة ناقصة بـ 0
                 final_future_exog = future_exog.reindex(columns=model_exog_names, fill_value=0.0).copy()
             else:
                 final_future_exog = future_exog.copy()
             final_future_exog = final_future_exog.astype('float32')
             # التأكد من التردد
             if not isinstance(final_future_exog.index, pd.DatetimeIndex) or final_future_exog.index.freqstr != 'MS':
                  logger.warning("فرض التردد 'MS' على future_exog قبل التنبؤ.")
                  final_future_exog = final_future_exog.asfreq('MS').fillna(0) # ملء أي NaN ناتج
        except Exception as e_exog_prep:
            logger.error(f"خطأ تحضير future_exog الشهري للتنبؤ: {e_exog_prep}", exc_info=True)
            return None
    elif future_exog is not None:
         logger.warning("تجاهل future_exog الشهري (النموذج لا يتطلبه).")


    try:
        # استخدام get_forecast للحصول على التنبؤات وفترات الثقة
        forecast_obj = model_results.get_forecast(steps=steps, exog=final_future_exog, alpha=CONFIDENCE_ALPHA)
        y_pred_future = forecast_obj.predicted_mean
        conf_int_future = forecast_obj.conf_int(alpha=CONFIDENCE_ALPHA)

        # إنشاء DataFrame للنتائج
        future_df = pd.DataFrame({
            DATE_COLUMN_OUTPUT: y_pred_future.index, # الفهرس يجب أن يكون datetime (بداية الشهر)
            'forecast': y_pred_future.values,
            'lower_ci': conf_int_future.iloc[:, 0].values,
            'upper_ci': conf_int_future.iloc[:, 1].values
        })

        # منع القيم السالبة
        neg_preds_mask = future_df['forecast'] < 0
        if neg_preds_mask.any():
            logger.info(f"ضبط {neg_preds_mask.sum()} تنبؤات شهرية سالبة إلى 0.")
            future_df.loc[neg_preds_mask, 'forecast'] = 0
        neg_lower_ci_mask = future_df['lower_ci'] < 0
        if neg_lower_ci_mask.any():
            logger.info(f"ضبط {neg_lower_ci_mask.sum()} حدود ثقة سفلية سالبة إلى 0.")
            future_df.loc[neg_lower_ci_mask, 'lower_ci'] = 0
            # التأكد أن الحد العلوي لا يزال أكبر من أو يساوي السفلي
            future_df['upper_ci'] = np.maximum(future_df['lower_ci'], future_df['upper_ci'])

        # تحويل عمود التاريخ إلى تنسيق YYYY-MM للعرض والدمج
        future_df[DATE_COLUMN_OUTPUT] = future_df[DATE_COLUMN_OUTPUT].dt.strftime('%Y-%m')

        logger.info(f"تم توليد التنبؤات الشهرية بنجاح ({time.time() - start_time:.2f} ث).")
        # الفرز حسب التاريخ النصي YYYY-MM
        return future_df.sort_values(DATE_COLUMN_OUTPUT).reset_index(drop=True)

    except Exception as e:
        logger.error(f"!!! خطأ أثناء توليد التنبؤات الشهرية: {e}", exc_info=True)
        return None

# --- **** الدالة الرئيسية الشهرية القابلة للاستدعاء **** ---
def train_and_forecast_monthly(monthly_features_df, forecast_horizon_months=12, seasonal_period=DEFAULT_SEASONAL_PERIOD_MONTHLY):
    """
    تدريب نموذج SARIMAX شهري والتنبؤ للمستقبل لمتوسط قيمة الفاتورة.

    Args:
        monthly_features_df (pd.DataFrame): DataFrame يحتوي على الميزات الشهرية.
        forecast_horizon_months (int): عدد الأشهر المستقبلية للتنبؤ.
        seasonal_period (int): طول الفترة الموسمية (عادة 12 للأشهر).

    Returns:
        pd.DataFrame | None: DataFrame يحتوي على التنبؤات الشهرية المستقبلية
                             (الأعمدة: date, forecast, lower_ci, upper_ci),
                             أو None في حالة الفشل الحاسم.
    """
    logger.info(f"--- بدء دالة train_and_forecast_monthly (Horizon={forecast_horizon_months}, Seasonality={seasonal_period}) ---")
    full_pipeline_start_time = time.time()
    future_predictions_df = None

    try:
        logger.info("--- [M 1/5] تحضير البيانات الشهرية التاريخية ---")
        endog_hist, exog_hist, last_date, exog_cols = _prepare_monthly_forecasting_data(
            monthly_features_df, TARGET_COLUMN_MONTHLY
        )
        if endog_hist is None:
            raise ValueError("فشل تحضير البيانات الشهرية التاريخية.")
        if len(endog_hist) < MIN_OBSERVATIONS_MONTHLY:
            raise ValueError(f"بيانات شهرية تاريخية غير كافية ({len(endog_hist)} نقطة، مطلوب {MIN_OBSERVATIONS_MONTHLY}).")

        logger.info("--- [M 2/5] البحث عن أفضل نموذج شهري ---")
        best_order, best_seasonal_order = _find_best_monthly_sarimax_model(endog_hist, exog_hist, seasonal_period)
        if best_order is None or best_seasonal_order is None:
            raise ValueError("فشل البحث عن أفضل نموذج شهري.")

        logger.info("--- [M 3/5] تدريب النموذج الشهري النهائي ---")
        final_model_results = _train_final_monthly_sarimax_model(endog_hist, exog_hist, best_order, best_seasonal_order)
        if final_model_results is None:
            raise RuntimeError("فشل تدريب النموذج الشهري النهائي.")

        logger.info("--- [M 4/5] إنشاء exog شهري مستقبلي ---")
        future_exog = None
        if exog_cols: # إذا كان هناك exog تاريخي
            future_exog = _create_future_monthly_exog(last_date, forecast_horizon_months, exog_cols, endog_hist)
            if future_exog is None:
                logger.warning("فشل إنشاء future_exog الشهري. سيتم التنبؤ بدون exog مستقبلي إذا كان النموذج يسمح.")
                # النموذج قد لا يتطلب exog للتنبؤ إذا تم تدريبه بدونه
        else:
            logger.info("لا توجد متغيرات خارجية تاريخية، لا حاجة لإنشاء future_exog شهري.")

        logger.info("--- [M 5/5] توليد التنبؤات الشهرية ---")
        future_predictions_df = _generate_monthly_future_forecast(final_model_results, forecast_horizon_months, future_exog)

        if future_predictions_df is None:
            logger.error("فشل توليد التنبؤات الشهرية في الخطوة الأخيرة.")
            # إرجاع DataFrame فارغ للسماح للخطوات التالية بالعمل إن أمكن
            future_predictions_df = pd.DataFrame(columns=[DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'])
        elif future_predictions_df.empty:
            logger.warning("تم توليد DataFrame تنبؤات شهرية فارغ.")
        else:
            logger.info(f"تم توليد {len(future_predictions_df)} أشهر تنبؤات بنجاح.")

    except (ValueError, RuntimeError) as pipeline_error:
         logger.error(f"خطأ مُعالج في خط أنابيب التنبؤ الشهري: {pipeline_error}", exc_info=False)
         # إرجاع DataFrame فارغ في حالة الخطأ المتوقع
         future_predictions_df = pd.DataFrame(columns=[DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'])
    except Exception as general_error:
         logger.error(f"خطأ عام غير متوقع في خط أنابيب التنبؤ الشهري: {general_error}", exc_info=True)
         # إرجاع DataFrame فارغ في حالة الخطأ غير المتوقع
         future_predictions_df = pd.DataFrame(columns=[DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'])

    total_time = time.time() - full_pipeline_start_time
    if future_predictions_df is not None:
        if not future_predictions_df.empty:
             logger.info(f"--- اكتمل train_and_forecast_monthly بنجاح ({total_time:.2f} ث) ---")
        else:
             logger.warning(f"--- اكتمل train_and_forecast_monthly ({total_time:.2f} ث)، لكن الناتج فارغ ---")
    else:
         # هذا لا يجب أن يحدث بسبب معالجة الأخطاء أعلاه، لكن كاحتياط
         logger.error(f"--- فشل train_and_forecast_monthly بشكل غير متوقع ({total_time:.2f} ث) ---")
         future_predictions_df = pd.DataFrame(columns=[DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'])

    return future_predictions_df

# --- جزء الاختبار المستقل (اختياري ومعدّل) ---
if __name__ == "__main__":
    print("\n--- اختبار مستقل لـ monthly_forecasting.py ---")
    # إنشاء بيانات اختبار شهرية وهمية
    dates_test = pd.date_range(start='2020-01-01', periods=36, freq='MS') # 3 سنوات
    data_test = {
        'avg_invoice_value': 100 + np.arange(36) * 2 + np.sin(np.arange(36) * np.pi / 6) * 10 + np.random.randn(36) * 5,
        'month': dates_test.month,
        'quarter': dates_test.quarter,
        'year': dates_test.year
    }
    test_features_df = pd.DataFrame(data_test, index=dates_test)
    test_features_df['avg_invoice_lag_1'] = test_features_df['avg_invoice_value'].shift(1)
    test_features_df['avg_invoice_lag_12'] = test_features_df['avg_invoice_value'].shift(12)
    test_features_df.dropna(inplace=True) # إزالة NaN الناتجة عن اللاجات
    print(f"بيانات الاختبار الشهرية (بعد إزالة NaN اللاجات): {test_features_df.shape}")
    print(test_features_df.head())

    test_forecast_horizon_m = 6 # التنبؤ لـ 6 أشهر قادمة

    try:
        print(f"\nاستدعاء train_and_forecast_monthly (Horizon={test_forecast_horizon_m})...")
        forecast_result_df_m = train_and_forecast_monthly(
            monthly_features_df=test_features_df,
            forecast_horizon_months=test_forecast_horizon_m
        )

        if forecast_result_df_m is not None and not forecast_result_df_m.empty:
            print("\n--- نجح الاختبار الشهري المستقل ---")
            print(f"تم إنشاء {len(forecast_result_df_m)} أشهر تنبؤات.")
            print("\nالنتيجة:")
            print(forecast_result_df_m)
        elif forecast_result_df_m is not None:
            print("\n--- اكتمل الاختبار الشهري، لكن الناتج فارغ ---")
        else: # forecast_result_df_m is None
             print("\n--- فشل الاختبار الشهري المستقل (النتيجة None) ---")

    except Exception as e_main_test_m:
         print(f"\n--- فشل الاختبار الشهري بخطأ: {str(e_main_test_m)} ---")
         traceback.print_exc()
