# forecasting.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm
from datetime import timedelta, datetime
import holidays
import logging
import traceback
import time
import warnings
import os # إضافة استيراد os إذا لزم الأمر لدوال أخرى (مثل حفظ الملفات في الاختبار)

# إعداد Logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# تجاهل تحذيرات معينة (اختياري)
warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information was") # لـ statsmodels

# --- الثوابت ---
TARGET_COLUMN = 'daily_sales'
# هذا الثابت لا يزال يستخدم لتحديد العمود في البيانات المدخلة
DATE_COLUMN_INPUT = 'sale_date'
# اسم العمود القياسي للتاريخ في DataFrame الناتج عن التنبؤ
DATE_COLUMN_OUTPUT = 'date' # <-- هذا هو الاسم الجديد المستخدم داخليا وعند الإرجاع
LEAKAGE_FEATURES = ['transaction_count', 'total_items']
DEFAULT_SEASONAL_PERIOD = 7
AUTO_ARIMA_MAX_P = 3
AUTO_ARIMA_MAX_Q = 3
AUTO_ARIMA_MAX_P_SEASONAL = 2
AUTO_ARIMA_MAX_Q_SEASONAL = 2
CONFIDENCE_ALPHA = 0.05 # 95% CI

# --- الدوال المساعدة ---

def _get_saudi_holidays(start_year, end_year):
    """ للحصول على قاموس الإجازات السعودية لفترة محددة """
    try:
        return holidays.SaudiArabia(years=range(start_year, end_year + 1))
    except Exception as e:
        logger.warning(f"لم يتم تحميل الإجازات السعودية للسنوات {start_year}-{end_year}: {e}")
        return holidays.HolidayBase()

def _prepare_forecasting_data(features_df, date_col_input, target_col): # استخدام date_col_input هنا
    """ تحضير البيانات من DataFrame الميزات المدخل. """
    logger.info(f"بدء تحضير بيانات التنبؤ من DataFrame بالشكل: {features_df.shape}")
    start_time = time.time()
    try:
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("DataFrame الميزات المدخل فارغ أو غير صالح.")

        df = features_df.copy()

        if date_col_input not in df.columns: raise ValueError(f"عمود التاريخ '{date_col_input}' مفقود.") # استخدام date_col_input
        if target_col not in df.columns: raise ValueError(f"عمود الهدف '{target_col}' مفقود.")

        # 1. معالجة عمود التاريخ والفهرس
        try:
            df[date_col_input] = pd.to_datetime(df[date_col_input]) # استخدام date_col_input
            df = df.set_index(date_col_input).sort_index() # استخدام date_col_input
        except Exception as e_date:
            raise ValueError(f"فشل معالجة عمود التاريخ '{date_col_input}': {e_date}")

        min_date, max_date = df.index.min(), df.index.max()
        logger.info(f"فترة البيانات التاريخية المدخلة: من {min_date.date()} إلى {max_date.date()}")

        # 2. التحقق من التردد اليومي وفرضه إذا لزم الأمر
        if len(df) < 2:
             logger.warning("البيانات تحتوي على أقل من نقطتين زمنيتين، لا يمكن فرض أو استنتاج التردد.")
        else:
             inferred_freq = pd.infer_freq(df.index)
             if inferred_freq != 'D':
                 logger.warning(f"التردد المستنتج ليس يوميًا ('{inferred_freq}'). فرض التردد 'D'.")
                 original_len = len(df)
                 try:
                     df = df.asfreq('D')
                     logger.info(f"تم فرض التردد 'D'. الطول الجديد: {len(df)} (كان {original_len}).")
                 except Exception as e_asfreq:
                     logger.error(f"فشل فرض التردد 'D': {e_asfreq}. المتابعة بالبيانات الأصلية.")
             else:
                 logger.info("تم التأكد من التردد اليومي 'D'.")


        # 3. معالجة الهدف (endog) وملء القيم المفقودة بعد asfreq
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        nan_in_target = df[target_col].isnull().sum()
        if nan_in_target > 0:
            logger.warning(f"تم العثور على {nan_in_target} قيم NaN في عمود الهدف '{target_col}'. سيتم ملؤها بـ ffill -> bfill -> 0.")
            df[target_col] = df[target_col].ffill().bfill().fillna(0)

        if df.empty: raise ValueError("لا توجد بيانات صالحة بعد معالجة عمود الهدف وفحص التردد.")
        endog = df[target_col].astype('float32')

        # 4. تحديد ومعالجة المتغيرات الخارجية (exog)
        potential_exog_cols = df.columns.drop(target_col, errors='ignore').tolist()
        exog_cols = [col for col in potential_exog_cols if col not in LEAKAGE_FEATURES]
        exog = None
        exog_cols_list = [] # القائمة النهائية للأعمدة المستخدمة

        if exog_cols:
            exog = df[exog_cols].copy()
            logger.info(f"تم تحديد {len(exog_cols)} متغير خارجي (Exog) مبدئي: {exog_cols}")

            cols_to_fill = []
            numeric_exog_cols = []
            for col in exog.columns:
                 if np.isinf(exog[col]).any():
                      inf_count = np.isinf(exog[col]).sum()
                      logger.warning(f"العمود '{col}' في exog يحتوي على {inf_count} قيم لانهائية (inf). سيتم استبدالها بـ NaN ثم ملؤها.")
                      exog[col] = exog[col].replace([np.inf, -np.inf], np.nan)

                 if not pd.api.types.is_numeric_dtype(exog[col]):
                     logger.warning(f"تحويل العمود غير الرقمي '{col}' في exog إلى رقمي (fillna=0).")
                     exog[col] = pd.to_numeric(exog[col], errors='coerce')
                     if exog[col].isnull().any(): cols_to_fill.append(col)
                 elif exog[col].isnull().any():
                     cols_to_fill.append(col)

                 if pd.api.types.is_numeric_dtype(exog[col]):
                      numeric_exog_cols.append(col)
                 else:
                      logger.warning(f"العمود '{col}' لا يزال غير رقمي بعد محاولة التحويل. سيتم تجاهله.")

            if len(numeric_exog_cols) < len(exog.columns):
                 dropped_non_numeric = list(set(exog.columns) - set(numeric_exog_cols))
                 logger.warning(f"تم تجاهل الأعمدة غير الرقمية التالية من exog: {dropped_non_numeric}")
                 exog = exog[numeric_exog_cols]

            cols_to_fill_numeric = [col for col in cols_to_fill if col in exog.columns]
            if cols_to_fill_numeric:
                 logger.warning(f"ملء القيم المفقودة (NaN) في أعمدة exog التالية بـ 0: {cols_to_fill_numeric}")
                 exog[cols_to_fill_numeric] = exog[cols_to_fill_numeric].fillna(0)

            if not exog.empty:
                if exog.isnull().values.any():
                     nan_cols = exog.columns[exog.isnull().any()].tolist()
                     raise ValueError(f"NaN متبقية في المتغيرات الخارجية: {nan_cols}")
                if np.isinf(exog.values).any():
                     inf_cols = exog.columns[np.isinf(exog).any()].tolist()
                     raise ValueError(f"قيم لانهائية متبقية في المتغيرات الخارجية: {inf_cols}")

                exog = exog.astype('float32')
                exog_cols_list = exog.columns.tolist()
                logger.info(f"تم تحديد {len(exog_cols_list)} متغير خارجي (Exog) رقمي نهائي.")
            else:
                 logger.warning("لم يتبق أي متغيرات خارجية (Exog) رقمية بعد المعالجة.")
                 exog = None
        else:
            logger.warning("لم يتم العثور على أي متغيرات خارجية (Exog) محتملة للاستخدام.")


        last_known_date = df.index.max()
        logger.info(f"اكتمل تحضير البيانات في {time.time() - start_time:.2f} ثانية. Endog: {endog.shape}, Exog: {exog.shape if exog is not None else 'None'}")
        return endog, exog, last_known_date, exog_cols_list

    except ValueError as ve:
        logger.error(f"خطأ في بيانات التنبؤ: {ve}", exc_info=True)
        return None, None, None, None
    except Exception as e:
        logger.error(f"خطأ غير متوقع أثناء تحضير بيانات التنبؤ: {e}", exc_info=True)
        return None, None, None, None

def _create_future_exog(last_known_date, horizon, exog_historical_cols, endog_historical):
    """ إنشاء DataFrame للمتغيرات الخارجية المستقبلية. """
    logger.info(f"إنشاء متغيرات خارجية مستقبلية لـ {horizon} أيام بعد {last_known_date.date()}...")
    if not exog_historical_cols:
        logger.info("لا توجد أعمدة exog تاريخية، لا حاجة لإنشاء exog مستقبلي.")
        return None

    future_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=horizon, freq='D')
    future_exog_df = pd.DataFrame(index=future_dates)
    min_future_year = future_dates.min().year
    max_future_year = future_dates.max().year
    saudi_holidays_future = _get_saudi_holidays(min_future_year, max_future_year)

    time_features_created = []
    known_time_features = ['day_of_week', 'is_weekend', 'month', 'quarter', 'day_of_month',
                           'week_of_year', 'is_holiday', 'is_month_start', 'is_month_end',
                           'day_of_year', 'year']
    for feature in known_time_features:
        if feature in exog_historical_cols:
            try:
                if feature == 'day_of_week': future_exog_df[feature] = future_exog_df.index.dayofweek.astype('int8')
                elif feature == 'is_weekend': future_exog_df[feature] = future_exog_df.index.dayofweek.isin([4, 5]).astype('int8')
                elif feature == 'month': future_exog_df[feature] = future_exog_df.index.month.astype('int8')
                elif feature == 'quarter': future_exog_df[feature] = future_exog_df.index.quarter.astype('int8')
                elif feature == 'day_of_month': future_exog_df[feature] = future_exog_df.index.day.astype('int8')
                elif feature == 'week_of_year':
                    try: future_exog_df[feature] = future_exog_df.index.isocalendar().week.astype('uint8')
                    except AttributeError: future_exog_df[feature] = future_exog_df.index.week.astype('uint8')
                elif feature == 'is_holiday': future_exog_df[feature] = future_exog_df.index.to_series().dt.date.apply(lambda x: 1 if x in saudi_holidays_future else 0).astype('int8')
                elif feature == 'is_month_start': future_exog_df[feature] = future_exog_df.index.is_month_start.astype('int8')
                elif feature == 'is_month_end': future_exog_df[feature] = future_exog_df.index.is_month_end.astype('int8')
                elif feature == 'day_of_year': future_exog_df[feature] = future_exog_df.index.dayofyear.astype('int16')
                elif feature == 'year': future_exog_df[feature] = future_exog_df.index.year.astype('int16')
                time_features_created.append(feature)
            except Exception as e_time_feat:
                 logger.error(f"خطأ أثناء إنشاء ميزة الوقت المستقبلية '{feature}': {e_time_feat}")

    logger.info(f"تم إنشاء ميزات الوقت المستقبلية: {time_features_created}")

    lag_cols = [col for col in exog_historical_cols if col.startswith('sales_lag_')]
    if lag_cols:
         logger.info("إنشاء ميزات اللاج المستقبلية بناءً على القيم الأخيرة للسلسلة التاريخية endog.")
         if endog_historical is None or endog_historical.empty or len(endog_historical) < 1:
              logger.error("لا يمكن إنشاء اللاجات المستقبلية، السلسلة التاريخية endog فارغة أو قصيرة جدًا.")
              for col_name in lag_cols: future_exog_df[col_name] = 0.0
         else:
              endog_hist_sorted = endog_historical.sort_index()
              max_lag_needed = 0
              try: max_lag_needed = max(int(col.split('_lag_')[-1]) for col in lag_cols)
              except ValueError: logger.warning("خطأ في استخلاص أرقام اللاج.")

              if len(endog_hist_sorted) < max_lag_needed:
                    logger.warning(f"طول endog ({len(endog_hist_sorted)}) أصغر من أقصى لاج ({max_lag_needed}). قد تكون اللاجات غير دقيقة.")

              for col_name in lag_cols:
                  try:
                      lag_num = int(col_name.split('_lag_')[-1])
                      if lag_num <= 0: continue
                      if lag_num <= len(endog_hist_sorted):
                           future_exog_df[col_name] = endog_hist_sorted.iloc[-lag_num]
                      else:
                           logger.warning(f" لا يمكن إنشاء '{col_name}' (لاج={lag_num}). ملء بـ 0.")
                           future_exog_df[col_name] = 0.0
                  except (ValueError, IndexError, TypeError) as e_lag:
                       logger.warning(f" خطأ في معالجة اللاج '{col_name}': {e_lag}. ملء بـ 0.")
                       future_exog_df[col_name] = 0.0

    missing_cols = [col for col in exog_historical_cols if col not in future_exog_df.columns]
    if missing_cols:
         logger.warning(f"أعمدة exog التالية مطلوبة ولكن لم تُنشأ للمستقبل (ملء بـ 0): {missing_cols}")
         for col in missing_cols: future_exog_df[col] = 0.0

    try:
        future_exog_df = future_exog_df.reindex(columns=exog_historical_cols, fill_value=0.0)
        future_exog_df = future_exog_df.astype('float32')
        if future_exog_df.isnull().values.any():
            nan_cols_final = future_exog_df.columns[future_exog_df.isnull().any()].tolist()
            logger.error(f"*** خطأ: NaN متبقية في future_exog: {nan_cols_final}. ملء بـ 0.")
            future_exog_df = future_exog_df.fillna(0.0)
        if np.isinf(future_exog_df.values).any():
             inf_cols_final = future_exog_df.columns[np.isinf(future_exog_df).any()].tolist()
             logger.error(f"*** خطأ: قيم لانهائية (inf) متبقية في future_exog: {inf_cols_final}. ملء بـ 0.")
             future_exog_df = future_exog_df.replace([np.inf, -np.inf], 0.0)
    except Exception as e_reindex:
        logger.error(f"خطأ أثناء إعادة فهرسة/تحويل future_exog: {e_reindex}", exc_info=True)
        return None

    logger.info(f"تم إنشاء DataFrame للمتغيرات الخارجية المستقبلية بالشكل: {future_exog_df.shape}")
    return future_exog_df

def _find_best_sarimax_model(endog_hist, exog_hist, seasonal_period):
    """ البحث عن أفضل معاملات SARIMAX. """
    logger.info(f"بدء البحث عن أفضل نموذج SARIMAX (m={seasonal_period})...")
    start_time = time.time()
    min_obs_for_seasonal = 2 * seasonal_period
    if endog_hist is None or len(endog_hist) < min_obs_for_seasonal:
        logger.error(f"السلسلة endog قصيرة جدًا ({len(endog_hist) if endog_hist is not None else 0}) للبحث الموسمي (m={seasonal_period}).")
        return None, None

    exog_for_search = None
    if exog_hist is not None:
        if isinstance(exog_hist, pd.DataFrame) and not exog_hist.empty and endog_hist.index.equals(exog_hist.index) and not exog_hist.isnull().values.any() and not np.isinf(exog_hist.values).any():
            logger.info("استخدام exog في البحث.")
            exog_for_search = exog_hist.astype('float32')
        else:
            logger.warning("تجاهل exog في البحث (عدم تطابق/فراغ/NaN/inf).")

    try:
        auto_model = pm.auto_arima(y=endog_hist,
                                   exogenous=exog_for_search,
                                   start_p=1, start_q=1,
                                   max_p=AUTO_ARIMA_MAX_P, max_q=AUTO_ARIMA_MAX_Q,
                                   m=seasonal_period,
                                   start_P=0, start_Q=0,
                                   max_P=AUTO_ARIMA_MAX_P_SEASONAL, max_Q=AUTO_ARIMA_MAX_Q_SEASONAL,
                                   seasonal=True,
                                   d=None, D=None,
                                   test='adf', seasonal_test='ocsb',
                                   trace=False, error_action='ignore', suppress_warnings=True,
                                   stepwise=True, n_jobs=-1, maxiter=100,
                                   information_criterion='aic')

        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        logger.info(f"auto_arima انتهى ({time.time() - start_time:.2f} ث).")
        logger.info(f"أفضل المعاملات: Order={best_order}, Seasonal={best_seasonal_order}")
        return best_order, best_seasonal_order

    except Exception as e:
        logger.error(f"خطأ auto_arima: {e}", exc_info=True)
        logger.warning(f"استخدام معاملات افتراضية: (1,1,1)(1,0,1,{seasonal_period}).")
        return (1, 1, 1), (1, 0, 1, seasonal_period)

def _train_final_sarimax_model(endog_hist, exog_hist, order, seasonal_order):
    """ تدريب نموذج SARIMAX النهائي. """
    logger.info(f"بدء تدريب النموذج النهائي بـ Order={order}, Seasonal={seasonal_order}...")
    start_time = time.time()
    if order is None or seasonal_order is None:
        logger.error("المعاملات غير متوفرة.")
        return None

    exog_for_train = None
    if exog_hist is not None:
        if isinstance(exog_hist, pd.DataFrame) and not exog_hist.empty and endog_hist.index.equals(exog_hist.index) and not exog_hist.isnull().values.any() and not np.isinf(exog_hist.values).any():
             exog_for_train = exog_hist.astype('float32')
             logger.info(f" استخدام {exog_for_train.shape[1]} exog في التدريب.")
        else:
             logger.warning(" تجاهل exog في التدريب.")

    try:
        endog_train = endog_hist.astype('float32')
        model = sm.tsa.SARIMAX(endog=endog_train,
                               exog=exog_for_train,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=200)
        logger.info(f"تم تدريب النموذج بنجاح ({time.time() - start_time:.2f} ث).")
        return results
    except np.linalg.LinAlgError as lae:
         logger.error(f"!!! خطأ جبر خطي أثناء التدريب: {lae}", exc_info=True)
         return None
    except ValueError as ve:
         logger.error(f"!!! خطأ قيمة أثناء التدريب: {ve}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"!!! خطأ تدريب النموذج النهائي: {e}", exc_info=True)
        return None

def _generate_future_forecast(model_results, steps, future_exog):
    """ توليد التنبؤات المستقبلية وفترات الثقة. """
    if model_results is None:
        logger.error("النموذج المدرب غير متوفر.")
        return None

    logger.info(f"بدء توليد التنبؤات لـ {steps} خطوات...")
    start_time = time.time()

    final_future_exog = None
    model_needs_exog = model_results.model.exog is not None
    model_exog_names = getattr(model_results.model, 'exog_names', [])

    if model_needs_exog:
        logger.info("النموذج يتطلب exog للتنبؤ.")
        if future_exog is None or not isinstance(future_exog, pd.DataFrame) or future_exog.empty:
            logger.error("future_exog مطلوب وغير متوفر/صالح.")
            return None
        if len(future_exog) != steps:
            logger.error(f"طول future_exog ({len(future_exog)}) لا يطابق steps ({steps}).")
            return None
        if future_exog.isnull().values.any() or np.isinf(future_exog.values).any():
             logger.error("future_exog يحتوي على NaN أو inf.")
             return None
        try:
            if set(future_exog.columns) != set(model_exog_names):
                 logger.warning(f"أعمدة future_exog تختلف. محاولة إعادة الترتيب...")
                 final_future_exog = future_exog.reindex(columns=model_exog_names).copy()
            else:
                final_future_exog = future_exog.copy()
            final_future_exog = final_future_exog.astype('float32')
        except Exception as e_exog_prep:
            logger.error(f"خطأ تحضير future_exog: {e_exog_prep}", exc_info=True)
            return None
    elif future_exog is not None:
         logger.warning("تجاهل future_exog (النموذج لا يتطلبه).")

    try:
        forecast_obj = model_results.get_forecast(steps=steps, exog=final_future_exog, alpha=CONFIDENCE_ALPHA)
        y_pred_future = forecast_obj.predicted_mean
        conf_int_future = forecast_obj.conf_int(alpha=CONFIDENCE_ALPHA)

        # --- *** استخدام DATE_COLUMN_OUTPUT هنا *** ---
        future_df = pd.DataFrame({
            DATE_COLUMN_OUTPUT: y_pred_future.index, # الفهرس هو datetime
            'forecast': y_pred_future.values,
            'lower_ci': conf_int_future.iloc[:, 0].values,
            'upper_ci': conf_int_future.iloc[:, 1].values
        })
        # ---------------------------------------------

        neg_preds_mask = future_df['forecast'] < 0
        if neg_preds_mask.any(): future_df.loc[neg_preds_mask, 'forecast'] = 0
        neg_lower_ci_mask = future_df['lower_ci'] < 0
        if neg_lower_ci_mask.any():
            future_df.loc[neg_lower_ci_mask, 'lower_ci'] = 0
            future_df['upper_ci'] = np.maximum(future_df['lower_ci'], future_df['upper_ci'])

        # --- *** استخدام DATE_COLUMN_OUTPUT هنا *** ---
        future_df[DATE_COLUMN_OUTPUT] = future_df[DATE_COLUMN_OUTPUT].dt.date
        # ---------------------------------------------

        logger.info(f"تم توليد التنبؤات بنجاح ({time.time() - start_time:.2f} ث).")
        # --- *** استخدام DATE_COLUMN_OUTPUT هنا *** ---
        return future_df.sort_values(DATE_COLUMN_OUTPUT).reset_index(drop=True)
        # ---------------------------------------------

    except Exception as e:
        logger.error(f"!!! خطأ توليد التنبؤات: {e}", exc_info=True)
        return None


# --- **** الدالة الرئيسية الجديدة القابلة للاستدعاء **** ---
def train_and_forecast(features_df, forecast_horizon=14, seasonal_period=DEFAULT_SEASONAL_PERIOD):
    """ تدريب SARIMAX والتنبؤ للمستقبل. """
    logger.info(f"--- بدء دالة train_and_forecast (Horizon={forecast_horizon}, Seasonality={seasonal_period}) ---")
    full_pipeline_start_time = time.time()
    future_predictions_df = None

    try:
        logger.info("--- [1/5] تحضير البيانات التاريخية ---")
        # --- *** استخدام DATE_COLUMN_INPUT هنا *** ---
        endog_hist, exog_hist, last_date, exog_cols = _prepare_forecasting_data(
            features_df, DATE_COLUMN_INPUT, TARGET_COLUMN
        )
        # -----------------------------------------
        if endog_hist is None: raise ValueError("فشل تحضير البيانات.")
        if len(endog_hist) < max(seasonal_period * 2, 10):
             raise ValueError(f"بيانات تاريخية غير كافية ({len(endog_hist)} نقطة).")

        logger.info("--- [2/5] البحث عن أفضل نموذج ---")
        best_order, best_seasonal_order = _find_best_sarimax_model(endog_hist, exog_hist, seasonal_period)
        if best_order is None: raise ValueError("فشل البحث عن النموذج.")

        logger.info("--- [3/5] تدريب النموذج النهائي ---")
        final_model_results = _train_final_sarimax_model(endog_hist, exog_hist, best_order, best_seasonal_order)
        if final_model_results is None: raise RuntimeError("فشل تدريب النموذج.")

        logger.info("--- [4/5] إنشاء exog مستقبلي ---")
        future_exog = None
        if exog_cols:
            future_exog = _create_future_exog(last_date, forecast_horizon, exog_cols, endog_hist)
            if future_exog is None: raise ValueError("فشل إنشاء future_exog.")
        else: logger.info("لا حاجة لإنشاء future_exog.")

        logger.info("--- [5/5] توليد التنبؤات ---")
        future_predictions_df = _generate_future_forecast(final_model_results, forecast_horizon, future_exog)
        if future_predictions_df is None:
            logger.error("فشل توليد التنبؤات.")
            # إرجاع فارغ للسماح للداشبورد بالعمل
            future_predictions_df = pd.DataFrame(columns=[DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'])
        elif future_predictions_df.empty: logger.warning("تم توليد DataFrame تنبؤات فارغ.")
        else: logger.info(f"تم توليد {len(future_predictions_df)} يوم تنبؤات.")

    except (ValueError, RuntimeError) as pipeline_error:
         logger.error(f"خطأ مُعالج في خط أنابيب التنبؤ: {pipeline_error}", exc_info=False)
         future_predictions_df = None
    except Exception as general_error:
         logger.error(f"خطأ عام غير متوقع في خط أنابيب التنبؤ: {general_error}", exc_info=True)
         future_predictions_df = None

    total_time = time.time() - full_pipeline_start_time
    if future_predictions_df is not None:
        if not future_predictions_df.empty: logger.info(f"--- اكتمل train_and_forecast بنجاح ({total_time:.2f} ث) ---")
        else: logger.warning(f"--- اكتمل train_and_forecast ({total_time:.2f} ث)، لكن الناتج فارغ ---")
        return future_predictions_df
    else:
         logger.error(f"--- فشل train_and_forecast بشكل حاسم ({total_time:.2f} ث) ---")
         return None


# --- جزء الاختبار المستقل (اختياري ومعدّل) ---
if __name__ == "__main__":
    print("--- اختبار مستقل لـ forecasting.py ---")
    test_data_paths_fc = {
        'sale_invoices': r'C:\Users\sheee\Downloads\ZodData\sale_invoices.xlsx',
        'sale_invoices_details': r'C:\Users\sheee\Downloads\ZodData\sale_invoices_details.xlsx'
    }
    test_forecast_horizon = 14

    try:
        from feature_engineering import generate_features_df
        print("استدعاء generate_features_df...")
        features_for_test = generate_features_df(test_data_paths_fc)

        if features_for_test is not None and not features_for_test.empty:
            print(f"تم إنشاء الميزات. الشكل: {features_for_test.shape}")
            print(f"استدعاء train_and_forecast...")
            forecast_result_df = train_and_forecast(
                features_df=features_for_test,
                forecast_horizon=test_forecast_horizon
            )

            if forecast_result_df is not None and not forecast_result_df.empty:
                print("\n--- نجح الاختبار المستقل ---")
                print(f"تم إنشاء {len(forecast_result_df)} يوم تنبؤات.")
                print("\nعينة:")
                print(forecast_result_df.head())
                try:
                     output_filename = "future_forecasts_standalone_test.csv"
                     forecast_result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                     print(f"\nتم الحفظ في: {output_filename}")
                except Exception as e_save: print(f"\nخطأ الحفظ: {e_save}")
            elif forecast_result_df is not None: print("\n--- اكتمل الاختبار، لكن الناتج فارغ ---")
            else: print("\n--- فشل الاختبار المستقل ---")
        else: print("\n--- فشل إنشاء الميزات، لا يمكن الاختبار ---")
    except ImportError: print("\n--- خطأ: لم يتم العثور على `feature_engineering.py`. ---")
    except Exception as e_main_test:
         print(f"\n--- فشل الاختبار بخطأ: {str(e_main_test)} ---")
         traceback.print_exc()