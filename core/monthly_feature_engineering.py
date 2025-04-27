# core/monthly_feature_engineering.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def generate_monthly_features(monthly_ts: pd.Series, max_lag: int = 12) -> pd.DataFrame | None:
    """
    يُنشئ ميزات زمنية ولاجات لسلسلة زمنية شهرية لمتوسط قيمة الفاتورة.

    Args:
        monthly_ts (pd.Series): السلسلة الزمنية الشهرية (الفهرس يجب أن يكون DatetimeIndex شهري).
        max_lag (int): أقصى فترة تأخر (لاجات) شهرية لإنشائها.

    Returns:
        pandas.DataFrame | None: DataFrame يحتوي على الميزات، جاهز للتنبؤ.
                                أو None في حالة حدوث خطأ أو عدم كفاية البيانات.
    """
    logger.info(f"بدء إنشاء الميزات الشهرية لسلسلة بطول {len(monthly_ts)}...")
    if not isinstance(monthly_ts, pd.Series) or monthly_ts.empty:
        logger.error("السلسلة الزمنية الشهرية المدخلة فارغة أو غير صالحة.")
        return None
    if not isinstance(monthly_ts.index, pd.DatetimeIndex):
        try:
            # محاولة تحويل الفهرس إذا كان نصيًا (مثل YYYY-MM)
            monthly_ts.index = pd.to_datetime(monthly_ts.index)
            logger.info("تم تحويل فهرس السلسلة الشهرية إلى DatetimeIndex.")
            # التأكد من التردد الشهري بعد التحويل
            if pd.infer_freq(monthly_ts.index) not in ('MS', 'M'):
                 monthly_ts = monthly_ts.asfreq('MS') # فرض بداية الشهر
                 logger.warning("تم فرض التردد الشهري 'MS' بعد تحويل الفهرس.")

        except Exception as e_index:
            logger.error(f"فهرس السلسلة الزمنية الشهرية ليس DatetimeIndex ولا يمكن تحويله: {e_index}")
            return None

    # التعامل مع القيم المفقودة في السلسلة الأصلية قبل البدء
    if monthly_ts.isnull().any():
        nan_count_orig = monthly_ts.isnull().sum()
        logger.warning(f"تم العثور على {nan_count_orig} NaN في السلسلة الشهرية الأصلية. ملء بـ ffill->bfill->0.")
        monthly_ts = monthly_ts.ffill().bfill().fillna(0)

    try:
        # إعادة تسمية السلسلة للوضوح وإنشاء DataFrame
        ts_df = monthly_ts.rename('avg_invoice_value').to_frame()

        # --- إضافة ميزات زمنية ---
        ts_df['month'] = ts_df.index.month.astype('int8')
        ts_df['quarter'] = ts_df.index.quarter.astype('int8')
        ts_df['year'] = ts_df.index.year.astype('int16')
        # يمكن إضافة ميزات دورية أكثر تعقيدًا إذا لزم الأمر (مثل sin/cos للشهر)
        # ts_df['month_sin'] = np.sin(2 * np.pi * ts_df['month'] / 12)
        # ts_df['month_cos'] = np.cos(2 * np.pi * ts_df['month'] / 12)
        logger.info("تمت إضافة الميزات الزمنية الشهرية (month, quarter, year).")

        # --- إضافة ميزات اللاج ---
        if max_lag > 0 and len(ts_df) > 1:
            logger.info(f"إضافة ميزات اللاج حتى {max_lag} شهرًا...")
            ts_df = ts_df.sort_index() # التأكد من الفرز
            lags_to_create = range(1, min(max_lag, len(ts_df)) + 1)
            lag_cols_created = []
            for lag in lags_to_create:
                col_name = f'avg_invoice_lag_{lag}'
                ts_df[col_name] = ts_df['avg_invoice_value'].shift(lag)
                lag_cols_created.append(col_name)
            logger.info(f"تمت إضافة {len(lag_cols_created)} ميزات لاج شهرية.")
        else:
            logger.warning("تخطي ميزات اللاج الشهرية (max_lag=0 أو طول السلسلة < 2).")

        # --- التنظيف النهائي ---
        # لا نملأ NaN اللاجات هنا، النموذج سيتعامل معها أو نحذفها

        # تحويل الأنواع لتوفير الذاكرة
        for col in ts_df.select_dtypes(include=['float64']).columns:
            ts_df[col] = ts_df[col].astype('float32')
        for col in ts_df.select_dtypes(include=['int64', 'Int64']).columns:
            # السنة قد تكون كبيرة
            if col != 'year':
                 try:
                     ts_df[col] = pd.to_numeric(ts_df[col], downcast='integer')
                 except Exception: pass
        # تأكد من أن الأعمدة المنطقية المحتملة (إذا أضيفت) هي int8

        # إزالة الصفوف الأولى التي تحتوي على NaN بسبب اللاجات (ضروري لـ SARIMAX)
        original_len = len(ts_df)
        ts_df.dropna(inplace=True) # حذف أي صف يحتوي على NaN (بسبب اللاجات بشكل أساسي)
        rows_dropped = original_len - len(ts_df)
        if rows_dropped > 0:
            logger.info(f"تم حذف {rows_dropped} صفوف شهرية بسبب NaN (عادةً من اللاجات الأولى).")

        if ts_df.empty:
            logger.error("أصبح DataFrame الميزات الشهرية فارغًا بعد حذف NaN اللاجات.")
            return None

        logger.info(f"اكتمل إنشاء الميزات الشهرية. الشكل النهائي: {ts_df.shape}")
        return ts_df

    except Exception as e:
        logger.error(f"خطأ عام أثناء إنشاء الميزات الشهرية: {e}", exc_info=True)
        return None
