# core/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import logging
import warnings
import os
from typing import Dict # <-- إضافة Dict

# إعداد Logger الأساسي
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format' is deprecated")

def _parse_excel_date_local(date_val):
    """يحاول تحليل التاريخ من الأرقام أو النصوص (نسخة محلية)."""
    if pd.isna(date_val): return pd.NaT
    try:
        if isinstance(date_val, (datetime, pd.Timestamp)): # تحقق من الأنواع الشائعة أولاً
            return pd.to_datetime(date_val)
        elif isinstance(date_val, (int, float)):
            # التعامل مع أرقام Excel التسلسلية
            if 1 < date_val < 2958466: # نطاق تقريبي
                 if date_val == 60: return pd.Timestamp('1900-02-29') # حالة خاصة
                 excel_epoch = pd.Timestamp('1899-12-30')
                 return excel_epoch + pd.to_timedelta(date_val, unit='D')
            else: return pd.NaT # خارج نطاق Excel
        elif isinstance(date_val, str):
            date_val = date_val.strip()
            if not date_val: return pd.NaT
            # محاولة تحليل مباشر للنصوص الشائعة
            return pd.to_datetime(date_val, errors='coerce', dayfirst=None)
        else:
            # أنواع أخرى
            return pd.to_datetime(date_val, errors='coerce')
    except Exception:
        return pd.NaT

class _SalesFeatureEngineerInternal:
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.logger = logging.getLogger(__name__)
        # --- التحقق من البيانات المدخلة ---
        if not isinstance(data_dict, dict):
            raise TypeError("Input must be a dictionary of DataFrames.")
        if 'sale_invoices' not in data_dict or 'sale_invoices_details' not in data_dict:
            raise ValueError("Input dictionary missing 'sale_invoices' or 'sale_invoices_details'.")
        if not isinstance(data_dict['sale_invoices'], pd.DataFrame) or not isinstance(data_dict['sale_invoices_details'], pd.DataFrame):
             raise TypeError("Values in input dictionary must be pandas DataFrames.")

        # --- استخدام النسخ لتجنب تعديل البيانات الأصلية ---
        self.data = {
            'sale_invoices': data_dict['sale_invoices'].copy(),
            'sale_invoices_details': data_dict['sale_invoices_details'].copy()
        }
        self.features = None
        self.saudi_holidays = self._get_saudi_holidays()
        self.logger.info(f"Feature Engineer initialized with {len(self.data['sale_invoices'])} invoices and {len(self.data['sale_invoices_details'])} details.")

    # --- إزالة دالة _load_data ---

    def _get_saudi_holidays(self):
        """Gets Saudi holidays for a relevant year range."""
        current_year = datetime.now().year
        years = range(max(2020, current_year - 3), current_year + 3) # Range for holidays
        try:
            return holidays.SaudiArabia(years=years)
        except Exception as e:
            self.logger.warning(f"Could not load Saudi holidays for years {min(years)}-{max(years)}: {e}. Using empty list.")
            return holidays.HolidayBase() # Return empty holiday object

    def _preprocess_dates(self):
        """Preprocesses the date column in the 'sale_invoices' DataFrame."""
        table = 'sale_invoices'
        if self.data[table].empty:
            self.logger.warning(f"'{table}' DataFrame is empty, cannot preprocess dates.")
            # Raise error as this is likely critical
            raise ValueError(f"'{table}' DataFrame is empty, date preprocessing failed.")

        # --- تحديد عمود التاريخ من الإعدادات ---
        # افترض أن عمود التاريخ تم تحديده في config.py
        from core.config import config # استيراد داخل الدالة لتجنب مشاكل الاستيراد الدائري المحتملة
        date_col = config.date_settings['date_columns']['sale_invoices'][0] # e.g., 'created_at'

        if date_col not in self.data[table].columns:
            self.logger.error(f"Configured date column '{date_col}' not found in '{table}'. Available: {self.data[table].columns.tolist()}")
            raise ValueError(f"Date column '{date_col}' missing in '{table}'.")

        self.logger.info(f"Preprocessing date column '{date_col}' in '{table}'. Initial rows: {len(self.data[table])}")

        self.data[table]['processed_date_dt'] = pd.to_datetime(self.data[table][date_col], errors='coerce')

        total_rows_before_drop = len(self.data[table])
        null_dates_count = self.data[table]['processed_date_dt'].isnull().sum()
        self.logger.info(f"After initial date conversion: Total rows = {total_rows_before_drop}, Invalid/Null dates (NaT) = {null_dates_count}")

        if null_dates_count == total_rows_before_drop:
             self.logger.error(f"Failed to parse any dates in column '{date_col}'. Check data formats.")
             raise ValueError(f"No valid dates found in '{table}'.")

        if null_dates_count > 0:
             self.data[table] = self.data[table].dropna(subset=['processed_date_dt'])
             self.logger.warning(f"Dropped {null_dates_count} rows due to invalid dates (NaT). Remaining rows: {len(self.data[table])}")

        if self.data[table].empty:
             self.logger.error(f"No rows remaining in '{table}' after date processing and dropping invalid dates.")
             raise ValueError(f"No valid dates found in '{table}'.")

        # تحويل إلى تاريخ فقط (date object)
        self.data[table]['processed_date'] = self.data[table]['processed_date_dt'].dt.date
        unique_dates_after_processing = self.data[table]['processed_date'].nunique()
        self.logger.info(f"Date preprocessing for '{table}' complete. Unique dates remaining: {unique_dates_after_processing}")
        # حذف العمود المؤقت datetime
        self.data[table] = self.data[table].drop(columns=['processed_date_dt'])

    def _create_daily_sales(self):
        """Aggregates sales data to a daily level."""
        try:
            details_table = 'sale_invoices_details'
            invoices_table = 'sale_invoices'

            if self.data[details_table].empty:
                raise ValueError(f"'{details_table}' data is empty.")
            if self.data[invoices_table].empty or 'processed_date' not in self.data[invoices_table].columns:
                 raise ValueError(f"'{invoices_table}' data is empty or missing processed date column.")

            # --- تحديد الأعمدة من الإعدادات ---
            from core.config import config # استيراد داخل الدالة
            # Details columns
            amount_col = config.required_columns['sale_invoices_details'][4] # totalPrice
            details_invoice_id_col = config.required_columns['sale_invoices_details'][2] # invoice_id
            qty_col = config.required_columns['sale_invoices_details'][3] # quantity
            # Invoices columns
            invoices_id_col = config.required_columns['sale_invoices'][0] # id

            # --- التحقق من وجود الأعمدة ---
            if amount_col not in self.data[details_table].columns: raise KeyError(f"Amount column '{amount_col}' missing in details.")
            if details_invoice_id_col not in self.data[details_table].columns: raise KeyError(f"Invoice ID column '{details_invoice_id_col}' missing in details.")
            if invoices_id_col not in self.data[invoices_table].columns: raise KeyError(f"Invoice ID column '{invoices_id_col}' missing in invoices.")
            if qty_col not in self.data[details_table].columns:
                 self.logger.warning(f"Quantity column '{qty_col}' missing, using default 1 for item count.")
                 self.data[details_table]['quantity_default'] = 1; qty_col = 'quantity_default'

            self.logger.info(f"Aggregating daily sales using amount='{amount_col}', quantity='{qty_col}'. Linking details:'{details_invoice_id_col}' to invoices:'{invoices_id_col}'.")

            # --- تحضير للدمج ---
            df_details = self.data[details_table][[details_invoice_id_col, amount_col, qty_col]].copy()
            df_invoices = self.data[invoices_table][[invoices_id_col, 'processed_date']].copy()

            # تحويل الأعمدة الرقمية المطلوبة قبل التجميع
            df_details[amount_col] = pd.to_numeric(df_details[amount_col], errors='coerce').fillna(0)
            df_details[qty_col] = pd.to_numeric(df_details[qty_col], errors='coerce').fillna(0)


            # تحويل أعمدة الربط إلى نوع مشترك (نصي هو الأكثر أمانًا) لتجنب أخطاء الدمج
            df_details[details_invoice_id_col] = df_details[details_invoice_id_col].astype(str)
            df_invoices[invoices_id_col] = df_invoices[invoices_id_col].astype(str)

            # حذف القيم المفقودة في أعمدة الربط قبل الدمج
            initial_details_rows = len(df_details)
            initial_invoices_rows = len(df_invoices)
            df_details.dropna(subset=[details_invoice_id_col], inplace=True)
            df_invoices.dropna(subset=[invoices_id_col, 'processed_date'], inplace=True)
            if len(df_details) < initial_details_rows: logger.warning(f"Dropped {initial_details_rows - len(df_details)} detail rows with missing invoice ID.")
            if len(df_invoices) < initial_invoices_rows: logger.warning(f"Dropped {initial_invoices_rows - len(df_invoices)} invoice rows with missing ID or date.")

            self.logger.info(f"Merging details ({len(df_details)} rows) with invoices ({len(df_invoices)} rows).")

            # دمج باستخدام left join للحفاظ على كل التفاصيل التي لها تاريخ فاتورة صالح
            merged = pd.merge(
                df_details,
                df_invoices,
                left_on=details_invoice_id_col,
                right_on=invoices_id_col,
                how='left'
            )
            self.logger.info(f"Rows after merge: {len(merged)}")

            # حذف التفاصيل التي لم تجد تاريخ فاتورة مطابق
            null_dates_after_merge = merged['processed_date'].isnull().sum()
            if null_dates_after_merge > 0:
                 self.logger.warning(f"Found {null_dates_after_merge} details without a matching valid invoice date. Dropping these rows.")
                 merged.dropna(subset=['processed_date'], inplace=True)
                 self.logger.info(f"Rows after dropping unmatched dates: {len(merged)}")

            if merged.empty:
                 self.logger.error("No valid data after merging invoices and details.")
                 return pd.DataFrame(columns=['sale_date', 'daily_sales', 'transaction_count', 'total_items'])

            # --- التجميع اليومي ---
            daily_sales = merged.groupby('processed_date').agg(
                daily_sales=(amount_col, 'sum'),
                transaction_count=(invoices_id_col, 'nunique'), # عدد الفواتير الفريدة
                total_items=(qty_col, 'sum') # مجموع الكميات
            ).reset_index()

            daily_sales = daily_sales.rename(columns={'processed_date': 'sale_date'})
            # التأكد من أن sale_date هو datetime object قبل الإرجاع
            daily_sales['sale_date'] = pd.to_datetime(daily_sales['sale_date'])

            num_days_final = len(daily_sales)
            self.logger.info(f"*** Aggregated daily sales for {num_days_final} days. ***")
            if num_days_final < 7: # تحذير إذا كانت البيانات قليلة جدًا
                 self.logger.warning(f"Resulting daily sales data has only {num_days_final} days, which might be too short for reliable forecasting.")

            return daily_sales.sort_values('sale_date')

        except (ValueError, KeyError) as e:
            self.logger.error(f"Handled error during daily sales creation: {str(e)}", exc_info=True)
            raise # Re-raise for the calling function
        except Exception as e:
            self.logger.error(f"Unexpected error creating daily sales: {str(e)}", exc_info=True)
            raise

    def _add_time_features(self, df):
        """Adds time-based features to the DataFrame."""
        if df.empty or 'sale_date' not in df.columns:
            return df

        df_out = df.copy()
        # Ensure 'sale_date' is datetime
        df_out['date_temp'] = pd.to_datetime(df_out['sale_date'])

        try:
            df_out['day_of_week'] = df_out['date_temp'].dt.dayofweek.astype('int8')
            df_out['is_weekend'] = df_out['day_of_week'].isin([4, 5]).astype('int8') # Assuming Fri/Sat weekend
            df_out['month'] = df_out['date_temp'].dt.month.astype('int8')
            df_out['quarter'] = df_out['date_temp'].dt.quarter.astype('int8')
            df_out['day_of_month'] = df_out['date_temp'].dt.day.astype('int8')
            try:
                 df_out['week_of_year'] = df_out['date_temp'].dt.isocalendar().week.astype('uint8')
            except AttributeError:
                 df_out['week_of_year'] = df_out['date_temp'].dt.isocalendar().week.astype('uint8') # pandas >= 1.1 might need .isocalendar()
            # Use .date for holiday lookup
            df_out['is_holiday'] = df_out['date_temp'].dt.date.apply(lambda x: 1 if x in self.saudi_holidays else 0).astype('int8')
            df_out['is_month_start'] = df_out['date_temp'].dt.is_month_start.astype('int8')
            df_out['is_month_end'] = df_out['date_temp'].dt.is_month_end.astype('int8')
            df_out['day_of_year'] = df_out['date_temp'].dt.dayofyear.astype('int16')
            df_out['year'] = df_out['date_temp'].dt.year.astype('int16')

            df_out = df_out.drop(columns=['date_temp'])
            # logger.debug("Added time features successfully.")
        except Exception as e:
            self.logger.error(f"Error adding time features: {e}", exc_info=True)
            return df # Return original df on error
        return df_out

    def _fill_missing_dates(self, df):
        """Fills missing dates and re-adds time features."""
        if df.empty or 'sale_date' not in df.columns:
             self.logger.warning("DataFrame empty or missing 'sale_date', cannot fill missing dates.")
             return df

        try:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
        except Exception as e_date_conv:
            self.logger.error(f"Error converting sale_date to datetime in _fill_missing_dates: {e_date_conv}")
            return df

        unique_dates_count = df['sale_date'].nunique()
        self.logger.info(f"Unique dates before filling: {unique_dates_count}")

        if unique_dates_count < 2:
            self.logger.warning(f"Only {unique_dates_count} unique dates found. Cannot reliably fill gaps. Adding time features only.")
            df_out = df.copy()
            try: df_out = self._add_time_features(df_out)
            except Exception as e_time: self.logger.error(f"Error adding time features to sparse data: {e_time}")
            return df_out.sort_values('sale_date') # No reset_index needed if index not changed

        # Set index and attempt to infer/set frequency to daily
        df_out = df.copy().set_index('sale_date').sort_index()
        try:
            inferred_freq = pd.infer_freq(df_out.index)
            if inferred_freq != 'D':
                if inferred_freq: self.logger.warning(f"Inferred frequency is '{inferred_freq}', forcing 'D'.")
                else: self.logger.warning("Could not infer frequency, forcing 'D'.")
                df_out = df_out.asfreq('D')
            else:
                self.logger.info("Daily frequency 'D' confirmed/inferred.")
                # Still use asfreq to ensure completeness even if inferred correctly
                df_out = df_out.asfreq('D')
        except Exception as e_freq:
            self.logger.error(f"Error setting daily frequency: {e_freq}. Proceeding without filling gaps.")
            df_out = df.copy() # Revert to original df if asfreq fails
            try: df_out = self._add_time_features(df_out)
            except Exception as e_time_nofill: self.logger.error(f"Error adding time features after freq failure: {e_time_nofill}")
            return df_out.sort_values('sale_date') # Return original with time features


        # Fill NaNs introduced by asfreq
        sales_cols = ['daily_sales', 'transaction_count', 'total_items']
        nan_filled_count = 0
        for col in sales_cols:
            if col in df_out.columns:
                original_nan_count = df_out[col].isnull().sum()
                if original_nan_count > 0:
                    df_out[col] = df_out[col].fillna(0) # Fill with 0 for sales/counts on missing days
                    nan_filled_count += original_nan_count
        if nan_filled_count > 0:
             self.logger.info(f"Filled {nan_filled_count} NaNs in sales/counts columns with 0 for new dates.")

        df_out = df_out.reset_index().rename(columns={'index': 'sale_date'})

        # Re-calculate time features for the complete date range
        try:
             df_out['sale_date'] = pd.to_datetime(df_out['sale_date']) # Ensure datetime again
             df_out = self._add_time_features(df_out)
             self.logger.info("Filled missing dates and recalculated time features.")
        except Exception as e_time_final:
            self.logger.error(f"Error adding final time features after filling: {e_time_final}")

        return df_out.sort_values('sale_date').reset_index(drop=True)


    def _add_lag_features(self, df):
        """Adds lag features for the target variable."""
        if df.empty or 'daily_sales' not in df.columns:
             return df
        if len(df) < 2 :
             self.logger.warning(f"DataFrame too short ({len(df)} rows) to calculate lag features.")
             return df

        target = 'daily_sales'
        # Define lags - consider adding more or fewer based on analysis needs
        lags = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90]
        df_out = df.copy()
        df_out = df_out.sort_values('sale_date') # Ensure sorted by date

        self.logger.info(f"Adding lag features for '{target}' for periods: {lags}")
        lags_added = []
        for lag in lags:
            col_name = f'sales_lag_{lag}'
            if lag < len(df_out):
                 df_out[col_name] = df_out[target].shift(lag)
                 lags_added.append(col_name)
            else:
                 self.logger.warning(f"Lag {lag} skipped as it's >= DataFrame length ({len(df_out)}).")
        self.logger.info(f"Added {len(lags_added)} lag features: {lags_added}")
        return df_out


    def _clean_data(self, df):
        """Final cleaning, type conversion, and column selection."""
        if df.empty:
             self.logger.warning("DataFrame is empty, skipping cleaning.")
             return df

        self.logger.info(f"Starting final data cleaning... Shape before: {df.shape}")
        df_out = df.copy()

        # Define required columns (base + time + lags)
        base_cols = ['sale_date', 'daily_sales', 'transaction_count', 'total_items']
        time_cols = ['day_of_week', 'is_weekend', 'month', 'quarter', 'day_of_month',
                     'week_of_year', 'is_holiday', 'is_month_start', 'is_month_end',
                     'day_of_year', 'year']
        lag_cols = [col for col in df_out.columns if col.startswith('sales_lag_')]
        required_columns = base_cols + time_cols + lag_cols

        # Keep only existing required columns
        existing_required_columns = [col for col in required_columns if col in df_out.columns]
        original_cols = df_out.columns.tolist()
        df_out = df_out[existing_required_columns]
        removed_cols = set(original_cols) - set(existing_required_columns)
        if removed_cols:
            self.logger.info(f"Removed non-required/missing columns: {list(removed_cols)}")

        # Handle potential NaNs (target should have been filled, lags are ok initially)
        rows_before_drop = len(df_out)
        if 'daily_sales' in df_out.columns and df_out['daily_sales'].isnull().any():
             self.logger.warning("Unexpected NaNs found in 'daily_sales'. Dropping rows.")
             df_out = df_out.dropna(subset=['daily_sales'])

        # Check for NaNs in time features (shouldn't happen if added correctly)
        time_cols_check = [col for col in time_cols if col in df_out.columns]
        nan_in_time_features = df_out[time_cols_check].isnull().sum().sum()
        if nan_in_time_features > 0:
             nan_time_cols_list = df_out[time_cols_check].columns[df_out[time_cols_check].isnull().any()].tolist()
             self.logger.error(f"*** CRITICAL ERROR: NaNs found in time features after creation! Columns: {nan_time_cols_list}. Attempting fill with 0.")
             df_out[nan_time_cols_list] = df_out[nan_time_cols_list].fillna(0)

        if len(df_out) < rows_before_drop:
            self.logger.warning(f"Dropped {rows_before_drop - len(df_out)} rows during cleaning (likely due to NaN in target).")

        if df_out.empty:
             self.logger.error("DataFrame became empty after cleaning.")
             return df_out

        # Convert data types for memory efficiency
        for col in df_out.select_dtypes(include=['float64']).columns:
            df_out[col] = df_out[col].astype('float32')
        for col in df_out.select_dtypes(include=['int64', 'Int64']).columns:
            if col not in ['id', 'invoice_id', 'product_id', 'user_id']: # Exclude potential large IDs
                 try: df_out[col] = pd.to_numeric(df_out[col], downcast='integer')
                 except Exception: pass
        bool_like_cols = ['is_weekend', 'is_holiday', 'is_month_start', 'is_month_end']
        for col in bool_like_cols:
            if col in df_out.columns:
                 df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0).astype('int8')

        # Ensure final sort and reset index
        df_out = df_out.sort_values('sale_date').reset_index(drop=True)

        self.logger.info(f"Data cleaning finished. Final shape: {df_out.shape}")
        # self.logger.debug(f"Final dtypes:\n{df_out.dtypes}")
        return df_out

    def generate_features(self):
        """Main function to run feature engineering steps."""
        try:
            self.logger.info("--- Starting Feature Generation Process ---")
            self._preprocess_dates()
            daily_sales = self._create_daily_sales()

            if daily_sales.empty:
                 self.logger.error("Initial daily sales creation failed or yielded no data.")
                 raise ValueError("Failed to create initial daily sales data.")

            daily_sales_filled = self._fill_missing_dates(daily_sales)
            daily_sales_lagged = self._add_lag_features(daily_sales_filled)
            self.features = self._clean_data(daily_sales_lagged)

            if self.features.empty:
                 self.logger.error("Feature generation resulted in an empty DataFrame after cleaning.")
                 # Depending on requirements, either return empty or raise error
                 # raise ValueError("Feature generation resulted in empty DataFrame.")

            self.logger.info("--- Feature Generation Process Completed Successfully ---")
            return self.features

        except (FileNotFoundError, ValueError, KeyError) as data_err:
             self.logger.error(f"Feature generation failed due to data/config error: {str(data_err)}", exc_info=True)
             raise # Re-raise specific error
        except Exception as e:
            self.logger.error(f"Unexpected general failure in feature generation: {str(e)}", exc_info=True)
            raise RuntimeError(f"General failure in generating features: {e}")


# --- **** الدالة العامة المعدلة القابلة للاستدعاء **** ---
def generate_features_df(data_dict: Dict[str, pd.DataFrame]):
    """
    Performs feature engineering on provided DataFrames and returns a DataFrame ready for forecasting.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary containing the required DataFrames.
                           Expected keys: 'sale_invoices', 'sale_invoices_details'.

    Returns:
        pandas.DataFrame: DataFrame with daily sales and derived features,
                          or raises an Exception on failure.
    """
    logger.info("--- Starting generate_features_df function (with DataFrames) ---")
    try:
        # --- استدعاء الكلاس المعدل بالـ DataFrames ---
        engineer = _SalesFeatureEngineerInternal(data_dict)
        features_df = engineer.generate_features()

        if features_df is None or features_df.empty:
             logger.error("Feature engineering process failed or returned empty/None DataFrame.")
             raise ValueError("Feature generation failed (result is empty or None).")

        # --- التحقق من الأعمدة الناتجة الأساسية ---
        required_output_cols = ['sale_date', 'daily_sales']
        if not all(col in features_df.columns for col in required_output_cols):
            missing_cols = list(set(required_output_cols) - set(features_df.columns))
            logger.error(f"Output DataFrame is missing essential columns: {missing_cols}")
            raise ValueError(f"Output DataFrame is incomplete (missing: {missing_cols}).")

        # --- التحقق من عدم وجود NaN في التاريخ ---
        if 'sale_date' in features_df.columns and features_df['sale_date'].isnull().any():
            logger.error("Output DataFrame contains NaN values in 'sale_date' column.")
            raise ValueError("Missing date values in final feature result.")

        logger.info(f"--- generate_features_df completed successfully. Shape: {features_df.shape} ---")
        return features_df

    except (FileNotFoundError, ValueError, KeyError, RuntimeError, TypeError) as data_err:
        logger.error(f"Handled error in generate_features_df: {data_err}", exc_info=True)
        raise # Re-raise the specific error
    except Exception as e:
        logger.error(f"Unexpected general error in generate_features_df: {e}", exc_info=True)
        raise RuntimeError(f"General failure in feature generation function: {e}")


# --- مثال للاختبار المستقل (معدّل لاستخدام DataFrames) ---
if __name__ == "__main__":
    print("--- Standalone test for feature_engineering.py (using DataFrames) ---")
    # !!! تحديث المسارات لاختبار التحميل المحلي فقط !!!
    test_file_paths = {
        'sale_invoices': r'C:\Users\sheee\Downloads\ZodData\sale_invoices.xlsx', # مسار ملف الفواتير
        'sale_invoices_details': r'C:\Users\sheee\Downloads\ZodData\sale_invoices_details.xlsx' # مسار ملف التفاصيل
    }
    test_data_input_dict = {}
    try:
        print("Loading test data from files into DataFrames...")
        for name, path in test_file_paths.items():
            if os.path.exists(path):
                test_data_input_dict[name] = pd.read_excel(path)
                print(f"  - Loaded '{name}' ({len(test_data_input_dict[name])} rows)")
            else:
                 print(f"  - ERROR: Test file not found: {path}")
                 raise FileNotFoundError(f"Test file missing: {path}")

        print(f"\nCalling generate_features_df with loaded DataFrames...")
        # --- استدعاء الدالة المعدلة ---
        features_result_df = generate_features_df(test_data_input_dict)

        print("\n--- Feature generation successful (standalone test) ---")
        print(f"Output DataFrame shape: {features_result_df.shape}")
        if not features_result_df.empty:
            print(f"Time range: {features_result_df['sale_date'].min().date()} to {features_result_df['sale_date'].max().date()}")
            print("\nFirst 5 rows:")
            print(features_result_df.head())
            print("\nLast 5 rows:")
            print(features_result_df.tail())
            print("\nDataFrame Info:")
            features_result_df.info()

            # --- التحقق من الفجوات الزمنية ---
            date_range_check = pd.date_range(start=features_result_df['sale_date'].min(), end=features_result_df['sale_date'].max(), freq='D')
            missing_dates_check = date_range_check[~date_range_check.isin(features_result_df['sale_date'])]
            if not missing_dates_check.empty:
                 print(f"\n*** WARNING: Found {len(missing_dates_check)} missing days within the processed date range!")
                 # print(missing_dates_check)
            else:
                 print("\nDate Check: No missing days found within the processed date range.")
        else:
             print("Output DataFrame is empty.")

    except (FileNotFoundError, ValueError, KeyError, RuntimeError, TypeError) as e:
        print(f"\n--- Standalone test FAILED with handled error: {str(e)} ---")
        traceback.print_exc()
    except Exception as e_main:
         print(f"\n--- Standalone test FAILED with unexpected error: {str(e_main)} ---")
         traceback.print_exc()
