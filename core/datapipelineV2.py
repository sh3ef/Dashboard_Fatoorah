# core/datapipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.config import config # Import updated settings
import logging
import os
import traceback
import time
from sqlalchemy import create_engine, exc as sqlalchemy_exc
import json
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
from enum import Enum

# --- Forecasting Imports ---
try:
    from core.feature_engineering import generate_features_df
    from core.forecasting import train_and_forecast, DATE_COLUMN_OUTPUT as FORECAST_DATE_COL_DAILY
    FORECASTING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"Imported Daily FORECAST_DATE_COL as '{FORECAST_DATE_COL_DAILY}'")
except ImportError as import_err:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import daily feature_engineering or forecasting: {import_err}. Daily forecasting disabled.")
    def generate_features_df(data_dict): logging.warning("generate_features_df (dummy): Module not found."); return pd.DataFrame()
    def train_and_forecast(df, forecast_horizon=14, seasonal_period=7): logging.warning("train_and_forecast (dummy): Module not found."); return pd.DataFrame()
    FORECASTING_AVAILABLE = False
    FORECAST_DATE_COL_DAILY = 'date' # Default date column name

try:
    from core.monthly_feature_engineering import generate_monthly_features
    from core.monthly_forecasting import train_and_forecast_monthly, DATE_COLUMN_OUTPUT as FORECAST_DATE_COL_MONTHLY
    MONTHLY_FORECASTING_AVAILABLE = True
    logger.info(f"Imported Monthly FORECAST_DATE_COL as '{FORECAST_DATE_COL_MONTHLY}'")
except ImportError as import_err_monthly:
    logger = logging.getLogger(__name__) # Ensure logger is defined
    logger.error(f"Failed to import monthly feature_engineering or forecasting: {import_err_monthly}. Monthly forecasting disabled.")
    def generate_monthly_features(ts, max_lag=12): logging.warning("generate_monthly_features (dummy): Module not found."); return None
    def train_and_forecast_monthly(df, horizon=12, seasonal=12): logging.warning("train_and_forecast_monthly (dummy): Module not found."); return pd.DataFrame()
    MONTHLY_FORECASTING_AVAILABLE = False
    FORECAST_DATE_COL_MONTHLY = 'date' # Default date column name

# --- تحديد اسم عمود التاريخ الموحد للاستخدام في المخرجات النهائية ---
try:
    # Prefer monthly if available, else daily
    if MONTHLY_FORECASTING_AVAILABLE:
        from core.monthly_forecasting import DATE_COLUMN_OUTPUT as FORECAST_DATE_COL
        logger.info(f"Using FORECAST_DATE_COL '{FORECAST_DATE_COL}' from monthly_forecasting.")
    elif FORECASTING_AVAILABLE:
        from core.forecasting import DATE_COLUMN_OUTPUT as FORECAST_DATE_COL
        logger.info(f"Using FORECAST_DATE_COL '{FORECAST_DATE_COL}' from daily_forecasting.")
    else:
        FORECAST_DATE_COL = 'date' # تعيين الافتراضي
        logger.warning(f"Using default FORECAST_DATE_COL '{FORECAST_DATE_COL}' as forecasting modules failed to import it.")
except ImportError:
     FORECAST_DATE_COL = 'date' # تعيين الافتراضي في حالة فشل الاستيراد تماماً
     logger = logging.getLogger(__name__) # Ensure logger is available
     logger.error(f"Failed to import DATE_COLUMN_OUTPUT from forecasting modules. Using default '{FORECAST_DATE_COL}'.")


# --- Pydantic Model Imports ---
try:
     from api.models import ChartData, ChartMetadata, ChartType, AxisInfo, SeriesInfo
     MODELS_IMPORTED = True
except ImportError:
     logger = logging.getLogger(__name__) # Ensure logger is defined
     logger.error("Failed to import Pydantic models from api.models. Prepare functions might fail.")
     # Define dummy models to avoid runtime errors if import fails
     class BaseModel: pass
     class ChartTypeEnum(str, Enum): BAR="bar"; LINE="line"; PIE="pie"; SCATTER="scatter"; TABLE="table"; COMBO="combo"
     class ChartType(BaseModel): pass
     ChartType = ChartTypeEnum # Use the enum directly
     class AxisInfo(BaseModel): name: str = ""; type: Optional[str] = None; title: Optional[str] = None
     class SeriesInfo(BaseModel): name: str = ""; color: Optional[str] = None; type: Optional[ChartType] = None
     class ChartMetadata(BaseModel): timestamp: str = ""; description: str = ""; frequency: str = ""; title: str = ""; chart_type: ChartType = ChartType.TABLE; x_axis: AxisInfo = AxisInfo(); y_axis: AxisInfo = AxisInfo(); series: List[SeriesInfo] = []
     class ChartData(BaseModel): metadata: ChartMetadata = ChartMetadata(); data: Union[List[Dict[str, Any]], Dict] = {}
     MODELS_IMPORTED = False

# --- Localization Import ---
try:
    from core.localization import get_translation, SUPPORTED_LANGUAGES
    LOCALIZATION_AVAILABLE = True
    logger.info(f"Localization module loaded. Supported languages: {SUPPORTED_LANGUAGES}")
except ImportError as loc_err:
    logger = logging.getLogger(__name__) # Ensure logger is defined
    logger.error(f"Failed to import localization module: {loc_err}. Localization disabled.")
    LOCALIZATION_AVAILABLE = False
    SUPPORTED_LANGUAGES = ['en'] # Fallback to English only
    # Define a dummy get_translation function if localization fails
    def get_translation(lang: str, key: str, default: str = "") -> str:
        # logger.debug(f"Localization dummy: lang='{lang}', key='{key}', returning default='{default}'")
        return default

# --- Plotly Import (Optional, for colors) ---
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False
    # logger.warning("Plotly Express not available...") # Use logger if defined

# --- Logger Setup ---
# Setup is done at the top level now, but ensure logger exists
if 'logger' not in globals():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Date Parsing Utility ---
def _parse_excel_date_utility(date_val):
    """
     يحاول تحليل التاريخ من أرقام Excel التسلسلية أو النصوص أو كائنات التاريخ.
     أكثر قوة للتعامل مع تنسيقات مختلفة.
    """
    if pd.isna(date_val):
        return pd.NaT
    # 1. التعامل مع كائنات التاريخ/الوقت الموجودة بالفعل
    if isinstance(date_val, (datetime, pd.Timestamp)):
        return pd.to_datetime(date_val) # التأكد من أنه Timestamp متوافق مع Pandas

    # 2. التعامل مع النصوص
    if isinstance(date_val, str):
        date_val = date_val.strip()
        if not date_val: return pd.NaT
        try:
            # محاولة التحليل المباشر (Pandas ذكي جدًا)
            return pd.to_datetime(date_val, errors='coerce', dayfirst=None)
        except Exception:
             return pd.NaT

    # 3. التعامل مع الأرقام (Excel serial أو Unix timestamp)
    if isinstance(date_val, (int, float)):
        try:
            # محاولة التعامل معها كـ Excel serial number أولاً
            if 1 < date_val < 2958466: # نطاق Excel التقريبي (1900-9999)
                 if date_val == 60: return pd.Timestamp('1900-02-29') # معالجة خطأ Excel
                 excel_epoch = pd.Timestamp('1899-12-30')
                 return excel_epoch + pd.to_timedelta(date_val, unit='D')
            else:
                 # إذا لم تكن ضمن نطاق Excel، جرب Unix timestamp (بالثواني)
                 try:
                     if abs(date_val - time.time()) < 10**10: # فرق معقول
                          return pd.to_datetime(date_val, unit='s', origin='unix')
                     else: return pd.NaT
                 except (ValueError, pd.errors.OutOfBoundsDatetime):
                     return pd.NaT
        except Exception:
             return pd.NaT

    # 4. أنواع أخرى
    try:
        return pd.to_datetime(date_val, errors='coerce')
    except Exception:
        return pd.NaT

# --- Constants ---
RESULTS_BASE_DIR = Path("data/client_results") # مكان حفظ ملفات JSON الناتجة

class DataPipeline:
    def __init__(self):
        self.raw_data = {}          # لتخزين البيانات الخام من قاعدة البيانات
        self.processed_data = {}    # لتخزين البيانات بعد التنظيف والمعالجة
        self.analytics = {}         # لتخزين نتائج التحليل (مثل product_flow, pareto_data)
        self.engine = None          # محرك SQLAlchemy للاتصال بقاعدة البيانات
        # استرجاع إعدادات التنبؤ من config
        self.forecast_horizon_daily = config.analysis_params.get('forecast_horizon_daily', 14)
        self.forecast_horizon_monthly = config.analysis_params.get('forecast_horizon_monthly', 6)
        self.seasonal_period_daily = config.analysis_params.get('seasonal_period_daily', 7)
        self.seasonal_period_monthly = config.analysis_params.get('seasonal_period_monthly', 12)
        # تخزين نتائج التنبؤ
        self.forecast_data = pd.DataFrame() # للتنبؤ اليومي
        self.monthly_avg_invoice_forecast = pd.DataFrame() # للتنبؤ الشهري

        # --- إنشاء اتصال قاعدة البيانات عند التهيئة ---
        try:
            db_url = config.db.get_db_url()
            logger.info(f"Attempting to create SQLAlchemy engine for: {config.db.DB_NAME} on {config.db.DB_HOST}:{config.db.DB_PORT}")
            temp_engine = create_engine(db_url, pool_pre_ping=True, connect_args={'connect_timeout': 15})
            # Test connection
            with temp_engine.connect() as connection:
                logger.info(f"SQLAlchemy engine connection test successful.")
            self.engine = temp_engine
            logger.info(f"SQLAlchemy engine initialized successfully.")
        except sqlalchemy_exc.OperationalError as db_conn_err:
             logger.critical(f"!!! Database Operational Error during engine creation: {db_conn_err}", exc_info=False)
             self.engine = None
        except Exception as e:
            logger.critical(f"!!! Failed to create SQLAlchemy engine during __init__: {e}", exc_info=True)
            self.engine = None

    def _load_data_from_db(self, client_id: int):
        """تحميل البيانات الخام من قاعدة البيانات لعميل محدد."""
        if self.engine is None:
            logger.error("Database engine is not initialized. Cannot load data.")
            raise ConnectionError("DB connection failed during pipeline initialization.")

        logger.info(f"Loading data from DB for Client ID: {client_id}...")
        self.raw_data = {}
        queries = {
            'products': f"SELECT * FROM products WHERE client_id = {client_id};",
            'sale_invoices': f"SELECT * FROM sale_invoices WHERE client_id = {client_id};",
            'sale_invoices_details': f"SELECT sid.* FROM sale_invoices_details sid JOIN sale_invoices si ON sid.invoice_id = si.id WHERE si.client_id = {client_id};",
            'invoice_deferreds': f"SELECT * FROM invoice_deferreds WHERE client_id = {client_id};"
        }
        success = True
        try:
            with self.engine.connect() as connection:
                for table_name, query in queries.items():
                    start_time = time.time()
                    df = pd.DataFrame() # Initialize df
                    try:
                        logger.debug(f"Executing query for '{table_name}'...")
                        df = pd.read_sql(query, connection)
                        load_time = time.time() - start_time
                        logger.info(f"Loaded '{table_name}' ({len(df)} rows) in {load_time:.2f}s.")
                        self.raw_data[table_name] = df
                    except Exception as e_query:
                        logger.error(f"Query failed for table '{table_name}': {e_query}", exc_info=False)
                        self.raw_data[table_name] = pd.DataFrame() # Ensure empty DF on error
                        if table_name not in ['invoice_deferreds']: # Allow deferreds to fail optionally
                            success = False
        except sqlalchemy_exc.OperationalError as db_err:
            logger.error(f"Database connection error during data loading: {db_err}", exc_info=True)
            raise ConnectionError(f"DB connection failed during data load: {db_err}")
        except Exception as e_conn:
            logger.error(f"Unexpected error during DB data loading: {e_conn}", exc_info=True)
            raise RuntimeError(f"Failed to load data from database: {e_conn}")

        if not success:
            logger.error("Failed to load one or more essential data tables from the database.")
            raise ValueError("Essential data table load failed from database.")

        logger.info("Finished loading data from DB.")

    def _validate_input_data(self):
        """التحقق من وجود البيانات المطلوبة والأعمدة بعد تحميلها من قاعدة البيانات."""
        logger.info("Validating input data loaded from DB...")
        required_tables = ['products', 'sale_invoices', 'sale_invoices_details', 'invoice_deferreds']
        all_data_valid = True

        for table in required_tables:
            df = self.raw_data.get(table)

            if df is None:
                if table != 'invoice_deferreds': # Deferreds is optional for some charts
                    logger.error(f"Essential table '{table}' DataFrame is missing from loaded data.")
                    all_data_valid = False
                else:
                    logger.warning(f"Optional table '{table}' DataFrame is missing. Proceeding without it.")
                    self.raw_data[table] = pd.DataFrame() # Ensure empty DF exists
                continue

            if df.empty:
                # Empty essential tables might cause issues later
                if table != 'invoice_deferreds':
                    logger.warning(f"Essential table '{table}' is empty.")
                else:
                    logger.info(f"Optional table '{table}' is empty.") # Less critical
                # Skip column check if empty
                continue

            # DataFrame is not empty, check columns
            req_cols = config.required_columns.get(table)
            if not req_cols:
                logger.warning(f"No required columns defined for table '{table}' in config. Skipping column check for this table.")
                continue

            missing = [col for col in req_cols if col not in df.columns]
            if missing:
                logger.error(f"Table '{table}' is missing required columns defined in config: {missing}")
                all_data_valid = False


        if not all_data_valid:
            raise ValueError("Input data validation failed after loading from DB. Check logs for missing tables/columns.")
        logger.info("Input data validation completed successfully.")

    def _preprocess_data(self):
        """تنظيف وتحضير البيانات المحملة من قاعدة البيانات."""
        logger.info("Starting data preprocessing...")
        self.processed_data = {}
        sid_raw = self.raw_data.get('sale_invoices_details', pd.DataFrame()).copy()
        si_raw = self.raw_data.get('sale_invoices', pd.DataFrame()).copy()
        products_raw = self.raw_data.get('products', pd.DataFrame()).copy()
        deferred_raw = self.raw_data.get('invoice_deferreds', pd.DataFrame()).copy()

        # --- معالجة المنتجات (products) ---
        if not products_raw.empty:
            try:
                prod_id_col, prod_name_col, prod_buy_col, prod_sale_col, prod_qty_col, _ = config.required_columns['products']
                required_prod_cols = [prod_id_col, prod_name_col, prod_buy_col, prod_sale_col, prod_qty_col]
                if not all(c in products_raw.columns for c in required_prod_cols):
                    raise ValueError(f"Missing one or more required columns in raw products data: {required_prod_cols}")
                products = products_raw[required_prod_cols].copy()
                products[prod_id_col] = products[prod_id_col].astype(str)
                for col in [prod_qty_col, prod_buy_col, prod_sale_col]:
                    products[col] = pd.to_numeric(products[col], errors='coerce').fillna(0)
                self.processed_data['products'] = products
            except KeyError as e: raise ValueError(f"Missing required product column definition in config: {e}")
            except ValueError as ve: raise ve
            except Exception as e: raise RuntimeError(f"Error preprocessing products: {e}")
        else:
            logger.warning("Product data is empty. Analysis results might be affected.")
            self.processed_data['products'] = pd.DataFrame()

        # --- معالجة فواتير البيع (sale_invoices) ---
        if not si_raw.empty:
            try:
                si_id_col, si_date_col_config, si_total_col, _ = config.required_columns['sale_invoices']
                required_si_cols = [si_id_col, si_date_col_config, si_total_col]
                if not all(c in si_raw.columns for c in required_si_cols):
                     raise ValueError(f"Missing one or more required columns in raw sale_invoices data: {required_si_cols}")
                si = si_raw[required_si_cols].copy()
                si[si_id_col] = pd.to_numeric(si[si_id_col], errors='coerce')
                si.dropna(subset=[si_id_col], inplace=True)
                si[si_id_col] = si[si_id_col].astype('Int64') # Use Pandas nullable integer type
                si['created_at_dt'] = si[si_date_col_config].apply(_parse_excel_date_utility)
                original_rows = len(si)
                si.dropna(subset=['created_at_dt'], inplace=True)
                dropped_rows = original_rows - len(si)
                if dropped_rows > 0: logger.warning(f"Sale Invoices: Dropped {dropped_rows} rows due to invalid dates.")
                if si.empty:
                    logger.warning("No valid sale invoices remaining after date cleaning.")
                    self.processed_data['sale_invoices'] = pd.DataFrame()
                else:
                    # --- **KEEP DATETIME OBJECT FOR ANALYSIS** ---
                    # si['created_at_per_day'] = si['created_at_dt'].dt.date # Kept for potential use elsewhere
                    si['year_month'] = si['created_at_dt'].dt.strftime("%Y-%m")
                    si[si_total_col] = pd.to_numeric(si[si_total_col], errors='coerce').fillna(0)
                    # --- ADD DAILY DATE OBJECT EXPLICITLY ---
                    si['created_at_per_day_obj'] = si['created_at_dt'].dt.date
                    # ----------------------------------------
                    self.processed_data['sale_invoices'] = si
            except KeyError as e: raise ValueError(f"Missing required sale_invoice column definition in config: {e}")
            except ValueError as ve: raise ve
            except Exception as e: raise RuntimeError(f"Error preprocessing sale invoices: {e}")
        else:
            logger.warning("Sale invoices data is empty.")
            self.processed_data['sale_invoices'] = pd.DataFrame()

        # --- معالجة تفاصيل فواتير البيع (sale_invoices_details) ---
        si_proc = self.processed_data.get('sale_invoices', pd.DataFrame())
        if not sid_raw.empty and not si_proc.empty:
            try:
                sid_id_col, sid_prod_id_col, sid_inv_id_col, sid_qty_col, sid_total_col, sid_buy_col, sid_date_col_config = config.required_columns['sale_invoices_details']
                required_sid_cols = [sid_id_col, sid_prod_id_col, sid_inv_id_col, sid_qty_col, sid_total_col, sid_buy_col, sid_date_col_config]
                if not all(c in sid_raw.columns for c in required_sid_cols):
                    raise ValueError(f"Missing one or more required columns in raw sale_invoices_details data: {required_sid_cols}")
                sid = sid_raw[required_sid_cols].copy()
                sid[sid_inv_id_col] = pd.to_numeric(sid[sid_inv_id_col], errors='coerce').astype('Int64')
                sid[sid_prod_id_col] = sid[sid_prod_id_col].astype(str)
                sid['created_at_dt'] = sid[sid_date_col_config].apply(_parse_excel_date_utility)
                null_dates_initial = sid['created_at_dt'].isnull().sum()
                si_id_col_from_si = config.required_columns['sale_invoices'][0]
                # --- **USE DATETIME OBJECT FOR MERGE** ---
                if 'created_at_dt' in si_proc.columns and si_id_col_from_si in si_proc.columns:
                     # si_dates = si_proc[[si_id_col_from_si, 'created_at_dt']].rename(columns={'created_at_dt': 'invoice_date_dt', si_id_col_from_si: sid_inv_id_col})
                     si_dates_for_merge = si_proc[[si_id_col_from_si, 'created_at_dt']].copy() # Use datetime column directly
                     si_dates_for_merge.rename(columns={'created_at_dt': 'invoice_date_dt', si_id_col_from_si: sid_inv_id_col}, inplace=True)
                     try:
                          # Ensure merge keys are compatible
                          if sid[sid_inv_id_col].dtype != si_dates_for_merge[sid_inv_id_col].dtype:
                               # Try converting both to a common type, e.g., Int64
                               sid[sid_inv_id_col] = pd.to_numeric(sid[sid_inv_id_col], errors='coerce').astype('Int64')
                               si_dates_for_merge[sid_inv_id_col] = pd.to_numeric(si_dates_for_merge[sid_inv_id_col], errors='coerce').astype('Int64')
                          sid = pd.merge(sid, si_dates_for_merge, on=sid_inv_id_col, how='left')
                          # Prioritize detail date, fill with invoice date if missing
                          sid['created_at_dt'] = sid['created_at_dt'].fillna(sid['invoice_date_dt'])
                          sid.drop(columns=['invoice_date_dt'], errors='ignore', inplace=True)
                          if null_dates_initial > 0:
                               filled_count = null_dates_initial - sid['created_at_dt'].isnull().sum()
                               logger.info(f"SID: Filled {filled_count} missing dates by merging with invoices.")
                     except Exception as merge_err: logger.error(f"Failed to merge SID with SI dates: {merge_err}", exc_info=True)
                else: logger.warning("SID: Cannot merge date from processed sale_invoices (missing columns or data).")
                original_rows_sid = len(sid)
                sid.dropna(subset=['created_at_dt', sid_inv_id_col, sid_prod_id_col], inplace=True)
                dropped_rows_sid = original_rows_sid - len(sid)
                if dropped_rows_sid > 0: logger.warning(f"SID: Dropped {dropped_rows_sid} rows due to missing final date or required IDs.")
                if sid.empty:
                    logger.warning("No valid sale invoice details remaining after cleaning.")
                    self.processed_data['sale_invoices_details'] = pd.DataFrame()
                else:
                    # --- ADD DAILY DATE OBJECT EXPLICITLY ---
                    sid['created_at_per_day_obj'] = sid['created_at_dt'].dt.date
                    # ----------------------------------------
                    # si['created_at_per_day'] = si['created_at_dt'].dt.date # Kept for potential use elsewhere
                    for col in [sid_qty_col, sid_total_col, sid_buy_col]:
                        sid[col] = pd.to_numeric(sid[col], errors='coerce').fillna(0)
                    try: sid['netProfit'] = sid[sid_total_col] - (sid[sid_buy_col] * sid[sid_qty_col])
                    except Exception: logger.warning(f"Could not calculate 'netProfit' for SID. Setting to 0."); sid['netProfit'] = 0.0
                    self.processed_data['sale_invoices_details'] = sid
            except KeyError as e: raise ValueError(f"Missing required sale_invoices_detail column definition in config: {e}")
            except ValueError as ve: raise ve
            except Exception as e: raise RuntimeError(f"Error preprocessing sale invoice details: {e}")
        elif sid_raw.empty:
            logger.warning("Sale invoice details data is empty.")
            self.processed_data['sale_invoices_details'] = pd.DataFrame()
        else: # si_proc is empty
             logger.warning("Processed sale invoices data is empty, cannot effectively process details.")
             self.processed_data['sale_invoices_details'] = pd.DataFrame()

        # --- معالجة المبالغ الآجلة (invoice_deferreds) ---
        if not deferred_raw.empty:
            try:
                def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col, _ = config.required_columns['invoice_deferreds']
                required_def_cols = [def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col]
                if all(c in deferred_raw.columns for c in required_def_cols):
                    deferred = deferred_raw[required_def_cols].copy()
                    for col in [def_amount_col, def_paid_col]:
                        deferred[col] = pd.to_numeric(deferred[col], errors='coerce').fillna(0)
                    deferred[def_user_id_col] = deferred[def_user_id_col].astype(str)
                    deferred[def_status_col] = pd.to_numeric(deferred[def_status_col], errors='coerce')
                    self.processed_data['invoice_deferreds'] = deferred
                else:
                    missing_cols = [c for c in required_def_cols if c not in deferred_raw.columns]
                    logger.warning(f"Missing required columns in raw deferred data: {missing_cols}. Skipping deferred processing.")
                    self.processed_data['invoice_deferreds'] = pd.DataFrame()
            except KeyError as e: raise ValueError(f"Missing required invoice_deferreds column definition in config: {e}")
            except Exception as e: raise RuntimeError(f"Error preprocessing invoice deferreds: {e}")
        else:
            logger.warning("Invoice deferreds data is empty.")
            self.processed_data['invoice_deferreds'] = pd.DataFrame()

        logger.info("Data preprocessing completed.")

    def _analyze_data(self):
        """تحليل البيانات المعالجة وإنشاء المقاييس التحليلية."""
        logger.info("Starting data analysis...")
        sid = self.processed_data.get('sale_invoices_details', pd.DataFrame())
        products = self.processed_data.get('products', pd.DataFrame())
        si = self.processed_data.get('sale_invoices', pd.DataFrame())
        deferred = self.processed_data.get('invoice_deferreds', pd.DataFrame())

        analysis_results = {
            'product_flow': pd.DataFrame(), 'pareto_data': pd.DataFrame(),
            'pie_data': {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}},
            'stagnant_products': pd.DataFrame(), 'outstanding_amounts': pd.DataFrame(),
            'monthly_avg_invoice_ts': pd.Series(dtype='float64'),
            'daily_profit_ts': pd.Series(dtype='float64') # Placeholder
        }

        # --- Product Flow Calculation ---
        if not sid.empty and not products.empty:
            try:
                logger.info("Analyzing product flow...")
                # Get column names from config
                prod_id_col = config.required_columns['products'][0]
                prod_name_col = config.required_columns['products'][1]
                prod_buy_col = config.required_columns['products'][2]
                prod_sale_col = config.required_columns['products'][3]
                prod_qty_col = config.required_columns['products'][4]
                sid_prod_id_col = config.required_columns['sale_invoices_details'][1]
                sid_qty_col = config.required_columns['sale_invoices_details'][3]
                sid_total_col = config.required_columns['sale_invoices_details'][4]
                sid_created_at_dt_col = 'created_at_dt' # From preprocessing
                sid_net_profit_col = 'netProfit' # From preprocessing

                # Aggregate sales details
                agg_dict = {'sales_quantity': (sid_qty_col, 'sum'), 'sales_amount': (sid_total_col, 'sum')}
                if sid_net_profit_col in sid.columns:
                    agg_dict['net_profit'] = (sid_net_profit_col, 'sum')
                product_flow = sid.groupby(sid_prod_id_col).agg(**agg_dict).reset_index()

                # Merge with product info
                products_to_merge = products[[prod_id_col, prod_name_col, prod_buy_col, prod_sale_col, prod_qty_col]].copy()
                products_to_merge.rename(columns={prod_id_col: sid_prod_id_col, prod_qty_col: 'current_stock', prod_name_col: 'name', prod_buy_col: 'buyPrice', prod_sale_col: 'salePrice'}, inplace=True)
                product_flow[sid_prod_id_col] = product_flow[sid_prod_id_col].astype(str) # Ensure consistent type for merge
                products_to_merge[sid_prod_id_col] = products_to_merge[sid_prod_id_col].astype(str)
                product_flow = pd.merge(product_flow, products_to_merge, on=sid_prod_id_col, how='left')

                # Fill missing product info
                fill_values = {'name': 'Unknown Product', 'buyPrice': 0, 'salePrice': 0, 'current_stock': 0, 'sales_quantity': 0, 'sales_amount': 0}
                if 'net_profit' in product_flow.columns: fill_values['net_profit'] = 0
                product_flow.fillna(fill_values, inplace=True)

                # Calculate efficiency
                if 'current_stock' in product_flow.columns and 'sales_quantity' in product_flow.columns:
                    s_q = product_flow['sales_quantity'].astype(float)
                    c_s = product_flow['current_stock'].astype(float)
                    product_flow['efficiency_ratio'] = np.where(s_q > 1e-6, c_s / s_q, np.nan) # Avoid division by zero
                    eff_bins = config.analysis_params['efficiency_bins']
                    eff_labels = config.analysis_params['efficiency_labels']
                    valid_ratios = product_flow['efficiency_ratio'].notna()
                    product_flow['efficiency'] = 'N/A' # Default
                    if valid_ratios.any():
                         product_flow.loc[valid_ratios, 'efficiency'] = pd.cut(product_flow.loc[valid_ratios, 'efficiency_ratio'], bins=eff_bins, labels=eff_labels, right=False)
                         product_flow['efficiency'] = pd.Categorical(product_flow['efficiency'], categories=eff_labels + ['N/A'], ordered=False)
                         product_flow['efficiency'] = product_flow['efficiency'].fillna('N/A') # Fill NaNs resulting from cut
                else:
                    product_flow['efficiency'] = 'N/A'

                # Calculate days since last sale
                if sid_created_at_dt_col in sid.columns:
                    last_sale = sid.groupby(sid_prod_id_col)[sid_created_at_dt_col].max().reset_index()
                    last_sale[sid_prod_id_col] = last_sale[sid_prod_id_col].astype(str) # Ensure type for merge
                    product_flow = pd.merge(product_flow, last_sale, on=sid_prod_id_col, how='left')
                    product_flow.rename(columns={sid_created_at_dt_col: 'last_sale_date'}, inplace=True)
                    product_flow['last_sale_date'] = pd.to_datetime(product_flow['last_sale_date'], errors='coerce')
                    # Timezone-aware calculation if necessary, otherwise normalize
                    valid_dates = product_flow['last_sale_date'].notna()
                    if valid_dates.any():
                         # Use current time's timezone if available, otherwise assume naive or UTC
                         now_ts = pd.Timestamp.now(tz=product_flow['last_sale_date'].dt.tz).normalize()
                         product_flow.loc[valid_dates, 'days_since_last_sale'] = (now_ts - product_flow.loc[valid_dates, 'last_sale_date'].dt.normalize()).dt.days
                    else:
                         product_flow['days_since_last_sale'] = None
                    product_flow['days_since_last_sale'] = pd.to_numeric(product_flow['days_since_last_sale'], errors='coerce').fillna(99999).astype(int) # Fill NaN after calculation
                else:
                    product_flow['last_sale_date'] = pd.NaT
                    product_flow['days_since_last_sale'] = 99999

                analysis_results['product_flow'] = product_flow
                logger.info("Product flow analysis completed.")
            except Exception as e_flow: logger.error(f"Error during Product Flow analysis: {e_flow}", exc_info=True); analysis_results['product_flow'] = pd.DataFrame()
        else: logger.warning("Skipping Product Flow analysis due to missing processed input data.")

        pf = analysis_results.get('product_flow', pd.DataFrame())

        # --- Pareto Analysis ---
        if not pf.empty and 'sales_quantity' in pf.columns:
            try:
                logger.info("Analyzing Pareto...")
                # Filter for products actually sold
                sorted_df = pf[pf['sales_quantity'] > 0].sort_values('sales_quantity', ascending=False).copy()
                total_sales_qty = sorted_df['sales_quantity'].sum()
                if total_sales_qty > 0:
                    sorted_df['cumulative_percentage'] = sorted_df['sales_quantity'].cumsum() / total_sales_qty * 100
                    pareto_threshold = config.analysis_params.get('pareto_threshold', 80)
                    analysis_results['pareto_data'] = sorted_df[sorted_df['cumulative_percentage'] <= pareto_threshold].copy()
                    logger.info("Pareto analysis completed.")
                else:
                    # Handle case where sales_quantity exists but sum is 0 or less
                    logger.warning("Pareto analysis skipped: Total sales quantity is not positive.")
                    analysis_results['pareto_data'] = pd.DataFrame()
            except Exception as e: logger.error(f"Error during Pareto analysis: {e}", exc_info=True); analysis_results['pareto_data'] = pd.DataFrame()
        else: logger.warning("Skipping Pareto analysis (Product flow empty or missing sales quantity).")

        # --- Pie Chart Data (Revenue & Profit) ---
        if not pf.empty and 'sales_amount' in pf.columns and 'name' in pf.columns:
            try:
                logger.info("Analyzing Pie Chart data (Revenue & Profit)...")
                profit_col_available = 'net_profit' in pf.columns

                # Revenue Pie Data
                top_rev = pf.nlargest(10, 'sales_amount')[['name', 'sales_amount']].rename(columns={'sales_amount': 'totalPrice'})
                other_rev = max(0, pf['sales_amount'].sum() - top_rev['totalPrice'].sum())
                if other_rev > 1e-6: # Add 'Other' category if significant
                    top_rev = pd.concat([top_rev, pd.DataFrame([{'name': 'Other', 'totalPrice': other_rev}])], ignore_index=True)

                # Profit Pie Data (if available)
                top_prof = pd.DataFrame()
                if profit_col_available:
                    top_prof = pf.nlargest(10, 'net_profit')[['name', 'net_profit']].rename(columns={'net_profit': 'netProfit'})
                    other_prof = max(0, pf['net_profit'].sum() - top_prof['netProfit'].sum())
                    if other_prof > 1e-6: # Add 'Other' category if significant
                        top_prof = pd.concat([top_prof, pd.DataFrame([{'name': 'Other', 'netProfit': other_prof}])], ignore_index=True)

                # Color Mapping (Optional, requires Plotly)
                try: import plotly.express as px_colors
                except ImportError: px_colors = None; logger.warning("Plotly Express not available for color mapping.")
                all_names = pd.concat([top_rev['name'], top_prof['name']]).unique()
                color_map = {}
                if px_colors:
                    color_map = {name: px_colors.colors.qualitative.Plotly[i % len(px_colors.colors.qualitative.Plotly)] for i, name in enumerate(all_names) if name != 'Other'}
                    color_map['Other'] = 'lightgrey' # Specific color for 'Other'

                # Store results (filter out zero/negative slices for cleaner pies)
                analysis_results['pie_data'] = {
                    'revenue': top_rev[top_rev['totalPrice'] > 1e-6].copy(),
                    'profit': top_prof[top_prof['netProfit'] > 1e-6].copy() if not top_prof.empty else pd.DataFrame(),
                    'color_mapping': color_map
                }
                logger.info("Pie chart data analysis completed.")
            except Exception as e: logger.error(f"Error during Pie Data analysis: {e}", exc_info=True); analysis_results['pie_data'] = {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}}
        else: logger.warning("Skipping Pie data analysis (Product flow empty or missing required columns).")

        # --- Stagnant Products Analysis ---
        if not pf.empty and 'days_since_last_sale' in pf.columns:
            try:
                logger.info("Analyzing stagnant products...")
                stagnant_config = config.analysis_params.get('stagnant_periods', {})
                stagnant_bins = stagnant_config.get('bins')
                stagnant_labels = stagnant_config.get('labels')
                apply_categories = False

                if not stagnant_bins or not stagnant_labels or len(stagnant_bins) != len(stagnant_labels) + 1:
                    logger.error("Stagnant periods config invalid. Using default threshold 90 days.")
                    stagnant_threshold = 90 # Fallback threshold
                else:
                    stagnant_threshold = stagnant_bins[0] # Use the first bin edge as the minimum stagnant period
                    apply_categories = True

                # Filter products that haven't sold for >= threshold days but have sold at some point (< 99999)
                stagnant_df = pf[(pf['days_since_last_sale'] >= stagnant_threshold) & (pf['days_since_last_sale'] < 99999)].copy()

                if not stagnant_df.empty:
                    if apply_categories:
                        stagnant_df['days_category'] = pd.cut(stagnant_df['days_since_last_sale'], bins=stagnant_bins, labels=stagnant_labels, right=False)
                        # Handle potential CategoricalDtype issues by converting to string
                        if pd.api.types.is_categorical_dtype(stagnant_df['days_category']):
                            stagnant_df['days_category'] = stagnant_df['days_category'].astype(str)
                    else: # Fallback if config was invalid
                        stagnant_df['days_category'] = f'>= {stagnant_threshold} days'

                    # Add helper columns for display
                    if 'last_sale_date' in stagnant_df.columns:
                        stagnant_df['last_sale_date_str'] = stagnant_df['last_sale_date'].dt.strftime('%Y-%m-%d').fillna('N/A')
                    sid_prod_id_col_name = config.required_columns['sale_invoices_details'][1] # Get product id column name
                    if sid_prod_id_col_name in stagnant_df.columns:
                        stagnant_df['product_id'] = stagnant_df[sid_prod_id_col_name] # Use the correct column name
                    else:
                        stagnant_df['product_id'] = 'N/A' # Fallback if column not found

                    analysis_results['stagnant_products'] = stagnant_df.sort_values('days_since_last_sale', ascending=False)
                    logger.info(f"Stagnant products analysis completed ({len(stagnant_df)} found).")
                else:
                    logger.info("No stagnant products found matching the criteria.")
                    analysis_results['stagnant_products'] = pd.DataFrame()
            except Exception as e: logger.error(f"Error during Stagnant Products analysis: {e}", exc_info=True); analysis_results['stagnant_products'] = pd.DataFrame()
        else: logger.warning("Skipping Stagnant products analysis (Product flow empty or missing days_since_last_sale).")

        # --- Outstanding Amounts Analysis ---
        if deferred is not None and not deferred.empty:
            try:
                logger.info("Analyzing outstanding amounts...")
                # Get column names from config
                def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col, _ = config.required_columns['invoice_deferreds']

                # Check if all required columns exist
                if all(col in deferred.columns for col in [def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col]):
                    buy_invoice_str = "Stocks\\Models\\BuyInvoice" # Check if this string matches exactly
                    status_vals = [0, 2] # Status values indicating outstanding/partially paid

                    # Filter for relevant buy invoices with outstanding status
                    filtered = deferred[(deferred[def_type_col] == buy_invoice_str) & (deferred[def_status_col].isin(status_vals))].copy()

                    if not filtered.empty:
                        # Calculate the outstanding amount
                        filtered["outstanding_amount"] = filtered[def_amount_col] - filtered[def_paid_col]
                        # Group by supplier (user_id) and sum outstanding amounts
                        outstanding_grouped = filtered.groupby(def_user_id_col, as_index=False)["outstanding_amount"].sum()
                        # Keep only suppliers with a positive outstanding amount
                        analysis_results['outstanding_amounts'] = outstanding_grouped[outstanding_grouped['outstanding_amount'] > 1e-6].copy()
                        logger.info(f"Outstanding amounts calculated ({len(analysis_results['outstanding_amounts'])} suppliers).")
                    else:
                        logger.info("No outstanding buy invoices found matching the criteria.")
                        analysis_results['outstanding_amounts'] = pd.DataFrame()
                else:
                    logger.warning("Skipping Outstanding amounts: Missing required columns in deferred data.")
                    analysis_results['outstanding_amounts'] = pd.DataFrame()
            except Exception as e: logger.error(f"Error during Outstanding Amounts analysis: {e}", exc_info=True); analysis_results['outstanding_amounts'] = pd.DataFrame()
        else: logger.info("Skipping Outstanding amounts: No deferred data available.")

        # --- Monthly Average Invoice Time Series ---
        if si is not None and not si.empty and 'year_month' in si.columns:
            si_total_col = config.required_columns['sale_invoices'][2]
            if si_total_col in si.columns:
                try:
                    logger.info("Calculating monthly average invoice time series...")
                    monthly_avg = si.groupby('year_month')[si_total_col].mean()
                    monthly_avg.index = pd.to_datetime(monthly_avg.index + '-01') # Convert 'YYYY-MM' to datetime start of month
                    if not monthly_avg.empty:
                        min_date_monthly = monthly_avg.index.min(); max_date_monthly = monthly_avg.index.max()
                        logger.info(f"Monthly TS: Original range {min_date_monthly.strftime('%Y-%m')} to {max_date_monthly.strftime('%Y-%m')}")
                        full_idx = pd.date_range(start=min_date_monthly, end=max_date_monthly, freq='MS') # 'MS' for Month Start frequency
                        # Reindex to ensure all months are present, fill missing months with 0
                        monthly_avg_ts_final = monthly_avg.reindex(full_idx).fillna(0)
                        analysis_results['monthly_avg_invoice_ts'] = monthly_avg_ts_final
                        logger.info(f"Monthly average invoice TS calculated and reindexed/filled. Final range {full_idx.min().strftime('%Y-%m')} to {full_idx.max().strftime('%Y-%m')}, length {len(monthly_avg_ts_final)}.")
                    else:
                        logger.warning("Monthly TS calculation resulted in empty series after grouping.")
                        analysis_results['monthly_avg_invoice_ts'] = pd.Series(dtype='float64')
                except Exception as e:
                    logger.error(f"Error calculating Monthly Avg Invoice TS: {e}", exc_info=True)
                    analysis_results['monthly_avg_invoice_ts'] = pd.Series(dtype='float64')
            else:
                logger.warning("Skipping Monthly TS: Missing total price column in sale invoices.")
                analysis_results['monthly_avg_invoice_ts'] = pd.Series(dtype='float64')
        else:
            logger.warning("Skipping Monthly TS: Sale invoices empty or missing 'year_month'.")
            analysis_results['monthly_avg_invoice_ts'] = pd.Series(dtype='float64')

        # --- Daily Profit Time Series (Optional - for potential other uses) ---
        if sid is not None and not sid.empty and 'created_at_per_day_obj' in sid.columns and 'netProfit' in sid.columns:
            try:
                logger.info("Calculating daily profit time series (for potential future use)...")
                daily_profit_raw = sid.groupby('created_at_per_day_obj')['netProfit'].sum()
                if not daily_profit_raw.empty:
                    daily_profit_raw.index = pd.to_datetime(daily_profit_raw.index) # Ensure datetime index
                    min_date_prof = daily_profit_raw.index.min(); max_date_prof = daily_profit_raw.index.max()
                    full_idx_prof = pd.date_range(start=min_date_prof, end=max_date_prof, freq='D') # Daily frequency
                    # Reindex and fill missing days with 0 profit
                    daily_profit_ts_final = daily_profit_raw.reindex(full_idx_prof).fillna(0)
                    analysis_results['daily_profit_ts'] = daily_profit_ts_final
                    logger.info(f"Daily profit TS calculated. Range: {min_date_prof.date()} to {max_date_prof.date()}, length {len(daily_profit_ts_final)}.")
                else:
                    logger.warning("Daily profit aggregation resulted in empty series.")
                    analysis_results['daily_profit_ts'] = pd.Series(dtype='float64')
            except Exception as e_dpts:
                logger.error(f"Error calculating Daily Profit TS: {e_dpts}", exc_info=True)
                analysis_results['daily_profit_ts'] = pd.Series(dtype='float64')
        else:
             logger.warning("Skipping Daily Profit TS calculation: SID empty or missing required columns ('created_at_per_day_obj', 'netProfit').")
             analysis_results['daily_profit_ts'] = pd.Series(dtype='float64')


        logger.info("Data analysis completed.")
        self.analytics = analysis_results # Store results in the instance variable

    def _run_forecasting(self):
        """تشغيل هندسة الميزات والتنبؤ اليومي."""
        if not FORECASTING_AVAILABLE:
            logger.warning("Daily forecasting modules unavailable. Skipping.")
            self.forecast_data = pd.DataFrame()
            return

        logger.info("--- Starting Daily Forecasting Pipeline ---")
        self.forecast_data = pd.DataFrame() # Reset

        features_input_dict = {
            'sale_invoices': self.processed_data.get('sale_invoices'),
            'sale_invoices_details': self.processed_data.get('sale_invoices_details')
        }
        if features_input_dict['sale_invoices'] is None or features_input_dict['sale_invoices_details'] is None \
           or features_input_dict['sale_invoices'].empty or features_input_dict['sale_invoices_details'].empty:
            logger.error("Missing or empty processed data for daily feature engineering. Skipping daily forecast.")
            return

        try:
            logger.info("Calling generate_features_df (for daily forecast)...")
            features_df_daily = generate_features_df(features_input_dict)

            if features_df_daily is None or features_df_daily.empty:
                logger.error("Daily feature engineering failed or produced empty DataFrame. Skipping daily forecast.")
                return

            # --- *** Minimum Observations Check for Daily Forecast *** ---
            min_daily_obs = config.analysis_params.get('min_daily_obs_for_forecast', 360) # Get from config
            if len(features_df_daily) < min_daily_obs:
                logger.warning(
                    f"Daily historical data ({len(features_df_daily)} days after feature eng.) "
                    f"is less than the required minimum ({min_daily_obs}). Skipping daily forecast."
                )
                self.forecast_data = pd.DataFrame() # Ensure it's empty
                return # Exit the forecasting process
            # --- *** END Check *** ---

            logger.info(f"Daily features generated successfully. Shape: {features_df_daily.shape}")
            logger.info(f"Calling train_and_forecast (Daily - Horizon={self.forecast_horizon_daily}, Seasonality={self.seasonal_period_daily})...")
            forecast_result_daily = train_and_forecast(
                features_df=features_df_daily,
                forecast_horizon=self.forecast_horizon_daily,
                seasonal_period=self.seasonal_period_daily
            )

            if forecast_result_daily is None:
                logger.error("Daily training and forecasting process failed (returned None).")
                self.forecast_data = pd.DataFrame()
            elif forecast_result_daily.empty:
                logger.warning("Daily training and forecasting completed, but produced an empty forecast DataFrame.")
                self.forecast_data = forecast_result_daily # Assign empty DF
            else:
                logger.info(f"Daily forecasts generated successfully. Shape: {forecast_result_daily.shape}")
                # Use the standard DATE_COLUMN_OUTPUT for checking columns
                required_cols = [DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci']
                if not all(col in forecast_result_daily.columns for col in required_cols):
                    missing_cols = list(set(required_cols) - set(forecast_result_daily.columns))
                    logger.error(f"Daily forecast DataFrame is missing required columns: {missing_cols}")
                    self.forecast_data = pd.DataFrame() # Set to empty if columns missing
                else:
                    self.forecast_data = forecast_result_daily.copy()

        except (ValueError, RuntimeError) as daily_fc_err:
             logger.error(f"Handled error in daily forecasting pipeline: {daily_fc_err}", exc_info=True)
             self.forecast_data = pd.DataFrame() # Ensure empty on error
        except Exception as e:
            logger.error(f"Unexpected general error during daily forecasting: {e}", exc_info=True)
            self.forecast_data = pd.DataFrame() # Ensure empty on error

        logger.info(f"--- Finished Daily Forecasting Pipeline. Forecast data shape: {self.forecast_data.shape} ---")

    def _run_monthly_forecasting(self):
        """تشغيل هندسة الميزات والتنبؤ لمتوسط الفاتورة الشهرية."""
        if not MONTHLY_FORECASTING_AVAILABLE:
            logger.warning("Monthly forecasting modules unavailable. Skipping.")
            self.monthly_avg_invoice_forecast = pd.DataFrame()
            return

        logger.info("--- Starting Monthly Forecasting Pipeline ---")
        self.monthly_avg_invoice_forecast = pd.DataFrame() # Reset

        monthly_ts = self.analytics.get('monthly_avg_invoice_ts')
        if monthly_ts is None or monthly_ts.empty:
            logger.warning("Monthly average invoice time series unavailable. Skipping monthly forecast.")
            return

        # --- *** Minimum Observations Check for Monthly Forecast *** ---
        min_obs = config.analysis_params.get('min_monthly_obs', 12)
        # Check length of non-NaN values in the time series
        if len(monthly_ts.dropna()) < min_obs:
             logger.warning(f"Monthly time series has too few non-NaN observations ({len(monthly_ts.dropna())} < {min_obs}). Skipping monthly forecast.")
             self.monthly_avg_invoice_forecast = pd.DataFrame() # Ensure empty
             return # Exit forecasting
        # --- *** END Check *** ---

        try:
            logger.info("Calling generate_monthly_features...")
            monthly_features_df = generate_monthly_features(monthly_ts)

            if monthly_features_df is None or monthly_features_df.empty:
                logger.error("Monthly feature engineering failed or produced empty DataFrame. Skipping monthly forecast.")
                return

            logger.info(f"Calling train_and_forecast_monthly (Horizon={self.forecast_horizon_monthly}, Seasonality={self.seasonal_period_monthly})...")
            forecast_result_monthly = train_and_forecast_monthly(
                monthly_features_df=monthly_features_df,
                forecast_horizon_months=self.forecast_horizon_monthly,
                seasonal_period=self.seasonal_period_monthly
            )

            if forecast_result_monthly is None:
                logger.error("Monthly training and forecasting process failed (returned None).")
                self.monthly_avg_invoice_forecast = pd.DataFrame()
            elif forecast_result_monthly.empty:
                 logger.warning("Monthly training and forecasting produced empty DataFrame.")
                 self.monthly_avg_invoice_forecast = forecast_result_monthly # Assign empty DF
            else:
                 logger.info(f"Monthly forecasts generated successfully. Shape: {forecast_result_monthly.shape}")
                 required_cols = [DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci'] # Use standard output name
                 if not all(col in forecast_result_monthly.columns for col in required_cols):
                      missing_cols = list(set(required_cols) - set(forecast_result_monthly.columns))
                      logger.error(f"Monthly forecast DataFrame is missing required columns: {missing_cols}")
                      self.monthly_avg_invoice_forecast = pd.DataFrame() # Set to empty if columns missing
                 else:
                      self.monthly_avg_invoice_forecast = forecast_result_monthly.copy()

        except (ValueError, RuntimeError) as monthly_fc_err:
             logger.error(f"Handled error in monthly forecasting pipeline: {monthly_fc_err}", exc_info=True)
             self.monthly_avg_invoice_forecast = pd.DataFrame() # Ensure empty on error
        except Exception as e:
            logger.error(f"Unexpected general error during monthly forecasting: {e}", exc_info=True)
            self.monthly_avg_invoice_forecast = pd.DataFrame() # Ensure empty on error

        logger.info(f"--- Finished Monthly Forecasting Pipeline. Monthly forecast data shape: {self.monthly_avg_invoice_forecast.shape} ---")

    # --- **** START: HELPER FOR LOCALIZED METADATA **** ---
    def _get_chart_metadata(self, fig_key: str, lang: str, chart_type: ChartType,
                            x_name: str = "", x_type_key: str = "category", x_title_key: Optional[str] = None,
                            y_name: str = "", y_type_key: str = "number", y_title_key: Optional[str] = None,
                            series_info: Optional[List[Dict[str, Any]]] = None) -> ChartMetadata:
        """Helper to generate ChartMetadata with localized titles and descriptions."""
        title = get_translation(lang, f'{fig_key}.title', f'{fig_key.upper()} Title')
        description = get_translation(lang, f'{fig_key}.description', f'{fig_key.upper()} Description')
        freq_key = get_translation(lang, f'{fig_key}.frequency', 'N/A')
        freq_display = get_translation(lang, f'frequencies.{freq_key}', freq_key.capitalize())

        x_axis_type = get_translation(lang, f'axis_types.{x_type_key}', x_type_key)
        x_axis_title_lookup = f'{fig_key}.x_axis.title' if x_title_key is None else f'{fig_key}.x_axis.{x_title_key}'
        x_axis_title = get_translation(lang, x_axis_title_lookup, default=get_translation(lang, f'{fig_key}.x_axis.title', x_name.replace('_', ' ').title()))

        y_axis_type = get_translation(lang, f'axis_types.{y_type_key}', y_type_key)
        y_axis_title_lookup = f'{fig_key}.y_axis.title' if y_title_key is None else f'{fig_key}.y_axis.{y_title_key}'
        y_axis_title = get_translation(lang, y_axis_title_lookup, default=get_translation(lang, f'{fig_key}.y_axis.title', y_name.replace('_', ' ').title()))

        generated_series: List[SeriesInfo] = []
        if series_info:
            for s_info in series_info:
                if isinstance(s_info, dict):
                    series_title_key = s_info.get("title_key", s_info.get("internal_name", ""))
                    series_title_lookup = f'{fig_key}.series.{series_title_key}'
                    series_name_loc = get_translation(lang, series_title_lookup, s_info.get('internal_name', '').replace('_', ' ').title())
                    generated_series.append(
                        SeriesInfo(
                            name=series_name_loc, # Localized name
                            color=s_info.get('color'),
                            type=s_info.get('type')
                        )
                    )
                elif isinstance(s_info, SeriesInfo):
                     logger.warning(f"Received SeriesInfo object instead of dict in _get_chart_metadata for {fig_key}. Using name directly.")
                     generated_series.append(s_info)
                else:
                     logger.error(f"Invalid type in series_info for {fig_key}: {type(s_info)}. Expected dict.")


        return ChartMetadata(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Add timestamp here
            title=title, description=description, frequency=freq_display, chart_type=chart_type,
            x_axis=AxisInfo(name=x_name, type=x_axis_type, title=x_axis_title),
            y_axis=AxisInfo(name=y_name, type=y_axis_type, title=y_axis_title),
            series=generated_series
        )
    # --- **** END: HELPER FOR LOCALIZED METADATA **** ---

    # --- **** START: HELPER FOR ERROR ChartData **** ---
    def _get_error_chart_data(self, fig_key: str, lang: str, error_key: str, default_error_msg: str, chart_type: ChartType = ChartType.TABLE) -> ChartData:
        """Helper to create a ChartData object with an error message."""
        metadata = self._get_chart_metadata(fig_key, lang, chart_type) # Get metadata for context
        error_message = get_translation(lang, f"{fig_key}.errors.{error_key}", default_error_msg)
        error_data = {"error": True, "message": error_message}
        logger.warning(f"[{fig_key}] (lang={lang}) Condition not met: {error_key} - {error_message}")
        # Ensure the returned object conforms to ChartData structure
        return ChartData(metadata=metadata, data=error_data)
    # --- **** END: HELPER FOR ERROR ChartData **** ---


    # --- **** START: PREPARE FUNCTIONS **** ---
    # --- Modified Prepare Functions with Error Handling ---

    def _prepare_fig1_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData: # Return ChartData guaranteed
        fig_key = 'fig1'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [ {'internal_name': 'current_stock', 'title_key': 'stock', 'color': 'rgb(55, 83, 109)'}, {'internal_name': 'sales_quantity', 'title_key': 'sales', 'color': 'rgb(26, 118, 255)'} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='value', y_type_key='number', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if 'sales_quantity' not in product_flow.columns or product_flow['sales_quantity'].sum() <= 0: return self._get_error_chart_data(fig_key, lang, "no_sales", "No sales recorded for any product.", chart_type)
        required_cols = ['name', 'current_stock', 'sales_quantity'];
        if not all(col in product_flow.columns for col in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns.", chart_type)

        min_sales_days_for_reliable_top20 = 30
        if 'last_sale_date' in product_flow.columns:
           sales_dates = product_flow['last_sale_date'].dropna()
           if not sales_dates.empty:
               date_range_days = (sales_dates.max() - sales_dates.min()).days
               if date_range_days < min_sales_days_for_reliable_top20:
                    logger.warning(f"[{fig_key}] Sales history spans less than {min_sales_days_for_reliable_top20} days.")
                    # return self._get_error_chart_data(fig_key, lang, "insufficient_sales_period", f"Insufficient sales history.", chart_type) # Uncomment if blocking

        try:
            pf = product_flow; top20 = pf.nlargest(20, 'sales_quantity')
            if top20.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "No products found for Top 20.", chart_type)
            stock_series_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig1.series.stock', 'Stock')), 'Current Stock')
            sales_series_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig1.series.sales', 'Sales')), 'Sales')
            x_axis_title = metadata.x_axis.title
            data = top20[['name', 'current_stock', 'sales_quantity']].rename(columns={'name': x_axis_title, 'current_stock': stock_series_name, 'sales_quantity': sales_series_name}).to_dict(orient='records')
            data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data]
            logger.info(f"Prepared {fig_key} (lang={lang}). Data length: {len(data)}")
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig2_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig2'; chart_type = ChartType.PIE
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        color_map = {'Undersupplied': '#ff7f0e', 'Balanced': '#2ca02c', 'Oversupplied': '#d62728', 'N/A': 'grey'}
        series_pie_structure = []; eff_labels = config.analysis_params.get('efficiency_labels', []) + ['N/A']
        temp_pf = product_flow if product_flow is not None else pd.DataFrame()
        if 'efficiency' in temp_pf.columns:
            for label in eff_labels:
                if label in temp_pf['efficiency'].unique(): series_pie_structure.append({'internal_name': label, 'title_key': f'efficiency_labels.{label}', 'color': color_map.get(label, 'grey')})
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name="category", y_name="value", series_info=series_pie_structure)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if 'efficiency' not in product_flow.columns: return self._get_error_chart_data(fig_key, lang, "calculation_error", "Efficiency calculation failed.", chart_type)
        # Condition: Must have at least one product with efficiency calculated
        if len(product_flow) == 0 or product_flow['efficiency'].nunique() == 0 or (product_flow['efficiency'].nunique() == 1 and product_flow['efficiency'].iloc[0] == 'N/A'):
             return self._get_error_chart_data(fig_key, lang, "no_efficiency_data", "Could not calculate inventory efficiency for any product.", chart_type)


        try:
            pf = product_flow; valid_cats_internal = config.analysis_params['efficiency_labels']
            pf['efficiency'] = pd.Categorical(pf['efficiency'], categories=valid_cats_internal + ['N/A'], ordered=False)
            efficiency_counts = pf['efficiency'].value_counts()
            # Check counts after calculating
            if efficiency_counts.empty or efficiency_counts.sum() == 0: return self._get_error_chart_data(fig_key, lang, "no_efficiency_data", "No products with calculated efficiency status found.", chart_type)
            localized_labels = [get_translation(lang, f'efficiency_labels.{label}', str(label)) for label in efficiency_counts.index]
            values = efficiency_counts.values.tolist()
            data = {'labels': localized_labels, 'values': values, 'colors': [color_map.get(str(label), 'grey') for label in efficiency_counts.index]}
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig3_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig3'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [ {'internal_name': 'sales_amount', 'title_key': 'revenue', 'color': 'rgb(26, 118, 255)'}, {'internal_name': 'COGS', 'title_key': 'cogs', 'color': 'rgb(255, 127, 14)'}, {'internal_name': 'Margin', 'title_key': 'margin', 'color': 'rgb(44, 160, 44)'} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='value', y_type_key='number', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        required_cols = ['name', 'sales_amount', 'sales_quantity', 'buyPrice']
        if not all(c in product_flow.columns for c in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns for revenue/COGS analysis.", chart_type)
        if product_flow['sales_amount'].sum() <= 0: return self._get_error_chart_data(fig_key, lang, "no_sales", "No revenue recorded for any product.", chart_type)
        if product_flow['buyPrice'].isnull().all(): return self._get_error_chart_data(fig_key, lang, "missing_cost_data", "Cost price (buyPrice) is missing for all products.", chart_type)
        # Condition: At least one product needed
        if len(product_flow) == 0: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "No product data available.", chart_type) # Condition from report


        try:
            pf = product_flow; top20 = pf.nlargest(20, 'sales_amount').copy()
            if top20.empty: return self._get_error_chart_data(fig_key, lang, "no_sales", "No products found in Top 20 by revenue.", chart_type)
            top20['COGS'] = top20['sales_quantity'] * top20['buyPrice']
            top20['Margin'] = top20['sales_amount'] - top20['COGS']
            x_axis_title = metadata.x_axis.title
            series_rev_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig3.series.revenue', 'Revenue')), 'Revenue')
            series_cogs_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig3.series.cogs', 'COGS')), 'COGS')
            series_margin_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig3.series.margin', 'Margin')), 'Margin')
            data_df = top20[['name', 'sales_amount', 'COGS', 'Margin']].rename(columns={ 'name': x_axis_title, 'sales_amount': series_rev_name, 'COGS': series_cogs_name, 'Margin': series_margin_name })
            data_list = data_df.to_dict(orient='records')
            data_list = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data_list]
            return ChartData(metadata=metadata, data=data_list)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig4_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig4'; chart_type = ChartType.COMBO
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [ {'internal_name': 'current_stock', 'title_key': 'stock', 'color': 'lightblue', 'type': ChartType.BAR}, {'internal_name': 'sales_quantity', 'title_key': 'sales', 'color': 'orange', 'type': ChartType.LINE} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='value', y_type_key='number', series_info=series_info)

        # Conditions are same as Fig 1
        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if 'sales_quantity' not in product_flow.columns or product_flow['sales_quantity'].sum() <= 0: return self._get_error_chart_data(fig_key, lang, "no_sales", "No sales recorded for any product.", chart_type)
        required_cols = ['name', 'current_stock', 'sales_quantity'];
        if not all(col in product_flow.columns for col in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns.", chart_type)
        min_sales_days_for_reliable_top20 = 30
        if 'last_sale_date' in product_flow.columns:
           sales_dates = product_flow['last_sale_date'].dropna()
           if not sales_dates.empty:
               date_range_days = (sales_dates.max() - sales_dates.min()).days
               if date_range_days < min_sales_days_for_reliable_top20:
                   logger.warning(f"[{fig_key}] Sales history spans less than {min_sales_days_for_reliable_top20} days.")
                   # return self._get_error_chart_data(fig_key, lang, "insufficient_sales_period", f"Insufficient sales history.", chart_type)

        try:
             pf = product_flow; top20_sales = pf.nlargest(20, 'sales_quantity')
             if top20_sales.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "No products found for Top 20.", chart_type)
             stock_series = next((s.name for s in metadata.series if s.type == ChartType.BAR), 'Current Stock')
             sales_series = next((s.name for s in metadata.series if s.type == ChartType.LINE), 'Sales')
             x_axis_title = metadata.x_axis.title
             data = top20_sales[['name', 'current_stock', 'sales_quantity']].rename( columns={'name': x_axis_title, 'current_stock': stock_series, 'sales_quantity': sales_series} ).to_dict(orient='records');
             data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data];
             logger.info(f"Prepared {fig_key} (lang={lang}). Data length: {len(data)}")
             return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig5_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig5'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [ {'internal_name': 'current_stock', 'title_key': 'stock', 'color': 'rgb(55, 83, 109)'}, {'internal_name': 'sales_quantity', 'title_key': 'sales', 'color': 'rgb(26, 118, 255)'} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='efficiency', x_type_key='category', y_name='value', y_type_key='number', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if not all(c in product_flow.columns for c in ['efficiency', 'current_stock', 'sales_quantity']): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns for efficiency aggregation.", chart_type)

        try:
            pf = product_flow; valid_cats_internal = config.analysis_params['efficiency_labels'];
            eff_group = pf[pf['efficiency'].isin(valid_cats_internal)].groupby('efficiency', observed=False)[['current_stock', 'sales_quantity']].sum().reset_index()
            if eff_group.empty or eff_group[['current_stock', 'sales_quantity']].sum().sum() <= 0: return self._get_error_chart_data(fig_key, lang, "no_efficiency_data", "No data found for efficiency categories.", chart_type)
            x_axis_title_display = metadata.x_axis.title
            eff_group[x_axis_title_display] = eff_group['efficiency'].apply(lambda x: get_translation(lang, f'efficiency_labels.{x}', str(x)))
            stock_series_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig5.series.stock', 'Stock')), 'Total Stock')
            sales_series_name = next((s.name for s in metadata.series if s.name == get_translation(lang, 'fig5.series.sales', 'Sales')), 'Total Sales')
            data_df = eff_group.rename( columns={'current_stock': stock_series_name, 'sales_quantity': sales_series_name} )
            data = data_df[[x_axis_title_display, stock_series_name, sales_series_name]].to_dict(orient='records')
            data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data];
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig6_data(self, pareto_data: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig6'; chart_type = ChartType.COMBO
        logger.info(f"[{fig_key}] Lang '{lang}': Preparing figure (Pareto)...")
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info_structure = [ {'internal_name': 'sales_quantity', 'title_key': 'sales', 'color': 'cornflowerblue', 'type': ChartType.BAR}, {'internal_name': 'cumulative_percentage', 'title_key': 'cumulative', 'color': 'red', 'type': ChartType.LINE} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='sales_quantity', y_type_key='number', series_info=series_info_structure)

        if pareto_data is None or pareto_data.empty: return self._get_error_chart_data(fig_key, lang, "no_pareto_data", "Pareto analysis data is missing (likely no sales).", chart_type)
        required_cols = ['name', 'sales_quantity', 'cumulative_percentage'];
        if not all(c in pareto_data.columns for c in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns for Pareto analysis.", chart_type)
        # Condition: At least one product needed for Pareto
        if len(pareto_data) == 0: return self._get_error_chart_data(fig_key, lang, "no_sales_quantity", "No products found with sales quantity > 0 for Pareto analysis.", chart_type)


        try:
             pareto = pareto_data.copy(); x_axis_title_display = metadata.x_axis.title
             sales_series_name = next((s.name for s in metadata.series if s.type == ChartType.BAR), 'Sales Quantity')
             cum_series_name = next((s.name for s in metadata.series if s.type == ChartType.LINE), 'Cumulative %')
             data_df = pareto.rename(columns={'name': x_axis_title_display, 'sales_quantity': sales_series_name, 'cumulative_percentage': cum_series_name})
             data_df_final = data_df[[x_axis_title_display, sales_series_name, cum_series_name]].copy()
             data_df_final['original_cumulative'] = pareto['cumulative_percentage'].copy()
             bins = list(np.arange(0, 80.1, 10));
             if not bins or bins[-1] < 80: bins.append(80.1);
             bins = sorted(list(set(bins)))
             labels = [f"{int(bins[i])}-{int(bins[i+1])}%" for i in range(len(bins)-1)] if len(bins) >= 2 else []
             cat_key_json = "percentage_category"
             if labels:
                  data_df_final['original_cumulative'] = pd.to_numeric(data_df_final['original_cumulative'], errors='coerce')
                  data_df_final[cat_key_json] = pd.cut(data_df_final['original_cumulative'], bins=bins, labels=labels, right=True, include_lowest=True)
                  data_df_final[cat_key_json] = data_df_final[cat_key_json].cat.add_categories('Other').fillna('Other')
                  data_df_final[cat_key_json] = data_df_final[cat_key_json].astype(str)
             else: data_df_final[cat_key_json] = 'Other'
             data_list = data_df_final.drop(columns=['original_cumulative']).to_dict(orient='records')
             cleaned_data = []
             for row in data_list:
                 cleaned_row = {}
                 for k, v in row.items():
                     if pd.isna(v): cleaned_row[k] = None
                     elif k == cum_series_name:
                         try: cleaned_row[k] = round(float(v), 2)
                         except (ValueError, TypeError): cleaned_row[k] = str(v)
                     elif isinstance(v, (int, float, np.number)): cleaned_row[k] = float(v)
                     else: cleaned_row[k] = str(v)
                 cleaned_data.append(cleaned_row)
             if not cleaned_data: return self._get_error_chart_data(fig_key, lang, "no_results_found", "No data remaining after processing.", chart_type)
             return ChartData(metadata=metadata, data=cleaned_data)
        except Exception as e:
            logger.error(f"[{fig_key}] Error preparing data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig7_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig7'; chart_type = ChartType.SCATTER
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [{'internal_name': 'product', 'title_key': 'products', 'color': 'blue'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='salePrice', x_type_key='number', y_name='sales_quantity', y_type_key='number', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        required_cols = ['salePrice', 'sales_quantity', 'name'];
        if not all(c in product_flow.columns for c in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns for scatter plot.", chart_type)

        try:
            pf = product_flow
            scatter_data = pf[(pf['salePrice'] > 0) & (pf['sales_quantity'] > 0)].copy()
            # Condition: Need at least one point with positive price and quantity
            if scatter_data.empty: return self._get_error_chart_data(fig_key, lang, "no_scatter_points", "No products found with positive sales price and quantity.", chart_type)
            x_axis_title_display = metadata.x_axis.title; y_axis_title_display = metadata.y_axis.title
            tooltip_name_key = get_translation(lang, 'common.product_name', 'Product Name')
            data_df = scatter_data[['name', 'salePrice', 'sales_quantity']].rename(columns={'name': tooltip_name_key, 'salePrice': x_axis_title_display, 'sales_quantity': y_axis_title_display})
            data = data_df[[tooltip_name_key, x_axis_title_display, y_axis_title_display]].to_dict(orient='records')
            data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data]
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig9_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig9'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        restock_threshold = config.analysis_params.get('restock_threshold', 10); restock_thr_str = str(restock_threshold)
        series_info = [{'internal_name': 'stock', 'title_key': 'stock', 'color': 'red'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='current_stock', y_type_key='number', series_info=series_info)
        metadata.title = get_translation(lang, f'{fig_key}.title', f'Products to Restock (<= {restock_thr_str})').replace("{threshold}", restock_thr_str)
        metadata.description = get_translation(lang, f'{fig_key}.description', f'Products with stock <= {restock_thr_str}').replace("{threshold}", restock_thr_str)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if not all(c in product_flow.columns for c in ['current_stock', 'name']): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns for restock analysis.", chart_type)

        try:
            pf = product_flow; restock = pf[pf['current_stock'] <= restock_threshold].sort_values('current_stock', ascending=True)
            # Condition: Need at least one product below threshold
            if restock.empty: return self._get_error_chart_data(fig_key, lang, "no_restock_needed", "No products found requiring restock (below threshold).", chart_type)
            x_axis_title = metadata.x_axis.title; y_axis_title = metadata.y_axis.title
            data = restock[['name', 'current_stock']].rename(columns={'name': x_axis_title, 'current_stock': y_axis_title}).to_dict(orient='records')
            data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data]
            logger.info(f"Prepared {fig_key} (lang={lang}). Data length: {len(data)}")
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig10_data(self, stagnant_products: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig10'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [{'internal_name': 'stagnancy', 'title_key': 'category'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='name', x_type_key='category', y_name='days_since_last_sale', y_type_key='number', series_info=series_info)

        # Check minimum history requirement first
        sid = self.processed_data.get('sale_invoices_details')
        min_history_days = 90 # 3 months approx
        if sid is not None and not sid.empty and 'created_at_dt' in sid.columns:
             sales_dates = sid['created_at_dt'].dropna()
             if not sales_dates.empty:
                  date_range_days = (sales_dates.max() - sales_dates.min()).days
                  if date_range_days < min_history_days: return self._get_error_chart_data(fig_key, lang, "insufficient_history", f"Insufficient history (<{min_history_days} days).", chart_type)
        # else: Could not check history, proceed cautiously

        # Check if stagnant analysis result is available and has data
        if stagnant_products is None: return self._get_error_chart_data(fig_key, lang, "no_stagnant_data", "Stagnant product analysis data is missing.", chart_type)
        if stagnant_products.empty: return self._get_error_chart_data(fig_key, lang, "no_stagnant_found", "No stagnant products found.", chart_type)
        required_cols = ['name', 'days_since_last_sale', 'days_category']
        if not all(col in stagnant_products.columns for col in required_cols): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns in stagnant data.", chart_type)

        try:
            stagnant_sorted = stagnant_products.sort_values('days_since_last_sale', ascending=False)
            x_axis_title_display = metadata.x_axis.title; y_axis_title_display = metadata.y_axis.title
            rename_map = {'name': x_axis_title_display, 'days_since_last_sale': y_axis_title_display}
            cols_to_select = ['name', 'days_since_last_sale']
            category_data_key = get_translation(lang, f'{fig_key}.data_keys.category', 'Stagnancy Period')
            stagnant_sorted[category_data_key] = stagnant_sorted['days_category'].apply(lambda x: get_translation(lang, f'stagnant_labels.{x}', str(x)))
            cols_to_select.append(category_data_key); rename_map['days_category'] = category_data_key
            # Corrected optional_cols_map definition
            optional_cols_map = {
                'last_sale_date_str': get_translation(lang, 'common.last_sale_date', 'Last Sale'),
                'current_stock': get_translation(lang, 'common.current_stock', 'Stock'),
                'product_id': get_translation(lang, 'common.product_id', 'ID')
            }
            for internal, translated in optional_cols_map.items():
                 if internal in stagnant_sorted.columns:
                     rename_map[internal] = translated
                     cols_to_select.append(internal)

            cols_to_select = [col for col in cols_to_select if col in stagnant_sorted.columns]
            data_df = stagnant_sorted[cols_to_select].rename(columns=rename_map)
            data = data_df.to_dict(orient='records');
            data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data];
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig11_data(self, daily_forecast: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig11'; chart_type = ChartType.LINE
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)
        series_info = [ {'internal_name': 'actual', 'title_key': 'actual', 'color': 'blue'}, {'internal_name': 'forecast', 'title_key': 'forecast', 'color': 'red'}, {'internal_name': 'ci', 'title_key': 'ci', 'color': 'rgba(255, 0, 0, 0.15)'} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name=FORECAST_DATE_COL, x_type_key='date', y_name='value', y_type_key='number', series_info=series_info)
        combined_data = []; actual_df = pd.DataFrame(); fc_df = pd.DataFrame()
        min_actual_days_needed = 1; min_daily_obs_for_fc = config.analysis_params.get('min_daily_obs_for_forecast', 360)
        actual_data_length_for_fc_check = 0
        try:
            sid = self.processed_data.get('sale_invoices_details')
            if sid is None or sid.empty: return self._get_error_chart_data(fig_key, lang, "no_details_data", "Sale invoice details data missing.", chart_type)
            sid_total_col = config.required_columns['sale_invoices_details'][4]; date_col_obj = 'created_at_per_day_obj'
            if date_col_obj not in sid.columns or sid_total_col not in sid.columns: return self._get_error_chart_data(fig_key, lang, "missing_columns", "Missing required date or total price columns.", chart_type)
            daily_actual_raw = sid.groupby(date_col_obj)[sid_total_col].sum()
            if daily_actual_raw.empty or len(daily_actual_raw) < min_actual_days_needed: return self._get_error_chart_data(fig_key, lang, "no_actual_sales", f"No actual daily sales data found.", chart_type)
            else:
                daily_actual_raw.index = pd.to_datetime(daily_actual_raw.index); min_date = daily_actual_raw.index.min(); max_date = daily_actual_raw.index.max()
                full_date_range = pd.date_range(start=min_date, end=max_date, freq='D'); daily_actual_complete = daily_actual_raw.reindex(full_date_range).fillna(0)
                actual_data_length_for_fc_check = len(daily_actual_complete)
                actual_df = daily_actual_complete.reset_index(); actual_df.columns = [FORECAST_DATE_COL, 'value']
                actual_df[FORECAST_DATE_COL] = actual_df[FORECAST_DATE_COL].dt.strftime('%Y-%m-%d')
                actual_df['type'] = 'actual'; actual_df['lower_ci'] = None; actual_df['upper_ci'] = None
                actual_df['value'] = actual_df['value'].astype(float).round(2); combined_data.extend(actual_df.to_dict(orient='records'))
            forecast_available = False; forecast_skipped_due_to_data = False
            if actual_data_length_for_fc_check < min_daily_obs_for_fc: forecast_skipped_due_to_data = True; logger.info(f"[{fig_key}] Skipping forecast check: actual data length {actual_data_length_for_fc_check} < {min_daily_obs_for_fc}")
            elif daily_forecast is not None and not daily_forecast.empty:
                required_fc_cols = [DATE_COLUMN_OUTPUT, 'forecast', 'lower_ci', 'upper_ci']
                if all(col in daily_forecast.columns for col in required_fc_cols):
                    fc_df_temp = daily_forecast[required_fc_cols].copy()
                    if not pd.api.types.is_string_dtype(fc_df_temp[DATE_COLUMN_OUTPUT]): fc_df_temp[DATE_COLUMN_OUTPUT] = pd.to_datetime(fc_df_temp[DATE_COLUMN_OUTPUT]).dt.strftime('%Y-%m-%d')
                    fc_df_temp.rename(columns={'forecast': 'value'}, inplace=True); fc_df_temp['type'] = 'forecast'
                    for col in ['value', 'lower_ci', 'upper_ci']: fc_df_temp[col] = pd.to_numeric(fc_df_temp[col], errors='coerce').fillna(0).astype(float).round(2)
                    fc_df = fc_df_temp; combined_data.extend(fc_df.to_dict(orient='records')); forecast_available = True
                else: logger.warning(f"[{fig_key}] Daily forecast DataFrame missing required columns.")
            else: logger.info(f"[{fig_key}] No daily forecast data provided (and actual data length was sufficient).")
            if not combined_data: return self._get_error_chart_data(fig_key, lang, "calculation_error", "Failed to combine data.", chart_type)
            combined_data = sorted(combined_data, key=lambda x: x.get(FORECAST_DATE_COL, ''))
            cleaned_combined_data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in combined_data]
            if forecast_skipped_due_to_data: metadata.description += " | " + get_translation(lang, 'fig11.errors.forecast_unavailable_data', f"Forecast requires approx. {min_daily_obs_for_fc} days of history.")
            logger.info(f"[{fig_key}] Prepared combined data. Total points: {len(cleaned_combined_data)}")
            return ChartData(metadata=metadata, data=cleaned_combined_data)
        except Exception as e: logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True); return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig12_data(self, monthly_avg_ts: Optional[pd.Series], monthly_forecast: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig12'; chart_type = ChartType.LINE
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        # --- Get Metadata (Use FORECAST_DATE_COL which is defined globally in the file) ---
        series_info = [ {'internal_name': 'actual', 'title_key': 'actual', 'color': 'royalblue'}, {'internal_name': 'forecast', 'title_key': 'forecast', 'color': 'firebrick'}, {'internal_name': 'ci', 'title_key': 'ci', 'color': 'rgba(255, 100, 100, 0.15)'} ]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name=FORECAST_DATE_COL, x_type_key='category', y_name='value', y_type_key='number', series_info=series_info)

        min_actual_months_needed = 2; min_fc_months_needed = config.analysis_params.get('min_monthly_obs', 12)
        actual_data_length_for_fc_check = 0

        if monthly_avg_ts is None or monthly_avg_ts.empty: return self._get_error_chart_data(fig_key, lang, "no_ts_data", "Monthly average invoice time series data is missing.", chart_type)
        actual_data_length_for_fc_check = len(monthly_avg_ts.dropna())
        if actual_data_length_for_fc_check < min_actual_months_needed: return self._get_error_chart_data(fig_key, lang, "insufficient_actual_months", f"Insufficient actual data: At least {min_actual_months_needed} months required.", chart_type)

        combined_data = []; actual_df = pd.DataFrame(); fc_df = pd.DataFrame()
        forecast_available = False; forecast_skipped_due_to_data = False

        try:
            ts_processed = monthly_avg_ts.copy()
            if not isinstance(ts_processed.index, pd.DatetimeIndex): ts_processed.index = pd.to_datetime(ts_processed.index)
            ts_processed = ts_processed.asfreq('MS').fillna(0) # Ensure MS freq and fill gaps
            actual_df = ts_processed.reset_index(); actual_df.columns = ['sort_date', 'value']
            actual_df['type'] = 'actual'; actual_df['lower_ci'] = None; actual_df['upper_ci'] = None
            actual_df['value'] = actual_df['value'].astype(float).round(2)
            actual_df[FORECAST_DATE_COL] = actual_df['sort_date'].dt.strftime('%Y-%m') # Use FORECAST_DATE_COL here
            combined_data.extend(actual_df[[FORECAST_DATE_COL, 'value', 'type', 'lower_ci', 'upper_ci']].to_dict(orient='records')) # Use FORECAST_DATE_COL here

            # Check forecast availability based on actual length BEFORE processing forecast df
            if actual_data_length_for_fc_check < min_fc_months_needed:
                forecast_skipped_due_to_data = True
                logger.info(f"[{fig_key}] Will skip using monthly forecast DF because actual data length ({actual_data_length_for_fc_check}) < required ({min_fc_months_needed}).")
            elif monthly_forecast is not None and not monthly_forecast.empty:
                required_fc_cols = [FORECAST_DATE_COL, 'forecast', 'lower_ci', 'upper_ci'] # Use FORECAST_DATE_COL here
                if all(c in monthly_forecast.columns for c in required_fc_cols):
                    fc_df_temp = monthly_forecast[required_fc_cols].copy()
                    if not pd.api.types.is_string_dtype(fc_df_temp[FORECAST_DATE_COL]): fc_df_temp[FORECAST_DATE_COL] = pd.to_datetime(fc_df_temp[FORECAST_DATE_COL]).dt.strftime('%Y-%m')
                    fc_df_temp.rename(columns={'forecast': 'value'}, inplace=True); fc_df_temp['type'] = 'forecast'
                    for col in ['value', 'lower_ci', 'upper_ci']: fc_df_temp[col] = pd.to_numeric(fc_df_temp[col], errors='coerce').fillna(0).astype(float).round(2)
                    fc_df = fc_df_temp; combined_data.extend(fc_df.to_dict(orient='records')); forecast_available = True
                else: logger.warning(f"[{fig_key}] Monthly forecast DataFrame missing required columns.")
            else: logger.info(f"[{fig_key}] No monthly forecast data provided or empty (and actual data length was sufficient).")


            if not combined_data: return self._get_error_chart_data(fig_key, lang, "calculation_error", "Failed to combine data.", chart_type)
            combined_data = sorted(combined_data, key=lambda x: x.get(FORECAST_DATE_COL, '')) # Use FORECAST_DATE_COL here
            cleaned_combined_data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in combined_data]
            if forecast_skipped_due_to_data: metadata.description += " | " + get_translation(lang, 'fig12.errors.forecast_unavailable_data', f"Forecast requires approx. {min_fc_months_needed} months of history.")
            logger.info(f"[{fig_key}] Prepared combined monthly data successfully. Total points: {len(cleaned_combined_data)}")
            return ChartData(metadata=metadata, data=cleaned_combined_data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig13_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig13'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [{'internal_name': 'sales', 'title_key': 'sales', 'color': 'value-based'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='sales_quantity', x_type_key='number', y_name='name', y_type_key='category', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if not all(c in product_flow.columns for c in ['sales_quantity', 'name']): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns.", chart_type)
        sold_products = product_flow[product_flow['sales_quantity'] > 0].copy()
        if sold_products.empty: return self._get_error_chart_data(fig_key, lang, "no_sold_products", "No products found with recorded sales.", chart_type)

        # --- *** NEW Condition Check: More than 10 sold products *** ---
        min_products_required = 10
        if len(sold_products) <= min_products_required:
            logger.warning(f"[{fig_key}] Only {len(sold_products)} products sold, which is not more than {min_products_required}.")
            return self._get_error_chart_data(fig_key, lang, "insufficient_product_count", f"Need more than {min_products_required} sold products to display this chart meaningfully.", chart_type)
        # --- *** END Condition Check *** ---

        try:
            bottom10 = sold_products.nsmallest(10, 'sales_quantity')
            if bottom10.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "Could not determine bottom selling products.", chart_type)
            x_axis_title_display = metadata.x_axis.title; y_axis_title_display = metadata.y_axis.title
            data_df = bottom10[['name', 'sales_quantity']].rename(columns={'name': y_axis_title_display, 'sales_quantity': x_axis_title_display})
            data = data_df.to_dict(orient='records'); data = [{k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else str(v))) for k, v in row.items()} for row in data]
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig14_data(self, sale_invoices_details: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig14'; chart_type = ChartType.TABLE
        logger.info(f"--- Entering _prepare_{fig_key}_data (lang={lang}) ---")
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        col1_header = get_translation(lang, f'{fig_key}.columns.quantity', 'Quantity Sold'); col2_header = get_translation(lang, f'{fig_key}.columns.num_products', 'Number of Products')
        series_info = [{'internal_name':'quantity', 'title_key':'quantity'}, {'internal_name':'num_products', 'title_key':'num_products'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, series_info=series_info); metadata.series = [SeriesInfo(name=col1_header), SeriesInfo(name=col2_header)]

        if sale_invoices_details is None or sale_invoices_details.empty: return self._get_error_chart_data(fig_key, lang, "no_details_data", "Sale invoice details data is missing or empty.", chart_type)
        sid_prod_id_col = config.required_columns['sale_invoices_details'][1]; sid_qty_col = config.required_columns['sale_invoices_details'][3]
        if not (sid_prod_id_col in sale_invoices_details.columns and sid_qty_col in sale_invoices_details.columns): return self._get_error_chart_data(fig_key, lang, "missing_columns", "Missing required product ID or quantity columns.", chart_type)

        try:
            sid_proc = sale_invoices_details; sales_per_product = sid_proc.groupby(sid_prod_id_col)[sid_qty_col].sum(); sales_per_product = sales_per_product[sales_per_product > 0]
            if sales_per_product.empty: return self._get_error_chart_data(fig_key, lang, "no_sales_found", "No products found with sales quantity > 0.", chart_type)
            min_variety_products = 20
            if len(sales_per_product) < min_variety_products:
                logger.warning(f"[{fig_key}] Found only {len(sales_per_product)} distinct products sold.")
                # return self._get_error_chart_data(fig_key, lang, "insufficient_variety", "Insufficient product variety.", chart_type) # Uncomment if blocking

            sales_freq = sales_per_product.value_counts().reset_index()
            if len(sales_freq.columns) == 2:
                sales_freq.columns = [col1_header, col2_header]
                sales_freq_table_data = sales_freq.sort_values(by=col1_header, ascending=True).head(15)
                if sales_freq_table_data.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "No frequency data to display.", chart_type)
                table_data_list = sales_freq_table_data.to_dict(orient='records'); table_data_list = [{k: (float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in row.items()} for row in table_data_list]
                return ChartData(metadata=metadata, data=table_data_list)
            else: return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error calculating sales frequency.", chart_type)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig15_data(self, product_flow: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig15'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [{'internal_name': 'sales', 'title_key': 'sales', 'color': 'value-based'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='sales_quantity', x_type_key='number', y_name='name', y_type_key='category', series_info=series_info)

        if product_flow is None or product_flow.empty: return self._get_error_chart_data(fig_key, lang, "no_product_flow", "Product flow data is missing or empty.", chart_type)
        if not all(c in product_flow.columns for c in ['sales_quantity', 'name']): return self._get_error_chart_data(fig_key, lang, "config_error", "Missing required columns.", chart_type)
        sold_products = product_flow[product_flow['sales_quantity'] > 0].copy();
        if sold_products.empty: return self._get_error_chart_data(fig_key, lang, "no_sold_products", "No products found with recorded sales.", chart_type)

        # --- *** NEW Condition Check: More than 10 sold products *** ---
        min_products_required = 10
        if len(sold_products) <= min_products_required:
            logger.warning(f"[{fig_key}] Only {len(sold_products)} products sold, which is not more than {min_products_required}.")
            return self._get_error_chart_data(fig_key, lang, "insufficient_product_count", f"Need more than {min_products_required} sold products to display this chart meaningfully.", chart_type)
        # --- *** END Condition Check *** ---

        try:
            top10 = sold_products.nlargest(10, 'sales_quantity')
            if top10.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "Could not determine top selling products.", chart_type)
            x_axis_title_display = metadata.x_axis.title; y_axis_title_display = metadata.y_axis.title
            data_df = top10[['name', 'sales_quantity']].rename(columns={'name': y_axis_title_display, 'sales_quantity': x_axis_title_display})
            data = data_df.to_dict(orient='records'); data = [{k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else str(v))) for k, v in row.items()} for row in data]
            return ChartData(metadata=metadata, data=data)
        except Exception as e:
            logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True)
            return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig16_data(self, pie_data_dict: Optional[Dict[str, Any]], lang: str) -> ChartData:
        fig_key = 'fig16'; chart_type = ChartType.PIE
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_structure_for_helper = [{'internal_name': 'revenue', 'title_key': 'revenue'}, {'internal_name': 'profit', 'title_key': 'profit'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name="label", y_name="value", series_info=series_structure_for_helper)

        if not pie_data_dict or 'revenue' not in pie_data_dict or 'profit' not in pie_data_dict: return self._get_error_chart_data(fig_key, lang, "no_pie_data", "Input data for pie charts (revenue/profit) is missing.", chart_type)
        rev_df = pie_data_dict.get('revenue'); prof_df = pie_data_dict.get('profit');
        no_revenue_data = not isinstance(rev_df, pd.DataFrame) or rev_df.empty or 'name' not in rev_df.columns or 'totalPrice' not in rev_df.columns
        no_profit_data = not isinstance(prof_df, pd.DataFrame) or prof_df.empty or 'name' not in prof_df.columns or 'netProfit' not in prof_df.columns
        if no_revenue_data and no_profit_data: return self._get_error_chart_data(fig_key, lang, "no_valid_data", "No valid revenue or profit data found for the Top 10 products.", chart_type)

        pf_for_pie = self.analytics.get('product_flow')
        min_products_for_meaningful_chart = 10
        if pf_for_pie is not None and len(pf_for_pie) < min_products_for_meaningful_chart:
            logger.warning(f"[{fig_key}] Pie charts based on less than {min_products_for_meaningful_chart} products.")
            # return self._get_error_chart_data(fig_key, lang, "too_few_products", "Too few products for meaningful shares.", chart_type) # Uncomment if blocking

        try:
            color_map = pie_data_dict.get('color_mapping', {}); revenue_data_list = []; profit_data_list = []
            if not no_revenue_data: revenue_data_list = rev_df[['name', 'totalPrice']].rename(columns={'name': 'label', 'totalPrice': 'value'}).to_dict(orient='records'); revenue_data_list = [{k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in row.items()} for row in revenue_data_list]
            if not no_profit_data: profit_data_list = prof_df[['name', 'netProfit']].rename(columns={'name': 'label', 'netProfit': 'value'}).to_dict(orient='records'); profit_data_list = [{k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in row.items()} for row in profit_data_list]
            if not revenue_data_list and not profit_data_list: return self._get_error_chart_data(fig_key, lang, "no_valid_data", "No valid revenue or profit data after processing.", chart_type)
            data = {'revenue_data': revenue_data_list, 'profit_data': profit_data_list, 'color_map': color_map}
            return ChartData(metadata=metadata, data=data)
        except Exception as e: logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True); return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)

    def _prepare_fig17_data(self, outstanding_amounts: Optional[pd.DataFrame], lang: str) -> ChartData:
        fig_key = 'fig17'; chart_type = ChartType.BAR
        if not MODELS_IMPORTED: return self._get_error_chart_data(fig_key, lang, "config_error", "Internal configuration error (models).", chart_type)

        series_info = [{'internal_name': 'amount', 'title_key': 'amount', 'color': 'value-based'}]
        metadata = self._get_chart_metadata(fig_key, lang, chart_type, x_name='user_id', x_type_key='category', x_title_key='title', y_name='outstanding_amount', y_type_key='number', y_title_key='title', series_info=series_info)

        if outstanding_amounts is None: return self._get_error_chart_data(fig_key, lang, "no_deferred_data", "Outstanding amounts data could not be calculated.", chart_type)
        if outstanding_amounts.empty: return self._get_error_chart_data(fig_key, lang, "no_outstanding_found", "No suppliers found with outstanding amounts.", chart_type)

        try:
            user_id_col = config.required_columns['invoice_deferreds'][4]; amount_col_out = 'outstanding_amount'
            if not (user_id_col in outstanding_amounts.columns and amount_col_out in outstanding_amounts.columns): return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error in calculated outstanding amounts.", chart_type)
            outstanding_filtered = outstanding_amounts[outstanding_amounts[amount_col_out] > 1e-6].copy()
            if outstanding_filtered.empty: return self._get_error_chart_data(fig_key, lang, "no_outstanding_found", "No suppliers found with outstanding amounts > 0.", chart_type)
            outstanding_sorted = outstanding_filtered.sort_values(amount_col_out, ascending=False).head(20)
            if outstanding_sorted.empty: return self._get_error_chart_data(fig_key, lang, "no_results_found", "No suppliers found for Top 20 outstanding.", chart_type)
            outstanding_sorted[user_id_col] = outstanding_sorted[user_id_col].astype(str)
            x_axis_title_display = metadata.x_axis.title; y_axis_title_display = metadata.y_axis.title
            data_df = outstanding_sorted[[user_id_col, amount_col_out]].rename(columns={ user_id_col: x_axis_title_display, amount_col_out: y_axis_title_display })
            data = data_df.to_dict(orient='records'); data = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in data]
            return ChartData(metadata=metadata, data=data)
        except Exception as e: logger.error(f"Error preparing {fig_key} data (lang={lang}): {e}", exc_info=True); return self._get_error_chart_data(fig_key, lang, "calculation_error", "Error during data preparation.", chart_type)


    # --- ** _save_results_to_disk & run_pipeline ** ---
    def _save_results_to_disk(self, client_id: int, analysis_data: dict, daily_forecast: pd.DataFrame, monthly_forecast: pd.DataFrame):
        """ يحفظ بيانات الرسوم البيانية المجهزة (ChartData) إلى ملفات JSON لكل لغة. """
        effective_languages = SUPPORTED_LANGUAGES if LOCALIZATION_AVAILABLE else ['en']
        logger.info(f"Saving results to disk for Client ID: {client_id} for languages: {effective_languages}")
        client_dir = RESULTS_BASE_DIR / str(client_id)
        try: client_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e: logger.error(f"Failed to create directory '{client_dir}': {e}"); raise RuntimeError(f"Cannot create results directory for client {client_id}")
        # Original V5.3 figure config (no fig8)
        figure_preparers_config = {
            "fig1": {"func": self._prepare_fig1_data, "inputs": [analysis_data.get('product_flow')]},
            "fig2": {"func": self._prepare_fig2_data, "inputs": [analysis_data.get('product_flow')]},
            "fig3": {"func": self._prepare_fig3_data, "inputs": [analysis_data.get('product_flow')]},
            "fig4": {"func": self._prepare_fig4_data, "inputs": [analysis_data.get('product_flow')]},
            "fig5": {"func": self._prepare_fig5_data, "inputs": [analysis_data.get('product_flow')]},
            "fig6": {"func": self._prepare_fig6_data, "inputs": [analysis_data.get('pareto_data')]},
            "fig7": {"func": self._prepare_fig7_data, "inputs": [analysis_data.get('product_flow')]},
            # fig8 intentionally omitted
            "fig9": {"func": self._prepare_fig9_data, "inputs": [analysis_data.get('product_flow')]},
            "fig10": {"func": self._prepare_fig10_data, "inputs": [analysis_data.get('stagnant_products')]},
            "fig11": {"func": self._prepare_fig11_data, "inputs": [daily_forecast]},
            "fig12": {"func": self._prepare_fig12_data, "inputs": [analysis_data.get('monthly_avg_invoice_ts'), monthly_forecast]},
            "fig13": {"func": self._prepare_fig13_data, "inputs": [analysis_data.get('product_flow')]},
            "fig14": {"func": self._prepare_fig14_data, "inputs": [self.processed_data.get('sale_invoices_details')]},
            "fig15": {"func": self._prepare_fig15_data, "inputs": [analysis_data.get('product_flow')]},
            "fig16": {"func": self._prepare_fig16_data, "inputs": [analysis_data.get('pie_data')]},
            "fig17": {"func": self._prepare_fig17_data, "inputs": [analysis_data.get('outstanding_amounts')]},
        }
        for lang in effective_languages:
            logger.info(f"Preparing and saving figures for language: '{lang}'")
            for fig_name, config_item in figure_preparers_config.items():
                preparer_func = config_item["func"]; inputs = config_item["inputs"]; start_prep_time = time.time()
                try:
                    chart_data = preparer_func(*inputs, lang=lang) # This now might return error structure
                    prep_time = time.time() - start_prep_time
                    if chart_data and isinstance(chart_data, ChartData):
                        file_path = client_dir / f"{fig_name}_{lang}.json"
                        try:
                            # Saving logic remains the same, it will save the error dict in 'data' if present
                            with open(file_path, "w", encoding='utf-8') as f:
                                if hasattr(chart_data, 'model_dump_json'): f.write(chart_data.model_dump_json(indent=2, exclude_none=True))
                                else: json.dump({'metadata': vars(chart_data.metadata), 'data': chart_data.data}, f, indent=2, ensure_ascii=False, default=str)
                            log_content_type = "error data" if isinstance(chart_data.data, dict) and chart_data.data.get("error") else "chart data"
                            logger.debug(f"Saved {file_path.name} (containing {log_content_type}) ({prep_time:.3f}s)")
                        except Exception as e_write: logger.error(f"Failed to write {file_path}: {e_write}")
                    elif chart_data is not None: logger.warning(f"Preparer {fig_name} (lang={lang}) returned non-ChartData type: {type(chart_data)}. File NOT saved. ({prep_time:.3f}s)")
                    else: logger.warning(f"Preparer {fig_name} (lang={lang}) returned None. File NOT saved. ({prep_time:.3f}s)")
                except Exception as e: logger.error(f"Error preparing/saving {fig_name} (lang={lang}): {e}", exc_info=True)
        try:
            logger.info(f"Saving intermediate analysis DataFrames for client {client_id}...")
            for key, df_or_series in analysis_data.items():
                 if isinstance(df_or_series, pd.DataFrame) and not df_or_series.empty and key != 'pie_data': df_or_series.to_parquet(client_dir / f"analysis_{key}.parquet", index=False)
                 elif isinstance(df_or_series, pd.Series) and not df_or_series.empty: df_or_series.to_frame().to_parquet(client_dir / f"analysis_{key}.parquet")
            if daily_forecast is not None and not daily_forecast.empty: daily_forecast.to_parquet(client_dir/"forecast_daily.parquet", index=False)
            if monthly_forecast is not None and not monthly_forecast.empty: monthly_forecast.to_parquet(client_dir/"forecast_monthly.parquet", index=False)
            logger.info("Intermediate analysis/forecast data saved as Parquet.")
        except Exception as e_parquet: logger.warning(f"Failed to save some analysis/forecast dataframes as Parquet: {e_parquet}")
        logger.info(f"Finished saving results for Client ID: {client_id}")

    def run_pipeline(self, client_id: int):
        """تشغيل خطوات المعالجة بالترتيب لعميل محدد."""
        try:
            logger.info(f"--- Starting data pipeline run for Client ID: {client_id} ---")
            start_run_time = time.time()
            self._load_data_from_db(client_id); self._validate_input_data(); self._preprocess_data(); self._analyze_data(); analysis_results = self.analytics
            daily_forecast_results = pd.DataFrame()
            if FORECASTING_AVAILABLE: self._run_forecasting(); daily_forecast_results = self.forecast_data
            else: logger.warning("Daily forecasting unavailable.")
            monthly_forecast_results = pd.DataFrame()
            if MONTHLY_FORECASTING_AVAILABLE: self._run_monthly_forecasting(); monthly_forecast_results = self.monthly_avg_invoice_forecast
            else: logger.warning("Monthly forecasting unavailable.")
            self._save_results_to_disk(client_id, analysis_results, daily_forecast_results, monthly_forecast_results)
            end_run_time = time.time()
            logger.info(f"--- Completed data pipeline run for Client ID: {client_id} in {end_run_time - start_run_time:.2f}s ---")
        except ConnectionError as ce: logger.error(f"DATABASE ERROR client {client_id}: {ce}", exc_info=True); raise
        except ValueError as ve: logger.error(f"DATA ERROR client {client_id}: {ve}", exc_info=True); raise
        except RuntimeError as rte: logger.error(f"RUNTIME ERROR client {client_id}: {rte}", exc_info=True); raise
        except Exception as e: logger.error(f"UNEXPECTED ERROR client {client_id}: {e}", exc_info=True); raise RuntimeError(f"Unexpected error: {e}")

    def get_current_stock_table(self):
        """إرجاع جدول المخزون الحالي (قد لا تكون ضرورية للـ API)."""
        logger.debug("Getting current stock table...")
        products = self.processed_data.get('products', pd.DataFrame())
        if products.empty: logger.warning("Products data is empty for stock table."); return pd.DataFrame()
        try:
            prod_id_col, prod_name_col, _, _, prod_qty_col, _ = config.required_columns['products']
            if not all(c in products.columns for c in [prod_id_col, prod_name_col, prod_qty_col]): logger.error("Missing required columns for stock table."); return pd.DataFrame()
            stock_table = products[[prod_id_col, prod_name_col, prod_qty_col]].copy()
            stock_table.rename(columns={prod_id_col: 'product_id', prod_name_col: 'name', prod_qty_col: 'current_stock'}, inplace=True)
            stock_table['product_id'] = stock_table['product_id'].astype(str)
            stock_table['current_stock'] = pd.to_numeric(stock_table['current_stock'], errors='coerce').fillna(0)
            return stock_table[['product_id', 'name', 'current_stock']]
        except KeyError as e: logger.error(f"Missing column key for stock table config: {e}"); return pd.DataFrame()
        except Exception as e: logger.error(f"Error getting stock table: {e}", exc_info=True); return pd.DataFrame()

# --- End of core/datapipeline.py ---