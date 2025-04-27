# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional, Any
from enum import Enum
import os
from pathlib import Path
import logging # Added logging for potential issues
# core/config.py
# ... other imports ...

print("=" * 60)
print("DEBUG: Checking Project Root Directory Contents")
try:
    try:
        project_root_for_listing = Path(__file__).resolve().parent.parent
    except NameError:
        project_root_for_listing = Path('.').resolve()

    print(f"DEBUG: Listing contents of: {project_root_for_listing}")

    if project_root_for_listing.is_dir():
        try:
            contents = list(project_root_for_listing.iterdir())
            if not contents:
                print("DEBUG: Directory appears empty or listing failed.")
            else:
                print("DEBUG: Directory Contents:")
                for item in contents:
                    item_type = "Dir" if item.is_dir() else "File" if item.is_file() else "Other"
                    print(f"DEBUG:  - {item.name:<40} [{item_type}]")
                    if item.name == ".env":
                        print(f"DEBUG:   ^^^ Found item named '.env'. Is file? {item.is_file()}")
        except PermissionError:
            print("DEBUG: ERROR - Permission denied to list directory contents.")
        except Exception as e_list:
            print(f"DEBUG: ERROR - Could not list directory contents: {e_list}")
    else:
        print(f"DEBUG: ERROR - Calculated project root '{project_root_for_listing}' is not a directory.")

except Exception as e_path_calc:
    print(f"DEBUG: ERROR - Could not calculate project root for listing: {e_path_calc}")
print("=" * 60)

# --- Setup Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Determine Project Root and .env File Path ---
try:
    # Assumes config.py is in the 'core' directory, so parent.parent is the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environments)
    PROJECT_ROOT = Path('.').resolve()
    logger.warning(f"__file__ not defined, assuming project root is current working directory: {PROJECT_ROOT}")

ENV_FILE_PATH = PROJECT_ROOT / ".env"

# --- Debugging: Check the determined path and file existence ---
print("-" * 50)
print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DEBUG: Calculated ENV_FILE_PATH: {ENV_FILE_PATH}")
env_exists = ENV_FILE_PATH.is_file() # Check if it's specifically a file
print(f"DEBUG: Does .env file exist at this path? {env_exists}")
if not env_exists:
    print("DEBUG: WARNING - .env file not found at the calculated path. Defaults or environment variables will be used.")
print("-" * 50)
# --- End Debugging ---


# --- Settings Models ---

class Frequency(str, Enum):
    """Enum for data processing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class VisualizationConfig(BaseSettings):
    """Configuration for visualization refresh schedules."""
    refresh_day: Optional[int] = None
    refresh_hour: Optional[int] = None
    visualizations: List[str] = []

class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    # Define defaults which are used if not found in .env or environment variables
    DB_USER: str = "default_user"
    DB_PASSWORD: str = "default_password"
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "default_db"

    def get_db_url(self) -> str:
        """Constructs the database connection URL."""
        return f"mysql+mysqlconnector://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        # Explicitly point to the calculated .env file path if it exists
        env_file=str(ENV_FILE_PATH) if env_exists else None,
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra environment variables
    )

class AuthSettings(BaseSettings):
    """Authentication settings."""
    # Secrets should ideally come only from .env or environment variables
    SECRET_KEY: str = "!!!_CHANGE_IN_ENV_OR_DOTENV_!!!" # Provide a default only for structure clarity
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        # Load from the same .env file if auth settings are there
        env_file=str(ENV_FILE_PATH) if env_exists else None,
        env_file_encoding='utf-8',
        env_prefix='AUTH_', # e.g., expect AUTH_SECRET_KEY in .env/environment
        extra='ignore'
    )

class DataPipelineConfig(BaseSettings):
    """Main application configuration class."""
    db: DatabaseSettings = DatabaseSettings()
    auth: AuthSettings = AuthSettings()

    # Visualization update configurations (might be better as constants if not changed often)
    update_config: Dict[Frequency, VisualizationConfig] = {
        Frequency.DAILY: VisualizationConfig(visualizations=['fig11', 'fig17']),
        Frequency.WEEKLY: VisualizationConfig(visualizations=['fig1', 'fig4', 'fig9']),
        Frequency.MONTHLY: VisualizationConfig(refresh_day=1, visualizations=['fig2', 'fig3', 'fig5', 'fig6', 'fig7', 'fig10', 'fig12', 'fig13', 'fig15', 'fig16']),
        Frequency.QUARTERLY: VisualizationConfig(refresh_day=1, visualizations=['fig14'])
    }

    # Required columns for data validation (using internal/database names)
    required_columns: Dict[str, List[str]] = {
        'products': ['id', 'name', 'buyPrice', 'salePrice', 'quantity', 'client_id'],
        'sale_invoices': ['id', 'created_at', 'totalPrice', 'client_id'],
        'sale_invoices_details': ['id', 'product_id', 'invoice_id', 'quantity', 'totalPrice', 'buyPrice', 'created_at'],
        'invoice_deferreds': ['invoice_type', 'status', 'amount', 'paid_amount', 'user_id', 'client_id']
    }

    # Date parsing settings (internal column names)
    date_settings: Dict[str, Any] = {
        'date_columns': {
            'sale_invoices': ['created_at'],
            'sale_invoices_details': ['created_at']
        }
    }

    # Parameters for data analysis steps
    analysis_params: Dict[str, Any] = {
        'efficiency_bins': [0, 0.8, 1.2, float('inf')],
        'efficiency_labels': ['Undersupplied', 'Balanced', 'Oversupplied'], # Internal labels
        'stagnant_periods': {
            'bins': [90, 180, 365, float('inf')], # Days thresholds
            'labels': ['3-6 months', '6-12 months', '>1 year'], # Internal labels
        },
        'restock_threshold': 10,
        'pareto_threshold': 80,
        'forecast_horizon_daily': 14,
        'seasonal_period_daily': 7,
        'forecast_horizon_monthly': 6,
        'seasonal_period_monthly': 12,
        'min_monthly_obs': 12,
    }

# --- Create the global config instance ---
# This instance will be loaded based on the model_config definitions above
config = DataPipelineConfig()

# --- Final Debug Print (Check loaded values) ---
# This will show the actual values being used by the application
print("-" * 50)
print("--- FINAL CONFIG CHECK (Loaded Values) ---")
print(f"Loaded DB User: '{config.db.DB_USER}'")
# Avoid printing full password in logs
password_status = "***" if config.db.DB_PASSWORD and config.db.DB_PASSWORD != "default_password" else "(default or empty/not loaded)"
print(f"Loaded DB Password Status: {password_status}")
print(f"Loaded DB Host: '{config.db.DB_HOST}'")
print(f"Loaded DB Port: {config.db.DB_PORT}")
print(f"Loaded DB Name: '{config.db.DB_NAME}'")
print(f"Generated DB URL: '{config.db.get_db_url()}'")
secret_key_status = "***" if config.auth.SECRET_KEY and config.auth.SECRET_KEY != "!!!_CHANGE_IN_ENV_OR_DOTENV_!!!" else "(default or empty/not loaded)"
print(f"Loaded Auth Secret Key Status: {secret_key_status}")
print("--- END FINAL CONFIG CHECK ---")
print("-" * 50)

