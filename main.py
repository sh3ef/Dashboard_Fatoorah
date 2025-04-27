# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Uncomment if you need static files
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path # Added Path for static files check
import os
import uvicorn
import logging

# --- Import application components ---
from api.endpoints import router as api_router
from core.datapipeline import DataPipeline # Needed for app.state initialization and startup check
from core.localization import SUPPORTED_LANGUAGES # Import supported languages

# --- Setup Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure basic logging if not already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("Application lifespan startup...")
    # Initialize processing locks dictionary at startup
    app.state.processing_locks = {}
    logger.info("Processing locks dictionary initialized.")

    # --- Database Connection Check during startup ---
    # The actual DataPipeline instance is created in create_app() and stored in app.state.
    # We check here if it was created successfully.
    pipeline_instance = getattr(app.state, 'pipeline', None)
    if pipeline_instance is None or pipeline_instance.engine is None:
        # This indicates a failure during create_app()
        logger.critical("DataPipeline instance or its DB engine is invalid during lifespan startup. Startup failed earlier.")
        # Technically, the app might have already failed to start due to the check in create_app().
        # This is an additional safety check within the lifespan context.
    else:
        logger.info("DataPipeline instance appears valid in app state during lifespan startup.")

    yield # Application runs here

    # --- Shutdown Logic ---
    logger.info("Application lifespan shutdown...")
    # Cleanup locks
    if hasattr(app.state, 'processing_locks'):
         app.state.processing_locks = {}
         logger.info("Processing locks cleared.")
    # Cleanup pipeline instance if it was stored in state
    pipeline_to_dispose = getattr(app.state, 'pipeline', None)
    if pipeline_to_dispose and hasattr(pipeline_to_dispose, 'engine') and pipeline_to_dispose.engine:
        logger.info("Disposing DataPipeline engine from app state...")
        try:
            pipeline_to_dispose.engine.dispose()
            logger.info("DataPipeline engine disposed.")
        except Exception as e_dispose:
            logger.error(f"Error disposing pipeline engine from state: {e_dispose}")
    app.state.pipeline = None # Clear the state reference
    logger.info("Lifespan resources cleaned up.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Inventory Analytics API (Multi-Client & Localized)",
        description="API لتحليل بيانات المخزون والمبيعات ودعم متعدد العملاء واللغات (AR/EN).",
        version="1.2.1", # Incremented version slightly
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan # Assign the lifespan manager
    )

    # CORS Middleware Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # TODO: Restrict in production
        allow_credentials=True,
        allow_methods=["GET", "POST"], # Restrict methods if possible
        allow_headers=["*"], # TODO: Restrict headers in production (e.g., Content-Type, X-Language, Authorization)
    )

    # Include API Routers
    app.include_router(api_router, prefix="/api/v1")

    # --- *** Pre-initialize Pipeline Instance with Startup Failure on DB Error *** ---
    logger.info("Attempting to create and validate DataPipeline instance...")
    try:
        # Create the pipeline instance
        pipeline_instance = DataPipeline()

        # **CRITICAL CHECK:** Verify DB connection was successful during __init__
        if pipeline_instance.engine is None:
             # If engine is None, it means __init__ failed to connect to the DB
             logger.critical("DataPipeline engine initialization failed! Database connection error during __init__.")
             # Raise a RuntimeError to prevent FastAPI/Uvicorn from starting successfully
             raise RuntimeError("Database connection failed, cannot start application.")
        else:
            # If engine exists, store the valid instance in app state
            app.state.pipeline = pipeline_instance
            logger.info("DataPipeline instance created successfully and stored in app state.")

    except Exception as pipe_create_err:
        # Catch any other exception during DataPipeline creation or the explicit RuntimeError above
        logger.critical(f"Failed to create/validate DataPipeline instance during app creation: {pipe_create_err}", exc_info=True)
        # Ensure app state is clean
        app.state.pipeline = None
        # Re-raise or raise a new RuntimeError to stop startup
        raise RuntimeError(f"Failed to initialize pipeline, cannot start application: {pipe_create_err}")
    # --- End of Pipeline Initialization ---

    # Optional: Mount static files directory if needed
    static_dir = "static"
    static_path = Path(static_dir)
    if static_path.exists() and static_path.is_dir():
        try:
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
            logger.info(f"Mounted static files from directory: {static_dir}")
        except Exception as e_static:
             logger.error(f"Failed to mount static directory '{static_dir}': {e_static}")
    else:
        logger.warning(f"Static files directory '{static_dir}' not found or is not a directory. Skipping mount.")

    return app

# Create the FastAPI app instance
# This might raise a RuntimeError now if DB connection fails during create_app()
try:
    app = create_app()
except RuntimeError as app_creation_error:
    logger.critical(f"--- APPLICATION FAILED TO START: {app_creation_error} ---", exc_info=False)
    # Exit the script or handle the failure appropriately
    import sys
    sys.exit(1) # Exit with a non-zero code to indicate failure


# --- Root Endpoint ---
@app.get("/", tags=["Root"], response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Provides basic information, supported languages, and available endpoint groups."""
    return {
        "message": "مرحباً بكم في نظام تحليل المخزون والمبيعات (إصدار متعدد العملاء واللغات)",
        "supported_languages": SUPPORTED_LANGUAGES,
        "language_header_info": {
            "name": "X-Language",
            "description": "Send this header with 'ar' or 'en' to get localized responses for GET requests.",
            "default": "en"
        },
        "endpoints": {
            "trigger_processing": {
                "path": "/api/v1/trigger-processing/{client_id}",
                "method": "POST",
                "description": "تشغيل معالجة البيانات وحفظ النتائج (لكل اللغات المدعومة) لعميل محدد.",
                "example_client_id": 185 # Example ID
            },
            "daily_visualizations": {
                "path": "/api/v1/daily/all/{client_id}",
                "method": "GET",
                "description": "الحصول على الرسوم البيانية اليومية المحفوظة لعميل محدد. يدعم Header 'X-Language'."
            },
            "weekly_visualizations": {
                "path": "/api/v1/weekly/all/{client_id}",
                "method": "GET",
                "description": "الحصول على الرسوم البيانية الأسبوعية المحفوظة لعميل محدد. يدعم Header 'X-Language'."
            },
            "monthly_visualizations": {
                "path": "/api/v1/monthly/all/{client_id}",
                "method": "GET",
                "description": "الحصول على جميع الرسوم البيانية الشهرية المحفوظة لعميل محدد. يدعم Header 'X-Language'."
            },
            "quarterly_visualizations": {
                "path": "/api/v1/quarterly/all/{client_id}",
                "method": "GET",
                "description": "الحصول على جميع الرسوم البيانية الربع سنوية المحفوظة لعميل محدد. يدعم Header 'X-Language'."
            },
            "documentation": {
                "path": ["/docs", "/redoc"],
                "method": "GET",
                "description": "الوصول إلى توثيق API التفاعلي (Swagger UI / ReDoc)."
            }
        }
    }

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # Use environment variables for configuration where possible
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    # Default reload to False for production, True for development
    reload = os.getenv("RELOAD", "true").lower() == "true" if os.getenv("DEV_MODE", "true").lower() == "true" else False
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    # Check if the app instance was created successfully before running uvicorn
    if 'app' not in globals():
        logger.critical("FastAPI app instance 'app' was not created due to startup errors. Exiting.")
    else:
        logger.info(f"Starting Uvicorn server on {host}:{port}")
        logger.info(f"Reload: {reload}, Workers: {workers}, Log Level: {log_level}")
        if workers > 1 and reload:
            logger.warning("Running with multiple workers and reload enabled is not recommended. Setting reload to False.")
            reload = False
        if workers > 1:
            logger.info("Running with multiple workers. Ensure locking mechanism is appropriate.")

        uvicorn.run(
            "main:app", # Point to the app instance in this file
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
            access_log=True,
            # proxy_headers=True, # Uncomment if behind reverse proxy
            # forwarded_allow_ips='*', # Adjust as needed if using proxy_headers
        )
