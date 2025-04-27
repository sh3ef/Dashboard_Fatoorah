# api/endpoints.py - V2.0 - Add 90-day filtering for fig11 actual data
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import JSONResponse
# from core.datapipeline import DataPipeline # Pipeline instance accessed via app.state
from core.config import config
import pandas as pd 
import numpy as np
import json
from typing import Dict, Optional, List, Union, Any
from datetime import datetime, timedelta 
from pydantic import BaseModel, Field 
from enum import Enum
import os
import traceback
import logging
import time
from pathlib import Path

# --- **** Import Localization Info **** ---
from core.localization import SUPPORTED_LANGUAGES 

# --- Import Pydantic Models ---
try:
    from .models import ChartData, ChartMetadata, ChartType, AxisInfo, SeriesInfo
    MODELS_IMPORTED_EP = True
except ImportError:
    logging.getLogger(__name__).error("Failed to import Pydantic models from api.models. Using dummy models.")
    # Define dummy models to avoid runtime errors if import fails
    class BaseModel: pass
    class ChartTypeEnum(str, Enum): BAR="bar"; LINE="line"; PIE="pie"; SCATTER="scatter"; TABLE="table"; COMBO="combo"
    class ChartType(BaseModel): pass
    ChartType = ChartTypeEnum # Use the enum directly
    class AxisInfo(BaseModel): name: str = ""; type: Optional[str] = None; title: Optional[str] = None
    class SeriesInfo(BaseModel): name: str = ""; color: Optional[str] = None; type: Optional[ChartType] = None
    class ChartMetadata(BaseModel): timestamp: str = ""; description: str = ""; frequency: str = ""; title: str = ""; chart_type: ChartType = ChartType.TABLE; x_axis: AxisInfo = AxisInfo(); y_axis: AxisInfo = AxisInfo(); series: List[SeriesInfo] = []
    class ChartData(BaseModel): metadata: ChartMetadata = ChartMetadata(); data: Union[List[Dict[str, Any]], Dict] = {}
    MODELS_IMPORTED_EP = False


# --- Setup Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')


# --- Define Results Directory ---
RESULTS_BASE_DIR = Path("data/client_results")


# --- Create API Router ---
router = APIRouter()


# --- Combined Response Models (Remain the same) ---
class MonthlyResponse(BaseModel):
    fig2: Optional[ChartData] = None
    fig3: Optional[ChartData] = None
    fig5: Optional[ChartData] = None
    fig6: Optional[ChartData] = None
    fig7: Optional[ChartData] = None
    fig10: Optional[ChartData] = None
    fig12: Optional[ChartData] = None
    fig13: Optional[ChartData] = None
    fig15: Optional[ChartData] = None
    fig16: Optional[ChartData] = None


class QuarterlyResponse(BaseModel):
    fig14: Optional[ChartData] = None


class WeeklyResponse(BaseModel):
    fig1: Optional[ChartData] = None
    fig4: Optional[ChartData] = None
    fig9: Optional[ChartData] = None


class DailyResponse(BaseModel):
    fig11: Optional[ChartData] = None
    fig17: Optional[ChartData] = None


# --- Processing Trigger Endpoint (Remains the same) ---
@router.post("/trigger-processing/{client_id}", summary="Trigger data processing and saving for a specific client", status_code=200)
async def trigger_client_processing(client_id: int, request: Request):
    """
    Triggers the data loading, processing, and result saving pipeline (all languages)
    for the specified client ID.
    """
    pipeline = getattr(request.app.state, 'pipeline', None)
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline service unavailable (failed initialization or missing).")
    if pipeline.engine is None: raise HTTPException(status_code=503, detail="Database connection for pipeline is not available.")
    lock_key = f"client_lock_{client_id}"
    if not hasattr(request.app.state, 'processing_locks'): request.app.state.processing_locks = {}
    lock = request.app.state.processing_locks.get(lock_key)
    if lock is None: import asyncio; lock = asyncio.Lock(); request.app.state.processing_locks[lock_key] = lock
    if lock.locked(): raise HTTPException(status_code=429, detail=f"Processing already in progress for client {client_id}.")
    async with lock:
        try:
            logger.info(f"--- Triggering processing & saving (all languages) for Client ID: {client_id} ---")
            start_time = time.time()
            pipeline.run_pipeline(client_id=client_id) # run_pipeline is synchronous
            processing_time = time.time() - start_time
            logger.info(f"--- Processing & saving (all languages) completed for Client {client_id} in {processing_time:.2f} seconds ---")
            return JSONResponse(content={"message": f"Data processing and saving triggered successfully for client {client_id} (all languages)."}, status_code=200)
        except ConnectionError as ce: logger.error(f"DB connection failed processing client {client_id}: {ce}", exc_info=True); raise HTTPException(status_code=503, detail=f"Database connection failed: {ce}")
        except ValueError as ve: logger.error(f"Data error processing client {client_id}: {ve}", exc_info=True); raise HTTPException(status_code=400, detail=f"Data error for client {client_id}: {ve}")
        except RuntimeError as rte: logger.error(f"Runtime error pipeline client {client_id}: {rte}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error during processing: {rte}")
        except Exception as e: logger.error(f"General error processing trigger client {client_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- **** Modified Helper Function to Load Localized Chart Data **** ---
async def load_chart_data(client_id: int, figure_name: str, lang: str) -> Optional[ChartData]:
    """
    Loads ChartData from a language-specific JSON file for a specific client and figure.
    Falls back to 'en' if the requested language file is not found.
    """
    if not MODELS_IMPORTED_EP: # Check if Pydantic models were imported
        logger.error("Cannot load chart data because Pydantic models were not imported correctly.")
        return None
    normalized_lang = lang.lower()
    fallback_lang = 'en'
    if normalized_lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language '{lang}' requested for client {client_id}, figure {figure_name}. Falling back to '{fallback_lang}'.")
        normalized_lang = fallback_lang
    file_path = RESULTS_BASE_DIR / str(client_id) / f"{figure_name}_{normalized_lang}.json"
    if not file_path.is_file():
        logger.warning(f"Data file not found for client {client_id}, figure {figure_name}, lang '{normalized_lang}' at {file_path}")
        if normalized_lang != fallback_lang:
             logger.info(f"Attempting to load fallback language '{fallback_lang}' instead.")
             file_path_fallback = RESULTS_BASE_DIR / str(client_id) / f"{figure_name}_{fallback_lang}.json"
             if file_path_fallback.is_file(): file_path = file_path_fallback; normalized_lang = fallback_lang; logger.info(f"Found fallback data file: {file_path}")
             else: logger.warning(f"Fallback data file also not found for '{fallback_lang}' at {file_path_fallback}"); return None
        else: return None
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            chart_data_dict = json.load(f)
            # --- Validate using Pydantic AFTER loading ---
            validated_data = ChartData.model_validate(chart_data_dict)
            logger.debug(f"Successfully loaded and validated {figure_name} for lang {normalized_lang}")
            return validated_data # Return the validated Pydantic object
    except json.JSONDecodeError as json_err: logger.error(f"JSON Decode Error loading {figure_name} (lang {normalized_lang}) for client {client_id}: {json_err} in file {file_path}"); return None
    except Exception as e: logger.error(f"Failed to load/parse/validate {figure_name} (lang {normalized_lang}) data for client {client_id} from {file_path}: {e}", exc_info=True); return None


# --- **** Combined GET Endpoints - Modified for Localization Header **** ---

@router.get("/daily/all/{client_id}", response_model=DailyResponse, summary="Get all daily reports for a specific client (Localized)")
async def get_all_daily_data_client(
    client_id: int,
    x_language: str = Header( default='en', alias='X-Language', description=f"Language code ({', '.join(SUPPORTED_LANGUAGES)})" )
):
    """
    Retrieves daily reports (fig11, fig17) for the client, localized based on the
    X-Language header. Filters fig11 actual data to the last 90 days.
    """
    logger.info(f"Fetching all daily data for client {client_id} (Requested lang={x_language})...")
    fig11_data_full = await load_chart_data(client_id, "fig11", x_language)
    fig17_data = await load_chart_data(client_id, "fig17", x_language)

    fig11_data_filtered = None
    if fig11_data_full and isinstance(fig11_data_full.data, list):
        logger.info(f"[{client_id}/fig11] Loaded {len(fig11_data_full.data)} total data points.")
        # --- *** START: Filter fig11 data here *** ---
        try:
            all_data = fig11_data_full.data
            # Find the latest date among actual data points
            latest_actual_date_str = None
            actual_dates = []
            for point in all_data:
                if point.get("type") == "actual" and isinstance(point.get("date"), str):
                    try:
                        actual_dates.append(datetime.strptime(point["date"], "%Y-%m-%d").date())
                    except ValueError:
                        logger.warning(f"[{client_id}/fig11] Could not parse date string: {point.get('date')}")
                        continue # Skip points with invalid dates

            if actual_dates:
                latest_actual_date = max(actual_dates)
                cutoff_date = latest_actual_date - timedelta(days=89)
                logger.info(f"[{client_id}/fig11] Filtering actual data from {cutoff_date} onwards.")

                filtered_list = []
                for point in all_data:
                    point_type = point.get("type")
                    point_date_str = point.get("date")

                    if point_type == "forecast":
                        filtered_list.append(point) # Always include forecast points
                    elif point_type == "actual" and isinstance(point_date_str, str):
                        try:
                            point_date = datetime.strptime(point_date_str, "%Y-%m-%d").date()
                            if point_date >= cutoff_date:
                                filtered_list.append(point) # Include actual points within the last 90 days
                        except ValueError:
                            continue # Skip actual points with invalid dates
                    else:
                         logger.warning(f"[{client_id}/fig11] Skipping point with missing/invalid type or date: {point}")

                # Create a *new* ChartData object with the filtered list
                # (or modify in place if Pydantic model allows direct list assignment)
                fig11_data_filtered = ChartData(
                    metadata=fig11_data_full.metadata, # Keep original metadata
                    data=sorted(filtered_list, key=lambda x: x.get("date", "")) # Sort the final list
                )
                logger.info(f"[{client_id}/fig11] Filtered data points: {len(fig11_data_filtered.data)}")
            else:
                logger.warning(f"[{client_id}/fig11] No valid actual dates found, cannot filter. Returning all data.")
                fig11_data_filtered = fig11_data_full # Return full data if no actual dates

        except Exception as e_filter:
            logger.error(f"[{client_id}/fig11] Error during endpoint filtering: {e_filter}. Returning full data.", exc_info=True)
            fig11_data_filtered = fig11_data_full # Fallback to full data on error
        # --- *** END: Filter fig11 data here *** ---
    elif fig11_data_full:
         logger.warning(f"[{client_id}/fig11] Loaded data is not a list, skipping filtering.")
         fig11_data_filtered = fig11_data_full # Return as is if not a list


    if fig11_data_filtered is None and fig17_data is None:
        raise HTTPException(status_code=404, detail=f"No daily data found for client {client_id} (lang={x_language}). Trigger processing first or check logs.")

    return DailyResponse(fig11=fig11_data_filtered, fig17=fig17_data)


@router.get("/weekly/all/{client_id}", response_model=WeeklyResponse, summary="Get all weekly reports for a specific client (Localized)")
async def get_all_weekly_data_client(
    client_id: int,
    x_language: str = Header('en', alias='X-Language', description=f"Language code ({', '.join(SUPPORTED_LANGUAGES)})")
):
    """
    Retrieves weekly reports (fig1, fig4, fig9) for the client, localized based on the
    X-Language header. Falls back to 'en'.
    """
    logger.info(f"Fetching all weekly data for client {client_id} (Requested lang={x_language})...")
    fig1_data = await load_chart_data(client_id, "fig1", x_language)
    fig4_data = await load_chart_data(client_id, "fig4", x_language)
    fig9_data = await load_chart_data(client_id, "fig9", x_language)

    if fig1_data is None and fig4_data is None and fig9_data is None:
        raise HTTPException(status_code=404, detail=f"No weekly data found for client {client_id} (lang={x_language}).")
    return WeeklyResponse(fig1=fig1_data, fig4=fig4_data, fig9=fig9_data)


@router.get("/monthly/all/{client_id}", response_model=MonthlyResponse, summary="Get all monthly reports for a specific client (Localized)")
async def get_all_monthly_data_client(
    client_id: int,
    x_language: str = Header('en', alias='X-Language', description=f"Language code ({', '.join(SUPPORTED_LANGUAGES)})")
):
    """
    Retrieves all monthly reports for the client, localized based on the
    X-Language header. Falls back to 'en'.
    """
    logger.info(f"Fetching all monthly data for client {client_id} (Requested lang={x_language})...")
    # List of monthly figures
    monthly_figs = [2, 3, 5, 6, 7, 10, 12, 13, 15, 16]
    fig_data: Dict[str, Optional[ChartData]] = {}
    found_any = False

    for n in monthly_figs:
        fig_name = f"fig{n}"
        # Pass the language to the loading function
        data = await load_chart_data(client_id, fig_name, x_language)
        fig_data[fig_name] = data
        if data is not None:
            found_any = True # Mark if at least one figure was found

    if not found_any:
         raise HTTPException(status_code=404, detail=f"No monthly data found for client {client_id} (lang={x_language}).")

    # Pass the dictionary to the Pydantic model for validation and response generation
    # Pydantic handles None values correctly if the field is Optional
    return MonthlyResponse(**fig_data)


@router.get("/quarterly/all/{client_id}", response_model=QuarterlyResponse, summary="Get all quarterly reports for a specific client (Localized)")
async def get_all_quarterly_data_client(
    client_id: int,
    x_language: str = Header('en', alias='X-Language', description=f"Language code ({', '.join(SUPPORTED_LANGUAGES)})")
):
    """
    Retrieves quarterly reports (fig14) for the client, localized based on the
    X-Language header. Falls back to 'en'.
    """
    logger.info(f"Fetching all quarterly data for client {client_id} (Requested lang={x_language})...")
    fig14_data = await load_chart_data(client_id, "fig14", x_language)

    if fig14_data is None:
        raise HTTPException(status_code=404, detail=f"No quarterly data (fig14) found for client {client_id} (lang={x_language}).")
    return QuarterlyResponse(fig14=fig14_data)