# dashboard_app.py - V5.3 - Fix Enum Comparison for Chart Type & Removed Trendline from Fig7

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import os
import logging
import json
import numpy as np
from datetime import datetime, timedelta
import traceback
from enum import Enum # Import Enum

# --- Import Core Components ---
try:
    from core.config import config
    from core.datapipeline import DataPipeline, FORECASTING_AVAILABLE, MONTHLY_FORECASTING_AVAILABLE
    from core.localization import SUPPORTED_LANGUAGES, get_translation as get_translation_core
    try:
        # Import ChartType specifically if it's an Enum
        from api.models import ChartData, ChartType
    except ImportError:
        logger.warning("Could not import ChartData or ChartType from api.models. Using dummy classes.")
        class ChartData:
            def __init__(self, metadata=None, data=None): self.metadata = metadata; self.data = data
            def model_dump(self, **kwargs): return {'metadata': self.metadata, 'data': self.data}
        # Dummy Enum for ChartType if import fails
        class ChartType(str, Enum):
            BAR = "bar"; LINE = "line"; PIE = "pie"; SCATTER = "scatter"; TABLE = "table"; COMBO = "combo"

    CORE_IMPORTED = True
except ImportError as e:
    st.error(f"Fatal Error: Could not import core modules. Details: {e}")
    CORE_IMPORTED = False
    class DataPipeline: pass
    class ChartData: pass
    class ChartType(str, Enum): # Define dummy here too
        BAR = "bar"; LINE = "line"; PIE = "pie"; SCATTER = "scatter"; TABLE = "table"; COMBO = "combo"
    config = None; SUPPORTED_LANGUAGES = ['en']; FORECASTING_AVAILABLE = False; MONTHLY_FORECASTING_AVAILABLE = False
    def get_translation_core(lang, key, default=""): return default
    st.stop()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Basic Settings ---
st.set_page_config(layout="wide", page_title="Client Dashboard (Direct DB) V5.3", page_icon="📊") # Version Bump

# --- Initialize DataPipeline ---
@st.cache_resource
def initialize_pipeline() -> Optional[DataPipeline]:
    logger.info("Attempting to initialize DataPipeline...")
    try:
        pipeline = DataPipeline()
        if pipeline.engine is None: st.error("Failed to connect to the database during pipeline initialization."); logger.critical("DataPipeline engine is None after initialization."); return None
        logger.info("DataPipeline initialized successfully with DB engine."); return pipeline
    except Exception as e: st.error(f"Critical error initializing DataPipeline: {e}"); logger.critical(f"Failed to create DataPipeline instance: {e}", exc_info=True); return None
pipeline: Optional[DataPipeline] = initialize_pipeline()
if not pipeline: st.warning("Data processing functionality is unavailable due to pipeline initialization failure."); st.stop()

# --- Localization Helpers ---
def get_translation_streamlit(lang: str, key: str, default: str = "") -> str:
    return get_translation_core(lang, key, default)

# --- Data Fetching / Processing Helpers ---
@st.cache_data(ttl=3600)
def get_clients_from_db(_pipeline_ref) -> List[str]:
    client_ids = []
    if not pipeline or not pipeline.engine: st.error("Database connection unavailable."); logger.error("Cannot fetch clients."); return []
    table_to_query = 'sale_invoices'; client_id_col_name_in_config = 'client_id'; actual_client_id_col = 'client_id'
    if config and 'required_columns' in dir(config) and 'sale_invoices' in config.required_columns and client_id_col_name_in_config in config.required_columns['sale_invoices']:
        try: client_id_col_index = config.required_columns['sale_invoices'].index(client_id_col_name_in_config); actual_client_id_col = config.required_columns['sale_invoices'][client_id_col_index]; logger.info(f"Found client ID column '{actual_client_id_col}' from config.")
        except ValueError: logger.warning(f"'{client_id_col_name_in_config}' not found in config list, using default '{actual_client_id_col}'.")
    else: logger.warning(f"Config definition missing or '{client_id_col_name_in_config}' not found, using default '{actual_client_id_col}'.")
    query = f"SELECT DISTINCT `{actual_client_id_col}` FROM {table_to_query} ORDER BY `{actual_client_id_col}` ASC;"
    logger.info(f"Fetching distinct client IDs using query: {query}")
    try:
        with pipeline.engine.connect() as connection: client_df = pd.read_sql(query, connection); actual_col_name_in_df = client_df.columns[0]; client_ids = client_df[actual_col_name_in_df].astype(str).tolist()
        logger.info(f"Found {len(client_ids)} distinct client IDs from DB.")
    except Exception as e: st.error(f"Failed to fetch client list from database: {e}"); logger.error(f"Error executing client list query '{query}': {e}", exc_info=True); return []
    return client_ids

@st.cache_data(ttl=1800, show_spinner="Processing client data (this may take a while)...")
def run_pipeline_and_get_data(_pipeline_ref, client_id: int, lang: str) -> Optional[Dict[str, Optional[ChartData]]]:
    if not pipeline: logger.error(f"Pipeline not initialized for client {client_id}."); return None
    logger.info(f"--- Starting on-demand pipeline run for Client ID: {client_id} (Lang: {lang}) ---"); prepared_data: Dict[str, Optional[ChartData]] = {}; analysis_results = {}; daily_forecast_results = pd.DataFrame(); monthly_forecast_results = pd.DataFrame()
    try:
        pipeline._load_data_from_db(client_id); pipeline._validate_input_data(); pipeline._preprocess_data(); pipeline._analyze_data(); analysis_results = pipeline.analytics
        if FORECASTING_AVAILABLE: pipeline._run_forecasting(); daily_forecast_results = pipeline.forecast_data
        else: logger.warning("Daily forecasting modules unavailable.")
        if MONTHLY_FORECASTING_AVAILABLE: pipeline._run_monthly_forecasting(); monthly_forecast_results = pipeline.monthly_avg_invoice_forecast
        else: logger.warning("Monthly forecasting modules unavailable.")
        logger.info(f"Preparing figures for language: '{lang}'...")
        figure_preparers_config = {
            "fig1": {"func": pipeline._prepare_fig1_data, "inputs": [analysis_results.get('product_flow')]}, "fig2": {"func": pipeline._prepare_fig2_data, "inputs": [analysis_results.get('product_flow')]}, "fig3": {"func": pipeline._prepare_fig3_data, "inputs": [analysis_results.get('product_flow')]},
            "fig4": {"func": pipeline._prepare_fig4_data, "inputs": [analysis_results.get('product_flow')]}, "fig5": {"func": pipeline._prepare_fig5_data, "inputs": [analysis_results.get('product_flow')]}, "fig6": {"func": pipeline._prepare_fig6_data, "inputs": [analysis_results.get('pareto_data')]},
            "fig7": {"func": pipeline._prepare_fig7_data, "inputs": [analysis_results.get('product_flow')]}, "fig9": {"func": pipeline._prepare_fig9_data, "inputs": [analysis_results.get('product_flow')]}, "fig10": {"func": pipeline._prepare_fig10_data, "inputs": [analysis_results.get('stagnant_products')]},
            "fig11": {"func": pipeline._prepare_fig11_data, "inputs": [daily_forecast_results]}, "fig12": {"func": pipeline._prepare_fig12_data, "inputs": [analysis_results.get('monthly_avg_invoice_ts'), monthly_forecast_results]}, "fig13": {"func": pipeline._prepare_fig13_data, "inputs": [analysis_results.get('product_flow')]},
            "fig14": {"func": pipeline._prepare_fig14_data, "inputs": [pipeline.processed_data.get('sale_invoices_details')]}, "fig15": {"func": pipeline._prepare_fig15_data, "inputs": [analysis_results.get('product_flow')]}, "fig16": {"func": pipeline._prepare_fig16_data, "inputs": [analysis_results.get('pie_data')]},
            "fig17": {"func": pipeline._prepare_fig17_data, "inputs": [analysis_results.get('outstanding_amounts')]},
        }
        for fig_key, config_item in figure_preparers_config.items():
            preparer_func = config_item["func"]; inputs = config_item["inputs"]
            try: chart_data_obj: Optional[ChartData] = preparer_func(*inputs, lang=lang)
            except Exception as e_prep_call: logger.error(f"Error calling prepare function for {fig_key} (lang={lang}): {e_prep_call}", exc_info=True); chart_data_obj = None
            if chart_data_obj and isinstance(chart_data_obj, ChartData): prepared_data[fig_key] = chart_data_obj; logger.debug(f"Successfully prepared {fig_key} (lang={lang})")
            else: logger.warning(f"Preparer {fig_key} (lang={lang}) returned None or wrong type. Storing None."); prepared_data[fig_key] = None
        logger.info(f"--- Finished on-demand pipeline run for Client ID: {client_id} (Lang: {lang}) ---")
        return prepared_data
    except Exception as e_pipe_run: st.error(f"Error running data pipeline for client {client_id}: {e_pipe_run}"); logger.error(f"Failed pipeline run for client {client_id}: {e_pipe_run}", exc_info=True); return None

# --- Column Finder ---
def find_column(df: pd.DataFrame, possibilities: List[Optional[str]], chart_title: str = "Chart", exclude: Optional[str] = None) -> Optional[str]:
    # ... (Keep find_column as is) ...
    df_cols_list = df.columns.tolist() if df is not None and not df.empty else ["<Empty or None DF>"]
    logger.debug(f"[{chart_title}] Finding column. Possibilities: {possibilities}, Exclude: {exclude}, DF Cols: {df_cols_list}")
    if df is None or df.empty: logger.warning(f"[{chart_title}] DataFrame is None or empty, cannot find column."); return None
    clean_possibilities = [p for p in possibilities if isinstance(p, str) and p]; df_cols_lower = {col.lower(): col for col in df.columns if isinstance(col, str)} # Filter non-strings
    def check_and_return(col_name: Optional[str]) -> Optional[str]:
        if col_name and col_name != exclude: return col_name; return None
    for name in clean_possibilities:
        col = check_and_return(name);
        if col and col in df.columns: logger.debug(f"[{chart_title}] Found column '{col}' using possibility '{name}' (case-sensitive)."); return col
        actual_col_insensitive = df_cols_lower.get(name.lower()); col = check_and_return(actual_col_insensitive)
        if col: logger.debug(f"[{chart_title}] Found column '{col}' using possibility '{name}' (case-insensitive)."); return col
    common_fallbacks = ['name', 'product', 'Product', 'المنتج', 'date', 'sale_date', 'Date', 'التاريخ', 'value', 'Value', 'القيمة', 'quantity', 'Quantity', 'sales_quantity', 'Quantity Sold', 'الكمية', 'الكمية المباعة', 'stock', 'Stock', 'current_stock', 'Current Stock', 'المخزون', 'المخزون الحالي', 'amount', 'Amount', 'totalPrice', 'المبلغ', 'price', 'Price', 'salePrice', 'Sale Price', 'سعر البيع', 'category', 'Category', 'الفئة', 'user_id', 'supplier_id', 'Supplier (ID)', 'المورد', 'المورد (المعرف)', 'outstanding_amount', 'Outstanding Amount', 'المبلغ المستحق', 'forecast', 'actual', 'lower_ci', 'upper_ci', 'type', 'COGS', 'cogs', 'تكلفة البضاعة', 'margin', 'Margin', 'هامش الربح', 'revenue', 'Revenue', 'الإيرادات', 'sales_amount', 'profit', 'Profit', 'netProfit', 'الربح', 'cumulative_percentage', 'النسبة التراكمية %', 'efficiency', 'efficiency_category', 'days_since_last_sale', 'days_category', 'label', 'id', 'product_id', 'percentage_category']
    all_possibilities = list(dict.fromkeys(clean_possibilities + common_fallbacks)); checked_fallbacks = set(clean_possibilities)
    for name in all_possibilities:
        if name and name not in checked_fallbacks:
             checked_fallbacks.add(name); col = check_and_return(name)
             if col and col in df.columns: logger.debug(f"[{chart_title}] Found column '{col}' using fallback possibility '{name}' (case-sensitive)."); return col
             actual_col_insensitive = df_cols_lower.get(name.lower()); col = check_and_return(actual_col_insensitive)
             if col: logger.debug(f"[{chart_title}] Found column '{col}' using fallback possibility '{name}' (case-insensitive)."); return col
    logger.warning(f"[{chart_title}] Column NOT FOUND from possibilities: {possibilities} or fallbacks (excluding '{exclude}')"); return None

# --- Main Plotting Function (V5.2 - Takes ChartData object) ---
def create_plotly_figure_replica(fig_key: str, chart_data_obj: Optional[ChartData], current_lang: str) -> Optional[go.Figure]:
    logger.info(f"Attempting V5.2 replication for key: {fig_key}")
    fig: Optional[go.Figure] = None; df: Optional[pd.DataFrame] = None
    title: str = f"Figure: {fig_key}"; description: str = ""; metadata: Dict = {}

    try:
        if not chart_data_obj or not isinstance(chart_data_obj, ChartData) or not chart_data_obj.metadata:
            st.warning(f"[{fig_key}] Invalid or missing ChartData object."); logger.warning(f"[{fig_key}] Invalid ChartData format.")
            return go.Figure().update_layout(title=title, annotations=[{'text': "Invalid ChartData Format", 'showarrow': False}])
        metadata_obj = chart_data_obj.metadata; data = chart_data_obj.data
        metadata = metadata_obj.model_dump() if hasattr(metadata_obj, 'model_dump') else vars(metadata_obj)

        # --- *** CORRECTED CHART TYPE HANDLING *** ---
        chart_type_enum = metadata.get('chart_type', ChartType.TABLE) # Get the enum or default
        # Convert enum to string value for comparison and logging
        chart_type_str = str(chart_type_enum.value) if isinstance(chart_type_enum, Enum) else str(chart_type_enum).lower()
        logger.debug(f"[{fig_key}] Raw chart_type: {chart_type_enum}, String value: {chart_type_str}")
        # --- *** END CORRECTION *** ---

        title = metadata.get('title', f"Figure: {fig_key}"); description = metadata.get('description', '')
        x_axis_info = metadata.get('x_axis', {}); y_axis_info = metadata.get('y_axis', {})
        x_title_trans = x_axis_info.get('title', 'X'); x_name_internal = x_axis_info.get('name')
        y_title_trans = y_axis_info.get('title', 'Y'); y2_title_trans = metadata.get('y2_axis', {}).get('title', 'Y2')

        if not data and not (isinstance(data, (list, dict)) and len(data) == 0):
             logger.warning(f"[{fig_key}] No data found within ChartData object."); st.info(f"No data available for: '{title}'")
             return go.Figure().update_layout(annotations=[{'text': get_translation_streamlit(current_lang, 'error.no_data', "No Data Available"), 'showarrow': False}])

        trans_product = get_translation_streamlit(current_lang, 'common.product', 'Product'); trans_quantity = get_translation_streamlit(current_lang, 'common.quantity', 'Quantity'); trans_qty_sold = get_translation_streamlit(current_lang, 'common.quantity_sold', 'Quantity Sold'); trans_stock = get_translation_streamlit(current_lang, 'common.stock', 'Stock'); trans_amount = get_translation_streamlit(current_lang, 'common.amount', 'Amount'); trans_current_stock = get_translation_streamlit(current_lang, 'fig1.series.stock', 'Current Stock'); trans_sales = get_translation_streamlit(current_lang, 'fig1.series.sales', 'Sales'); trans_supplier_id = get_translation_streamlit(current_lang, 'common.supplier_id', 'Supplier ID'); trans_outstanding = get_translation_streamlit(current_lang, 'common.outstanding_amount', 'Outstanding Amount'); trans_sale_price = get_translation_streamlit(current_lang, 'common.sale_price', 'Sale Price'); trans_days = get_translation_streamlit(current_lang, 'common.days_since_last_sale', 'Days Since Last Sale'); trans_period = get_translation_streamlit(current_lang, 'common.stagnancy_period', 'Stagnancy Period'); trans_id_hover = get_translation_streamlit(current_lang, 'common.product_id', 'ID'); trans_last_sale_hover = get_translation_streamlit(current_lang, 'common.last_sale_date', 'Last Sale')

        # ======================================================================
        # --- START: Conditional Plotting Based on Data Type (Corrected Flow)---
        # ======================================================================

        # --- CASE 1: Dictionary Data (PIE CHARTS) ---
        # --- *** Use chart_type_str for comparison *** ---
        if isinstance(data, dict) and chart_type_str == ChartType.PIE.value:
            # --- Subcase 1.1: fig16 (Dual Pie) ---
            if fig_key == 'fig16':
                # ... (fig16 logic remains the same) ...
                logger.debug(f"[{fig_key}] Handling fig16 pies directly."); color_map = data.get('color_mapping', {})
                rev_title_trans = get_translation_streamlit(current_lang, 'fig16.series.revenue', "Revenue"); st.subheader(rev_title_trans)
                rev_data = data.get('revenue_data')
                if rev_data and isinstance(rev_data, list) and rev_data: df_rev = pd.DataFrame(rev_data);
                else: df_rev = pd.DataFrame()
                if not df_rev.empty and 'label' in df_rev.columns and 'value' in df_rev.columns: fig_rev = px.pie(df_rev, names='label', values='value', title=f"{title} - {rev_title_trans}", color='label', color_discrete_map=color_map); fig_rev.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value'); st.plotly_chart(fig_rev, use_container_width=True)
                else: st.info(f"No valid revenue data for '{title}'.")
                prof_title_trans = get_translation_streamlit(current_lang, 'fig16.series.profit', "Profit"); st.subheader(prof_title_trans)
                prof_data = data.get('profit_data')
                if prof_data and isinstance(prof_data, list) and prof_data: df_prof = pd.DataFrame(prof_data);
                else: df_prof = pd.DataFrame()
                if not df_prof.empty and 'label' in df_prof.columns and 'value' in df_prof.columns: fig_prof = px.pie(df_prof, names='label', values='value', title=f"{title} - {prof_title_trans}", color='label', color_discrete_map=color_map); fig_prof.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value'); st.plotly_chart(fig_prof, use_container_width=True)
                else: st.info(f"No valid profit data for '{title}'.")
                return None

            # --- Subcase 1.2: Other Single Pies (like fig2) ---
            else:
                # ... (single pie logic remains the same) ...
                logger.debug(f"[{fig_key}] Handling single pie chart."); labels = data.get('labels'); values = data.get('values'); colors = data.get('colors')
                if labels is not None and values is not None and isinstance(labels, list) and isinstance(values, list) and len(labels) == len(values) and any(v for v in values if v is not None and v > 0):
                    fig = px.pie(names=labels, values=values, title=title) # Create figure here
                    if colors and isinstance(colors, list) and len(colors) == len(labels): fig.update_traces(marker=dict(colors=colors))
                    fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value')
                    if description: fig.update_layout(annotations=[dict(text=description, showarrow=False, xref='paper', yref='paper', x=0.5, y=-0.15, xanchor='center', yanchor='top', align='center')])
                    logger.info(f"Successfully created PIE figure object for key: {fig_key}")
                    return fig
                else:
                    logger.warning(f"[{fig_key}] Invalid data structure or empty values for pie chart. Labels: {labels}, Values: {values}"); st.info(f"No data available to plot for {fig_key} ({title}).")
                    return go.Figure().update_layout(title=title, annotations=[{'text': get_translation_streamlit(current_lang, 'error.no_data', "No Data Available"), 'showarrow': False}])

        # --- CASE 2: List Data ---
        elif isinstance(data, list):
            # ... (DataFrame plotting logic remains the same as V5.2) ...
            if not data: logger.warning(f"[{fig_key}] Data list is empty."); st.info(f"No data available for: '{title}'"); return go.Figure().update_layout(title=title, annotations=[{'text': get_translation_streamlit(current_lang, 'error.no_data', "No Data Available"), 'showarrow': False}])
            df = pd.DataFrame(data)
            if df.empty: logger.warning(f"[{fig_key}] Created DataFrame is empty."); st.info(f"No data available for: '{title}'"); return go.Figure().update_layout(title=title, annotations=[{'text': get_translation_streamlit(current_lang, 'error.no_data', "No Data Available"), 'showarrow': False}])

            fig = go.Figure().update_layout(title=title, xaxis_title=x_title_trans, yaxis_title=y_title_trans, legend_title_text=get_translation_streamlit(current_lang, 'common.series', 'Series'), margin=dict(l=40, r=20, t=60, b=40), hovermode="x unified", uirevision=title)
            x_col_actual = find_column(df, [x_title_trans, x_name_internal], title)
            if not x_col_actual:
                if df.columns.any(): x_col_actual = df.columns[0]; logger.warning(f"[{fig_key}] Defaulting X to '{x_col_actual}'"); fig.update_layout(xaxis_title=x_col_actual)
                else: raise ValueError("DataFrame has no columns.")

            # --- Specific Figure Logic ---
            if fig_key == 'fig1':
                y_col_stock = find_column(df, [trans_current_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_sales, 'sales_quantity'], title)
                if y_col_stock and y_col_sales: fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_current_stock, marker_color='rgb(55, 83, 109)')); fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_sales], name=trans_sales, marker_color='rgb(26, 118, 255)')); fig.update_layout(barmode="group", yaxis_title=trans_quantity)
                else: raise ValueError(f"Missing cols fig1. Need X='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")
            elif fig_key == 'fig3':
                trans_cogs = get_translation_streamlit(current_lang, 'fig3.series.cogs', 'COGS'); trans_revenue = get_translation_streamlit(current_lang, 'fig3.series.revenue', 'Revenue'); trans_margin = get_translation_streamlit(current_lang, 'fig3.series.margin', 'Margin')
                y_col_cogs = find_column(df, [trans_cogs, 'COGS'], title); y_col_revenue = find_column(df, [trans_revenue, 'sales_amount'], title); y_col_margin = find_column(df, [trans_margin, 'margin', 'Margin'], title)
                if y_col_cogs and y_col_revenue and y_col_margin: fig = px.bar(df, x=x_col_actual, y=[y_col_cogs, y_col_revenue, y_col_margin], barmode="group", title=title, labels={'value': trans_amount, 'variable': get_translation_streamlit(current_lang, 'common.type', 'Type'), x_col_actual: x_title_trans}, color_discrete_map={y_col_cogs: "rgb(255, 127, 14)", y_col_revenue: "rgb(26, 118, 255)", y_col_margin: "rgb(44, 160, 44)"}); # Reassign fig from px
                else: raise ValueError(f"Missing cols fig3. Need X='{x_col_actual}', COGS='{y_col_cogs}', Rev='{y_col_revenue}', Margin='{y_col_margin}'")
            elif fig_key == 'fig4':
                y_col_stock = find_column(df, [trans_current_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_sales, 'sales_quantity'], title)
                if y_col_stock and y_col_sales:
                    fig = make_subplots(specs=[[{"secondary_y": True}]]); # Reassign fig
                    fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_current_stock, marker_color='lightblue'), secondary_y=False); fig.add_trace(go.Scatter(x=df[x_col_actual], y=df[y_col_sales], mode='lines+markers', name=trans_sales, line=dict(color='orange')), secondary_y=True)
                    fig.update_layout(title=title, xaxis_title=x_title_trans); fig.update_yaxes(title_text=f"{trans_quantity} ({trans_stock})", secondary_y=False); fig.update_yaxes(title_text=f"{trans_quantity} ({trans_sales})", secondary_y=True)
                else: raise ValueError(f"Missing cols fig4. Need X='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")
            elif fig_key == 'fig5':
                trans_eff_cat = get_translation_streamlit(current_lang, 'common.efficiency_category', 'Efficiency Category'); trans_total_stock = get_translation_streamlit(current_lang, 'fig5.series.stock', 'Total Stock'); trans_total_sales = get_translation_streamlit(current_lang, 'fig5.series.sales', 'Total Sales')
                x_col_actual = find_column(df, [x_title_trans, trans_eff_cat, 'efficiency_category_translated', 'efficiency'], title);
                if not x_col_actual and df.columns.any(): x_col_actual = df.columns[0]; logger.warning(f"[{fig_key}] Defaulting X to '{x_col_actual}'")
                y_col_stock = find_column(df, [trans_total_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_total_sales, 'sales_quantity'], title)
                if x_col_actual and y_col_stock and y_col_sales:
                    fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_total_stock, marker_color='rgb(55, 83, 109)', opacity=0.6)); fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_sales], name=trans_total_sales, marker_color='rgb(26, 118, 255)', opacity=0.6))
                    fig.update_layout(barmode='group', xaxis_title=x_title_trans, yaxis_title=get_translation_streamlit(current_lang, 'common.total_quantity','Total Quantity'))
                else: raise ValueError(f"Missing cols fig5. Cat='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")
            elif fig_key == 'fig6':
                trans_sales_qty = get_translation_streamlit(current_lang, 'fig6.series.sales', 'Sales Quantity'); trans_cumul_perc = get_translation_streamlit(current_lang, 'common.cumulative_percentage', 'Cumulative %')
                y_col_sales = find_column(df, [trans_sales_qty, 'sales_quantity'], title); y_col_cumul = find_column(df, [trans_cumul_perc, 'cumulative_percentage'], title); category_col_actual = find_column(df, ['percentage_category', 'percent_cat'], title)
                if x_col_actual and y_col_sales and y_col_cumul and category_col_actual:
                    if category_col_actual not in df.columns: raise ValueError(f"Required category column '{category_col_actual}' not found for line coloring.")
                    try: df[y_col_cumul] = pd.to_numeric(df[y_col_cumul], errors='coerce'); df_sorted = df.dropna(subset=[y_col_cumul]).sort_values(y_col_cumul, ascending=True).copy()
                    except Exception as e_conv: raise ValueError(f"Cumulative percentage column '{y_col_cumul}' error: {e_conv}")
                    if df_sorted.empty: raise ValueError("No valid data remaining for Pareto plot.")
                    fig = make_subplots(specs=[[{"secondary_y": True}]]); # Reassign fig
                    fig.add_trace(go.Bar( x=df_sorted[x_col_actual], y=df_sorted[y_col_sales], name=trans_sales_qty, marker_color='cornflowerblue' ), secondary_y=False)
                    categories = sorted(df_sorted[category_col_actual].unique()); colors = px.colors.qualitative.Plotly; last_point = None
                    for i, category in enumerate(categories):
                        cat_data = df_sorted[df_sorted[category_col_actual] == category]
                        if not cat_data.empty:
                            segment_data = cat_data.copy();
                            if last_point is not None: segment_data = pd.concat([last_point, segment_data], ignore_index=True).sort_values(y_col_cumul)
                            line_color = colors[i % len(colors)]
                            fig.add_trace(go.Scatter( x=segment_data[x_col_actual], y=segment_data[y_col_cumul], name=f"{category}", mode='lines+markers', yaxis="y2", line=dict(color=line_color, dash='dash', width=2), marker=dict(size=5), showlegend=True ), secondary_y=True)
                            last_point = cat_data.iloc[-1:]
                    fig.update_layout( title=title, xaxis_title=x_title_trans, yaxis_title=trans_sales_qty, yaxis2=dict( title=y2_title_trans, range=[0, 105], overlaying='y', side='right', showgrid=False ), legend_title_text=get_translation_streamlit(current_lang, 'common.categories', 'Categories'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) )
                else: raise ValueError(f"Missing cols fig6. Need X='{x_col_actual}', Sales='{y_col_sales}', Cumul='{y_col_cumul}', Cat='{category_col_actual}'")
            elif fig_key == 'fig7':
                y_col_qty = find_column(df, [trans_qty_sold, 'sales_quantity', 'Quantity Sold'], title, exclude=x_col_actual);
                if not y_col_qty: raise ValueError(f"Could not find Y-axis (Quantity Sold) column for fig7.")
                x_col_price = find_column(df, [x_title_trans, trans_sale_price, 'salePrice', 'Sale Price'], title, exclude=y_col_qty);
                if not x_col_price: raise ValueError(f"Could not find X-axis (Sale Price) column for fig7 (different from Y='{y_col_qty}').")
                col_name_hover = find_column(df, [trans_product, 'name', 'Product Name'], title, exclude=x_col_price);
                hover_cols = [col_name_hover] if col_name_hover else None;
                # --- إزالة حساب خط الاتجاه هنا ---
                # x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col_price]);
                # trendline_opt = "lowess" if x_is_numeric else None # تم تعيينها مباشرة إلى None بالأسفل
                # if not x_is_numeric: logger.warning(f"[{fig_key}] X-axis ('{x_col_price}') not numeric. Trendline disabled.")
                df_plot = df.copy(); x_col_plot = x_col_price
                fig = px.scatter(df_plot, x=x_col_plot, y=y_col_qty, title=title,
                                 labels={x_col_plot: x_title_trans, y_col_qty: y_title_trans},
                                 trendline=None, # <-- التعديل: تعطيل خط الاتجاه دائماً
                                 hover_data=hover_cols)
            elif fig_key == 'fig9':
                y_col_stock = find_column(df, [y_title_trans, trans_current_stock, 'current_stock'], title);
                if x_col_actual and y_col_stock: fig = px.bar(df, x=x_col_actual, y=y_col_stock, title=title, labels={x_col_actual: x_title_trans, y_col_stock: y_title_trans}); fig.update_layout(xaxis={'categoryorder':'total ascending'})
                else: raise ValueError(f"Missing cols fig9. Name='{x_col_actual}', Stock='{y_col_stock}'")
            elif fig_key == 'fig10':
                y_col_days = find_column(df, [y_title_trans, trans_days, 'days_since_last_sale'], title);
                col_cat = find_column(df, [trans_period, 'days_category', 'Stagnancy Period'], title); hover_data_list = []
                col_id_hover = find_column(df, [trans_id_hover, 'product_id', 'product_id_str', 'ID'], title); col_last_sale_hover = find_column(df, [trans_last_sale_hover, 'last_sale_date_str', 'Last Sale'], title); col_stock_hover = find_column(df, [trans_stock, 'current_stock', 'current_stock_str', 'Stock'], title)
                main_cols = {x_col_actual, y_col_days, col_cat}
                if col_id_hover and col_id_hover not in main_cols: hover_data_list.append(col_id_hover)
                if col_last_sale_hover and col_last_sale_hover not in main_cols: hover_data_list.append(col_last_sale_hover)
                if col_stock_hover and col_stock_hover not in main_cols: hover_data_list.append(col_stock_hover)
                if x_col_actual and y_col_days and col_cat:
                    fig = px.bar(df, x=x_col_actual, y=y_col_days, color=col_cat, title=title, labels={x_col_actual: x_title_trans, y_col_days: y_title_trans, col_cat: trans_period}, hover_data=hover_data_list if hover_data_list else None )
                    fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, xaxis={'categoryorder':'total descending'}, hovermode="x unified");
                    try: y_val = float(180); fig.add_hline(y=y_val, line_dash="dot", annotation_text=get_translation_streamlit(current_lang,'common.6_month_limit','6 Month Limit'), annotation_position="top right", line_color="red")
                    except: pass
                else: raise ValueError(f"Missing cols fig10. Name='{x_col_actual}', Days='{y_col_days}', Cat='{col_cat}'")
            elif fig_key == 'fig11':
                col_value = find_column(df, ['value', 'actual', 'forecast'], title); col_type = find_column(df, ['type'], title); col_lower = find_column(df, ['lower_ci'], title); col_upper = find_column(df, ['upper_ci'], title)
                if x_col_actual and col_value and col_type:
                    df[x_col_actual] = pd.to_datetime(df[x_col_actual]); df = df.sort_values(x_col_actual); df_actual_all = df[df[col_type] == 'actual']; df_forecast_all = df[df[col_type] == 'forecast']
                    df_actual_filtered = df_actual_all
                    if not df_actual_all.empty: latest_actual_date = df_actual_all[x_col_actual].max(); cutoff_date = latest_actual_date - timedelta(days=89); df_actual_filtered = df_actual_all[df_actual_all[x_col_actual] >= cutoff_date]; logger.info(f"[{fig_key}] Filtering actual data from {cutoff_date.date()} onwards. Kept {len(df_actual_filtered)} of {len(df_actual_all)} actual points.")
                    df_actual = df_actual_filtered; df_forecast = df_forecast_all
                    if not df_actual.empty: fig.add_trace(go.Scatter(x=df_actual[x_col_actual], y=df_actual[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig11.series.actual','Actual Sales'), marker=dict(color='rgba(0, 116, 217, 0.8)', size=5), line=dict(color='rgba(0, 116, 217, 0.8)', width=2)))
                    if not df_forecast.empty:
                        connected_forecast_df = df_forecast.copy();
                        if not df_actual.empty: last_actual_row = df_actual.iloc[-1]; connection_point_data = {x_col_actual: last_actual_row[x_col_actual], col_value: last_actual_row[col_value]};
                        if col_lower and col_lower in df_forecast.columns: connection_point_data[col_lower] = np.nan
                        if col_upper and col_upper in df_forecast.columns: connection_point_data[col_upper] = np.nan;
                        connection_df = pd.DataFrame([connection_point_data]); connected_forecast_df = pd.concat([connection_df, connected_forecast_df], ignore_index=True).sort_values(x_col_actual)
                        fig.add_trace(go.Scatter(x=connected_forecast_df[x_col_actual], y=connected_forecast_df[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig11.series.forecast','Forecasted Sales'), line=dict(color='rgba(255, 65, 54, 0.9)', dash='dash', width=2), marker=dict(size=5)))
                        if col_lower and col_upper and col_lower in connected_forecast_df.columns and col_upper in connected_forecast_df.columns:
                            ci_valid_df = connected_forecast_df.iloc[1:].dropna(subset=[col_lower, col_upper]);
                            if not ci_valid_df.empty: fig.add_trace(go.Scatter(x=ci_valid_df[x_col_actual].tolist()+ci_valid_df[x_col_actual].tolist()[::-1], y=ci_valid_df[col_upper].tolist()+ci_valid_df[col_lower].tolist()[::-1], fill='toself', fillcolor='rgba(255, 65, 54, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=get_translation_streamlit(current_lang, 'fig11.series.ci','95% CI'), showlegend=True, hoverinfo='skip'))
                    fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    series_to_concat = []
                    if not df_actual.empty: series_to_concat.append(df_actual[x_col_actual])
                    if not df_forecast.empty: series_to_concat.append(df_forecast[x_col_actual])
                    if series_to_concat:
                        all_dates_plot = pd.concat(series_to_concat).unique(); all_dates_plot = pd.to_datetime(all_dates_plot)
                        if len(all_dates_plot) > 0: min_date, max_date = all_dates_plot.min(), all_dates_plot.max(); delta = (max_date - min_date).days if max_date > min_date else 1; margin = timedelta(days=max(7, delta*0.05)); fig.update_xaxes(range=[min_date - margin, max_date + margin])
                    else: logger.warning(f"[{fig_key}] No dates found to set x-axis range.")
                else: raise ValueError(f"Missing cols fig11. Date='{x_col_actual}', Val='{col_value}', Type='{col_type}'")
            elif fig_key == 'fig12':
                col_value = find_column(df, ['value'], title); col_type = find_column(df, ['type'], title); col_lower = find_column(df, ['lower_ci'], title); col_upper = find_column(df, ['upper_ci'], title)
                if x_col_actual and col_value and col_type:
                    try: df['sort_key'] = pd.to_datetime(df[x_col_actual], format='%Y-%m')
                    except: df['sort_key'] = df[x_col_actual];
                    df = df.sort_values('sort_key'); df_actual = df[df[col_type] == 'actual']; df_forecast = df[df[col_type] == 'forecast']
                    if not df_actual.empty: fig.add_trace(go.Scatter(x=df_actual[x_col_actual], y=df_actual[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig12.series.actual','Actual Avg'), line=dict(color='royalblue')))
                    if not df_forecast.empty:
                        connected_forecast_df = df_forecast.copy();
                        if not df_actual.empty: last_actual_row = df_actual.iloc[-1]; connection_point_data = {x_col_actual: last_actual_row[x_col_actual], col_value: last_actual_row[col_value], 'sort_key': last_actual_row['sort_key']};
                        if col_lower and col_lower in df_forecast.columns: connection_point_data[col_lower] = np.nan
                        if col_upper and col_upper in df_forecast.columns: connection_point_data[col_upper] = np.nan;
                        connection_df = pd.DataFrame([connection_point_data]); connected_forecast_df = pd.concat([connection_df, connected_forecast_df], ignore_index=True).sort_values('sort_key')
                        fig.add_trace(go.Scatter(x=connected_forecast_df[x_col_actual], y=connected_forecast_df[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig12.series.forecast','Forecasted Avg'), line=dict(color='firebrick', dash='dash'), marker=dict(size=5)))
                        if col_lower and col_upper and col_lower in connected_forecast_df.columns and col_upper in connected_forecast_df.columns:
                            ci_valid_df = connected_forecast_df.iloc[1:].dropna(subset=[col_lower, col_upper]);
                            if not ci_valid_df.empty: fig.add_trace(go.Scatter(x=ci_valid_df[x_col_actual].tolist()+ci_valid_df[x_col_actual].tolist()[::-1], y=ci_valid_df[col_upper].tolist()+ci_valid_df[col_lower].tolist()[::-1], fill='toself', fillcolor='rgba(255, 100, 100, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=get_translation_streamlit(current_lang, 'fig12.series.ci','95% CI'), showlegend=True, hoverinfo='skip'))
                    fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, hovermode="x unified", xaxis={'type':'category'})
                else: raise ValueError(f"Missing cols fig12. Date='{x_col_actual}', Val='{col_value}', Type='{col_type}'")
            elif fig_key == 'fig13':
                y_col_name = find_column(df, [trans_product, 'name'], title); col_id = find_column(df, ['product_id'], title); hover_cols = [col_id] if col_id else None
                if x_col_actual and y_col_name: fig = px.bar(df, x=x_col_actual, y=y_col_name, orientation='h', title=title, labels={x_col_actual: x_title_trans, y_col_name: y_title_trans}, color=x_col_actual, color_continuous_scale=px.colors.sequential.Viridis, hover_data=hover_cols); fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title=x_title_trans, yaxis_title=y_title_trans);
                else: raise ValueError(f"Missing cols fig13. Qty='{x_col_actual}', Name='{y_col_name}'")
            elif fig_key == 'fig14':
                 if df is not None and not df.empty: st.subheader(title); st.dataframe(df, use_container_width=True)
                 else: st.info(f"No data available for table: '{title}'")
                 return None # Indicate handled directly
            elif fig_key == 'fig15':
                y_col_name = find_column(df, [trans_product, 'name'], title); col_id = find_column(df, ['product_id'], title); hover_cols = [col_id] if col_id else None
                if x_col_actual and y_col_name: fig = px.bar(df, x=x_col_actual, y=y_col_name, orientation='h', title=title, labels={x_col_actual: x_title_trans, y_col_name: y_title_trans}, color=x_col_actual, color_continuous_scale=px.colors.sequential.Viridis, hover_data=hover_cols); fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title=x_title_trans, yaxis_title=y_title_trans);
                else: raise ValueError(f"Missing cols fig15. Qty='{x_col_actual}', Name='{y_col_name}'")
            elif fig_key == 'fig17':
                y_col_amount = find_column(df, [y_title_trans, trans_outstanding, 'outstanding_amount'], title);
                if not y_col_amount: raise ValueError("Cannot find outstanding amount column for fig17.")
                if x_col_actual and y_col_amount:
                    try: df[y_col_amount] = pd.to_numeric(df[y_col_amount], errors='coerce').fillna(0)
                    except Exception as e_conv: raise ValueError(f"Y-axis column '{y_col_amount}' could not be converted to numeric: {e_conv}")
                    df[x_col_actual] = df[x_col_actual].astype(str); total_outstanding = df[y_col_amount].sum(); total_outstanding_str = f"{total_outstanding:,.2f}" if isinstance(total_outstanding, (int, float, np.number)) else "N/A"
                    fig = px.bar(df, x=x_col_actual, y=y_col_amount, color=y_col_amount, title=title, labels={x_col_actual: x_title_trans, y_col_amount: y_title_trans}, color_continuous_scale="Inferno");
                    fig.update_coloraxes(colorbar_title=y_title_trans); fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, xaxis={'categoryorder': 'total descending', 'type': 'category'})
                    total_outstanding_label = get_translation_streamlit(current_lang, 'fig17.annotation.total', 'Total Outstanding')
                    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text=f"{total_outstanding_label}: {total_outstanding_str}", showarrow=False, font=dict(size=14), bgcolor="rgba(211,211,211,0.7)", bordercolor="black", borderwidth=1, borderpad=4, align="right")
                else: raise ValueError(f"Missing required columns for fig17. X found: '{x_col_actual}', Y found: '{y_col_amount}'")
            else:
                logger.warning(f"No specific logic for {fig_key} using DataFrame. Check if it should be handled differently."); return fig.update_layout(annotations=[{'text': f"Unhandled Figure Type: {fig_key}", 'showarrow': False}])

            # --- Add description and final check for DataFrame charts ---
            if fig and description: fig.update_layout(annotations=[dict(text=description, showarrow=False, xref='paper', yref='paper', x=0.5, y=-0.15, xanchor='center', yanchor='top', align='center')])
            if fig and not fig.data: logger.warning(f"[{fig_key}] DataFrame Figure created, but no data traces added."); return fig.update_layout(annotations=[{'text': get_translation_streamlit(current_lang,'error.no_series_plotted',"No data series could be plotted"), 'showarrow': False}])
            logger.info(f"Successfully created DataFrame-based figure object for key: {fig_key}")
            return fig # Return the figure created from DataFrame

        # --- CASE 3: Other Dictionary Data (Not Pie) ---
        elif isinstance(data, dict):
             # --- *** MODIFY: Check if chart_type is TABLE for fig14 *** ---
             if chart_type_str == ChartType.TABLE.value and fig_key == 'fig14':
                 logger.info(f"[{fig_key}] Handling table data directly.")
                 df_table = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', [])) # Handle potential nested data
                 if not df_table.empty:
                     st.subheader(title)
                     st.dataframe(df_table, use_container_width=True)
                 else:
                     st.info(f"No data available for table: '{title}'")
                 return None # Handled directly
             # --- *** END MODIFY *** ---
             else:
                 logger.warning(f"[{fig_key}] Received dictionary data but chart type is not handled ('{chart_type_str}').")
                 st.error(f"Cannot display {fig_key} ({title}): Unsupported dictionary format for chart type '{chart_type_str}'.")
                 return go.Figure().update_layout(title=title, annotations=[{'text': f"Unsupported Dictionary Format for {chart_type_str}", 'showarrow': False}])

        # --- CASE 4: Unsupported Data Format ---
        else:
             logger.error(f"[{fig_key}] MAJOR ERROR: Unexpected data type received: {type(data)}")
             st.error(f"Cannot display {fig_key} ({title}): Critical error with data format.")
             return go.Figure().update_layout(title=title, annotations=[{'text': f"Critical Error: Unexpected Data Format {type(data)}", 'showarrow': False}])

    except ValueError as ve: st.error(f"Data Error creating '{fig_key}' ('{title}'): {ve}"); logger.error(f"ValueError creating '{fig_key}': {ve}", exc_info=False); return go.Figure().update_layout(title=title, annotations=[{'text': f"Data Error: Check Logs", 'showarrow': False}])
    except Exception as e: st.error(f"Critical error creating '{fig_key}' ('{title}'): {e}"); logger.critical(f"Critical error creating '{fig_key}': {e}", exc_info=True); return go.Figure().update_layout(title=title, annotations=[{'text': f"Plotting Error: Check Logs", 'showarrow': False}])


# --- Main UI (Tabs Layout) ---
st.title(" Client Analytics Dashboard (Direct DB V5.3)") # Version Bump
st.markdown("Select a client and language to view their dashboards. Data is processed on demand.")

if not pipeline: st.error("Pipeline initialization failed. Cannot load dashboard."); st.stop()

col_select1, col_select2 = st.columns([1, 1])
with col_select1:
    available_clients = get_clients_from_db(pipeline)
    if not available_clients: st.error("Could not fetch client list from database."); st.stop()
    default_client_index = 0
    if "185" in available_clients:
        try: default_client_index = available_clients.index("185")
        except ValueError: pass
    selected_client_str = st.selectbox("Select Client ID:", available_clients, index=default_client_index, key="client_sel_db")
    selected_client_id = int(selected_client_str) if selected_client_str and selected_client_str.isdigit() else None

with col_select2:
    lang_options = SUPPORTED_LANGUAGES
    default_lang_index = lang_options.index('ar') if 'ar' in lang_options else 0
    selected_lang = st.selectbox("Select Language:", lang_options, index=default_lang_index, key="lang_sel_db")

FIGURE_FREQUENCY_MAP = {
    'fig11': 'daily', 'fig17': 'daily',
    'fig1': 'weekly', 'fig4': 'weekly', 'fig9': 'weekly',
    'fig2': 'monthly', 'fig3': 'monthly', 'fig5': 'monthly', 'fig6': 'monthly',
    'fig7': 'monthly', 'fig10': 'monthly', 'fig12': 'monthly', 'fig13': 'monthly',
    'fig15': 'monthly', 'fig16': 'monthly',
    'fig14': 'quarterly',
}

if selected_client_id is not None:
    st.divider()
    st.header(f"Dashboard for Client: {selected_client_id} (Lang: {selected_lang})", anchor=False)
    logger.info(f"Displaying dashboard V5.3 - Client: {selected_client_id}, Lang: {selected_lang}")

    all_prepared_data = run_pipeline_and_get_data(pipeline, selected_client_id, selected_lang) # Get ChartData objects

    if all_prepared_data is None:
        st.error("Failed to process data for the selected client. Check application logs for details.")
    else:
        tab_titles = { freq: get_translation_streamlit(selected_lang, f"frequencies.{freq}", freq.capitalize()) for freq in ["daily", "weekly", "monthly", "quarterly"] }
        tab_daily, tab_weekly, tab_monthly, tab_quarterly = st.tabs([ f" {tab_titles['daily']}", f" {tab_titles['weekly']}", f" {tab_titles['monthly']}", f" {tab_titles['quarterly']}" ])
        tabs_config = { "daily": tab_daily, "weekly": tab_weekly, "monthly": tab_monthly, "quarterly": tab_quarterly }

        figures_by_frequency: Dict[str, Dict[str, Optional[ChartData]]] = { freq: {} for freq in tabs_config }
        for fig_key, chart_data_obj in all_prepared_data.items():
            freq = FIGURE_FREQUENCY_MAP.get(fig_key)
            if freq: figures_by_frequency[freq][fig_key] = chart_data_obj
            else: logger.warning(f"Figure '{fig_key}' has no defined frequency in FIGURE_FREQUENCY_MAP.")

        for freq, tab in tabs_config.items():
            with tab:
                st.subheader(f"{tab_titles[freq]} Reports")
                figures_for_tab = figures_by_frequency[freq]
                if not figures_for_tab: st.info(f"No visualizations defined or processed for {freq} frequency."); continue

                figures_drawn = 0; figure_keys_sorted = sorted(figures_for_tab.keys(), key=lambda x: int(x.replace('fig','')) if x.startswith('fig') and x[3:].isdigit() else 99)
                num_cols = 2; cols = st.columns(num_cols); col_index = 0

                for fig_key in figure_keys_sorted:
                    chart_data_obj = figures_for_tab.get(fig_key) # Get ChartData object
                    with cols[col_index % num_cols]:
                        if chart_data_obj:
                            try:
                                plotly_fig = create_plotly_figure_replica(fig_key, chart_data_obj, selected_lang) # Pass ChartData
                                if plotly_fig: st.plotly_chart(plotly_fig, use_container_width=True); figures_drawn += 1; col_index += 1
                                elif fig_key in ['fig14', 'fig16']: figures_drawn += 1; col_index += 1 # Handled directly
                                else: logger.warning(f"create_plotly_figure_replica returned None for {fig_key} (Freq: {freq}).")
                            except Exception as e_plot: st.error(f"Failed to display {fig_key}: {e_plot}"); logger.error(f"Error rendering {fig_key} (Freq: {freq}): {e_plot}", exc_info=True); col_index += 1
                        else:
                            st.warning(f"Could not load data for: {fig_key}"); logger.warning(f"ChartData object was None for {fig_key} (Freq: {freq})"); col_index += 1

                if figures_drawn == 0 and figures_for_tab: st.warning(f"Could not display any visualizations for {freq} frequency."); logger.warning(f"No figures drawn for freq '{freq}', client {selected_client_id}, lang {selected_lang}, although figures were expected.")

else:
    st.info("Please select a Client ID to run the analysis.")

# --- Sidebar ---
st.sidebar.header("App Information")
st.sidebar.info(f"Data Source: Direct Database Connection")
if pipeline and pipeline.engine:
    try: db_url = pipeline.engine.url; st.sidebar.info(f"DB Host: {db_url.host}"); st.sidebar.info(f"DB Name: {db_url.database}")
    except Exception: st.sidebar.warning("DB Status: Error getting URL details")
else: st.sidebar.warning("DB Status: Not Connected")
st.sidebar.info(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
st.sidebar.caption(f"Dashboard Version: Direct DB V5.3") # Version Bump

logger.info("--- Streamlit dashboard V5.3 script execution finished ---")
