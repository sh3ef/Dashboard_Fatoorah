# dashboard_app.py - V3.13 - Add markers to fig11 forecast line
import streamlit as st
import requests
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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Basic Settings ---
st.set_page_config(layout="wide", page_title="Client Dashboard Replica V3.13", page_icon="📊") # Version Bump

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/api/v1")
RESULTS_DIR = Path("data/client_results")
SUPPORTED_LANGUAGES = ['ar', 'en']

# --- Localization Helpers ---
LOCALE_DIR_STREAMLIT = Path("locales")
_translations_streamlit: Dict[str, Dict[str, Any]] = {}

def _load_translations_streamlit():
    global _translations_streamlit
    if _translations_streamlit: return
    logger.info(f"Streamlit: Loading translations from: {LOCALE_DIR_STREAMLIT.resolve()}")
    if not LOCALE_DIR_STREAMLIT.is_dir(): logger.error(f"Streamlit: Locale dir not found: {LOCALE_DIR_STREAMLIT.resolve()}"); return
    for lang in SUPPORTED_LANGUAGES:
        f_path = LOCALE_DIR_STREAMLIT / f"{lang}.json"
        if f_path.is_file():
            try:
                with open(f_path, "r", encoding="utf-8") as f: _translations_streamlit[lang] = json.load(f)
                logger.info(f"Streamlit: Loaded translation: {f_path.name}")
            except Exception as e: logger.error(f"Streamlit: Error loading {f_path.name}: {e}")
        else: logger.warning(f"Streamlit: Translation file not found: {f_path}")
_load_translations_streamlit()

def get_translation_streamlit(lang: str, key: str, default: str = "") -> str:
    """Gets translation for a specific key using dot notation (e.g., 'common.product')."""
    normalized_lang = lang.lower()
    dic = _translations_streamlit.get(normalized_lang, {})
    if not dic: return default
    try:
        keys = key.split('.')
        val = dic
        for k in keys:
            if isinstance(val, dict): val = val[k]
            else: raise TypeError("Invalid path segment")
        return str(val) if val is not None else default
    except (KeyError, TypeError): return default
    except Exception as e: logger.error(f"Unexpected error getting translation for key '{key}', lang '{normalized_lang}': {e}"); return default
# --- End Localization ---


# --- Data Fetching Helpers ---
@st.cache_data(ttl=600)
def get_available_clients(results_dir: Path) -> List[str]:
    client_ids = []; logger.info(f"Scanning for clients in: {results_dir.resolve()}")
    if results_dir.is_dir():
        try:
            for item in results_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    # Check for presence of at least one expected file pattern
                    if list(item.glob('*_en.json')) or list(item.glob('*_ar.json')):
                        client_ids.append(item.name); logger.debug(f"Found client dir: {item.name}")
                    else:
                        logger.warning(f"Skipping client dir '{item.name}': No JSON files found.")
        except Exception as e: st.error(f"Error scanning results dir: {e}"); logger.error(f"Error scanning '{results_dir.resolve()}': {e}", exc_info=True)
    else: st.warning(f"Results directory not found: {results_dir.resolve()}")
    if not client_ids: logger.warning("No valid client directories found.")
    else: logger.info(f"Found {len(client_ids)} client(s): {client_ids}")
    return sorted(client_ids, key=int)

@st.cache_data(ttl=300)
def fetch_data_from_api(client_id: str, frequency: str, lang: str) -> Optional[Dict[str, Any]]:
    endpoint_map = { "daily": f"{API_BASE_URL}/daily/all/{client_id}", "weekly": f"{API_BASE_URL}/weekly/all/{client_id}", "monthly": f"{API_BASE_URL}/monthly/all/{client_id}", "quarterly": f"{API_BASE_URL}/quarterly/all/{client_id}"}
    url = endpoint_map.get(frequency); headers = {"X-Language": lang}; logger.info(f"Fetching {frequency} data for client {client_id} (lang={lang}) from {url}")
    try: response = requests.get(url, headers=headers, timeout=45); response.raise_for_status(); logger.info(f"Successfully fetched {frequency} data."); return response.json()
    except requests.exceptions.RequestException as e: st.warning(f"API Request Failed ({frequency}): {e}"); logger.warning(f"API Request Failed for {url}: {e}", exc_info=False); return None
    except json.JSONDecodeError as e: st.error(f"API Error ({frequency}): Invalid JSON received."); logger.error(f"JSON Decode Error for {url}. Text: {response.text[:200]}...", exc_info=True); return None
    except Exception as e: st.error(f"Unexpected error fetching {frequency} data: {e}"); logger.error(f"Unexpected error fetching data from {url}: {e}", exc_info=True); return None
# --- End Data Fetching ---

# --- Column Finder ---
def find_column(df: pd.DataFrame, possibilities: List[Optional[str]], chart_title: str = "Chart", exclude: Optional[str] = None) -> Optional[str]:
    """Finds the first matching column name from a list of possibilities."""
    # Log inputs
    df_cols_list = df.columns.tolist() if df is not None and not df.empty else ["<Empty or None DF>"]
    logger.info(f"[{chart_title}] Finding column. Possibilities: {possibilities}, Exclude: {exclude}, DF Cols: {df_cols_list}")

    if df is None or df.empty:
        logger.warning(f"[{chart_title}] DataFrame is None or empty, cannot find column.")
        return None

    clean_possibilities = [p for p in possibilities if isinstance(p, str) and p]
    df_cols_lower = {col.lower(): col for col in df.columns}

    def check_and_return(col_name: Optional[str]) -> Optional[str]:
        """Helper to check if col_name is valid and not excluded."""
        if col_name and col_name != exclude: return col_name
        return None

    # 1. Check primary possibilities (case-sensitive first)
    for name in clean_possibilities:
        col = check_and_return(name)
        if col and col in df.columns:
            logger.info(f"[{chart_title}] Found column '{col}' using possibility '{name}' (case-sensitive).")
            return col
        # Check case-insensitive
        actual_col_insensitive = df_cols_lower.get(name.lower())
        col = check_and_return(actual_col_insensitive)
        if col:
            logger.info(f"[{chart_title}] Found column '{col}' using possibility '{name}' (case-insensitive).")
            return col

    # 2. Check common fallbacks if primary failed
    common_fallbacks = [ 'name', 'product', 'Product', 'المنتج', 'date', 'sale_date', 'Date', 'التاريخ', 'value', 'Value', 'القيمة', 'quantity', 'Quantity', 'sales_quantity', 'Quantity Sold', 'الكمية', 'الكمية المباعة', 'stock', 'Stock', 'current_stock', 'Current Stock', 'المخزون', 'المخزون الحالي', 'amount', 'Amount', 'totalPrice', 'المبلغ', 'price', 'Price', 'salePrice', 'Sale Price', 'سعر البيع', 'category', 'Category', 'الفئة', 'user_id', 'supplier_id', 'Supplier (ID)', 'المورد', 'المورد (المعرف)', 'outstanding_amount', 'Outstanding Amount', 'المبلغ المستحق', 'forecast', 'actual', 'lower_ci', 'upper_ci', 'type', 'COGS', 'cogs', 'تكلفة البضاعة', 'margin', 'Margin', 'هامش الربح', 'revenue', 'Revenue', 'الإيرادات', 'sales_amount', 'profit', 'Profit', 'netProfit', 'الربح', 'cumulative_percentage', 'النسبة التراكمية %', 'efficiency', 'efficiency_category', 'days_since_last_sale', 'days_category', 'label', 'id', 'product_id', 'percentage_category'] # Added percentage_category
    all_possibilities = list(dict.fromkeys(clean_possibilities + common_fallbacks))
    checked_fallbacks = set(clean_possibilities)

    for name in all_possibilities:
        if name and name not in checked_fallbacks:
             checked_fallbacks.add(name)
             col = check_and_return(name)
             if col and col in df.columns:
                 logger.info(f"[{chart_title}] Found column '{col}' using fallback possibility '{name}' (case-sensitive).")
                 return col
             actual_col_insensitive = df_cols_lower.get(name.lower())
             col = check_and_return(actual_col_insensitive)
             if col:
                 logger.info(f"[{chart_title}] Found column '{col}' using fallback possibility '{name}' (case-insensitive).")
                 return col

    logger.warning(f"[{chart_title}] Column NOT FOUND from possibilities: {possibilities} or fallbacks (excluding '{exclude}')")
    return None
# --- End Column Finder ---


# --- Main Plotting Function ---
def create_plotly_figure_replica(fig_key: str, chart_data: Dict[str, Any], current_lang: str) -> Optional[go.Figure]:
    logger.info(f"Attempting V3.13 replication for key: {fig_key}") # Incremented version number
    fig: Optional[go.Figure] = None; df: Optional[pd.DataFrame] = None
    title: str = f"Figure: {fig_key}"; description: str = ""; metadata: Dict = {}

    try:
        # --- Initial Data Validation & Setup ---
        if not chart_data or not isinstance(chart_data, dict) or 'metadata' not in chart_data or not isinstance(chart_data['metadata'], dict) or 'data' not in chart_data:
            st.warning(f"[{fig_key}] Invalid chart data format."); logger.warning(f"[{fig_key}] Invalid format.")
            return go.Figure().update_layout(title=title, annotations=[{'text': "Invalid Data Format", 'showarrow': False}])
        metadata = chart_data['metadata']; data = chart_data['data']
        chart_type_raw = metadata.get('chart_type', 'table'); chart_type = chart_type_raw.lower() if isinstance(chart_type_raw, str) else 'table'
        title = metadata.get('title', f"Figure: {fig_key}"); description = metadata.get('description', '')
        x_axis_info = metadata.get('x_axis', {}); y_axis_info = metadata.get('y_axis', {})
        x_title_trans = x_axis_info.get('title', 'X'); x_name_internal = x_axis_info.get('name')
        y_title_trans = y_axis_info.get('title', 'Y'); y2_title_trans = metadata.get('y2_axis', {}).get('title', 'Y2') # Might not exist for fig6
        fig = go.Figure().update_layout(title=title, xaxis_title=x_title_trans, yaxis_title=y_title_trans, legend_title_text=get_translation_streamlit(current_lang, 'common.series', 'Series'), margin=dict(l=40, r=20, t=60, b=40), hovermode="x unified", uirevision=title) # uirevision helps maintain zoom
        if not data: logger.warning(f"[{fig_key}] No data."); st.info(f"No data for: '{title}'"); return fig.update_layout(annotations=[{'text': "No Data", 'showarrow': False}])

        # --- Create DataFrame ---
        if isinstance(data, list):
            if not data: return fig.update_layout(annotations=[{'text': "Empty Data", 'showarrow': False}])
            df = pd.DataFrame(data)
            if df.empty: return fig.update_layout(annotations=[{'text': "Empty Data", 'showarrow': False}])
        elif isinstance(data, dict) and (chart_type == 'pie' or fig_key == 'fig16'): df = None
        elif isinstance(data, dict):
             first_list = next((v for v in data.values() if isinstance(v, list)), None)
             if first_list: df = pd.DataFrame(first_list)
             if df is None or df.empty: raise ValueError("Cannot create DataFrame from dict data.")
        else: raise ValueError(f"Unsupported data type: {type(data)}")

        # --- Define Translated Variables ---
        trans_product = get_translation_streamlit(current_lang, 'common.product', 'Product')
        trans_quantity = get_translation_streamlit(current_lang, 'common.quantity', 'Quantity')
        trans_qty_sold = get_translation_streamlit(current_lang, 'common.quantity_sold', 'Quantity Sold')
        trans_stock = get_translation_streamlit(current_lang, 'common.stock', 'Stock')
        trans_amount = get_translation_streamlit(current_lang, 'common.amount', 'Amount')
        trans_current_stock = get_translation_streamlit(current_lang, 'fig1.series.stock', 'Current Stock')
        trans_sales = get_translation_streamlit(current_lang, 'fig1.series.sales', 'Sales')
        trans_supplier_id = get_translation_streamlit(current_lang, 'common.supplier_id', 'Supplier ID')
        trans_outstanding = get_translation_streamlit(current_lang, 'common.outstanding_amount', 'Outstanding Amount')
        trans_sale_price = get_translation_streamlit(current_lang, 'common.sale_price', 'Sale Price')
        trans_days = get_translation_streamlit(current_lang, 'common.days_since_last_sale', 'Days Since Last Sale')
        trans_period = get_translation_streamlit(current_lang, 'common.stagnancy_period', 'Stagnancy Period')
        trans_id_hover = get_translation_streamlit(current_lang, 'common.product_id', 'ID')
        trans_last_sale_hover = get_translation_streamlit(current_lang, 'common.last_sale_date', 'Last Sale')
        # ---------------------------------

        # --- FIGURE-SPECIFIC LOGIC ---
        if df is not None: # Logic requiring DataFrame
            x_col_actual = find_column(df, [x_title_trans, x_name_internal], title)
            if not x_col_actual:
                if df.columns.any(): x_col_actual = df.columns[0]; logger.warning(f"[{fig_key}] Defaulting X to '{x_col_actual}'"); fig.update_layout(xaxis_title=x_col_actual)
                else: raise ValueError("DataFrame has no columns.")

            # --- Logic for fig1 to fig5 ---
            if fig_key == 'fig1':
                 y_col_stock = find_column(df, [trans_current_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_sales, 'sales_quantity'], title)
                 if y_col_stock and y_col_sales: fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_current_stock, marker_color='rgb(55, 83, 109)')); fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_sales], name=trans_sales, marker_color='rgb(26, 118, 255)')); fig.update_layout(barmode="group", yaxis_title=trans_quantity)
                 else: raise ValueError(f"Missing cols fig1. Need X='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")

            elif fig_key == 'fig3':
                trans_cogs = get_translation_streamlit(current_lang, 'fig3.series.cogs', 'COGS'); trans_revenue = get_translation_streamlit(current_lang, 'fig3.series.revenue', 'Revenue'); trans_margin = get_translation_streamlit(current_lang, 'fig3.series.margin', 'Margin')
                y_col_cogs = find_column(df, [trans_cogs, 'COGS'], title); y_col_revenue = find_column(df, [trans_revenue, 'sales_amount'], title); y_col_margin = find_column(df, [trans_margin, 'margin', 'Margin'], title)
                if y_col_cogs and y_col_revenue and y_col_margin: fig = px.bar(df, x=x_col_actual, y=[y_col_cogs, y_col_revenue, y_col_margin], barmode="group", title=title, labels={'value': trans_amount, 'variable': 'النوع', x_col_actual: x_title_trans}, color_discrete_map={y_col_cogs: "rgb(255, 127, 14)", y_col_revenue: "rgb(26, 118, 255)", y_col_margin: "rgb(44, 160, 44)"}) # Corrected colors
                else: raise ValueError(f"Missing cols fig3. Need X='{x_col_actual}', COGS='{y_col_cogs}', Rev='{y_col_revenue}', Margin='{y_col_margin}'")

            elif fig_key == 'fig4':
                 y_col_stock = find_column(df, [trans_current_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_sales, 'sales_quantity'], title)
                 if y_col_stock and y_col_sales:
                     fig = make_subplots(specs=[[{"secondary_y": True}]])
                     fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_current_stock, marker_color='lightblue'), secondary_y=False); fig.add_trace(go.Scatter(x=df[x_col_actual], y=df[y_col_sales], mode='lines+markers', name=trans_sales, line=dict(color='orange')), secondary_y=True)
                     fig.update_layout(title=title, xaxis_title=x_title_trans); fig.update_yaxes(title_text=f"{trans_quantity} ({trans_stock})", secondary_y=False); fig.update_yaxes(title_text=f"{trans_quantity} ({trans_sales})", secondary_y=True)
                 else: raise ValueError(f"Missing cols fig4. Need X='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")

            elif fig_key == 'fig5':
                 trans_eff_cat = get_translation_streamlit(current_lang, 'common.efficiency_category', 'Efficiency Category'); trans_total_stock = get_translation_streamlit(current_lang, 'fig5.series.stock', 'Total Stock'); trans_total_sales = get_translation_streamlit(current_lang, 'fig5.series.sales', 'Total Sales')
                 # Find X column using the *translated* category name first
                 x_col_actual = find_column(df, [trans_eff_cat, 'efficiency_category_translated'], title)
                 if not x_col_actual: # Fallback if not found (should be found if backend is correct)
                     x_col_actual = find_column(df, ['efficiency'], title) # Check internal name
                     if not x_col_actual and df.columns.any(): # Last resort
                         x_col_actual = df.columns[0]
                         logger.warning(f"[{fig_key}] Defaulting X to '{x_col_actual}'")
                 y_col_stock = find_column(df, [trans_total_stock, 'current_stock'], title); y_col_sales = find_column(df, [trans_total_sales, 'sales_quantity'], title)
                 if x_col_actual and y_col_stock and y_col_sales:
                     fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_stock], name=trans_total_stock, marker_color='rgb(55, 83, 109)', opacity=0.6)); fig.add_trace(go.Bar(x=df[x_col_actual], y=df[y_col_sales], name=trans_total_sales, marker_color='rgb(26, 118, 255)', opacity=0.6)) # Corrected colors
                     fig.update_layout(barmode='group', xaxis_title=x_title_trans, yaxis_title=get_translation_streamlit(current_lang, 'common.total_quantity','Total Quantity'))
                 else: raise ValueError(f"Missing cols fig5. Cat='{x_col_actual}', Stock='{y_col_stock}', Sales='{y_col_sales}'")

            # --- fig6 Logic (Segmented Colored Line) ---
            elif fig_key == 'fig6':
                 trans_sales_qty = get_translation_streamlit(current_lang, 'fig6.series.sales', 'Sales Quantity')
                 trans_cumul_perc = get_translation_streamlit(current_lang, 'common.cumulative_percentage', 'Cumulative %')
                 y_col_sales = find_column(df, [trans_sales_qty, 'sales_quantity'], title)
                 y_col_cumul = find_column(df, [trans_cumul_perc, 'cumulative_percentage'], title)
                 category_col_actual = find_column(df, ['percentage_category', 'percent_cat'], title)

                 if x_col_actual and y_col_sales and y_col_cumul and category_col_actual:
                     if category_col_actual not in df.columns:
                          logger.error(f"[{fig_key}] Category column '{category_col_actual}' not found. Cols: {df.columns.tolist()}")
                          raise ValueError(f"Required category column '{category_col_actual}' not found for line coloring.")
                     try:
                         df[y_col_cumul] = pd.to_numeric(df[y_col_cumul], errors='coerce')
                         df_sorted = df.dropna(subset=[y_col_cumul]).sort_values(y_col_cumul, ascending=True).copy()
                     except Exception as e_conv:
                         logger.error(f"[{fig_key}] Could not convert cumulative percentage column '{y_col_cumul}' to numeric: {e_conv}")
                         raise ValueError(f"Cumulative percentage column '{y_col_cumul}' error.")
                     if df_sorted.empty:
                          logger.warning(f"[{fig_key}] DataFrame became empty after sorting/cleaning cumulative percentage.")
                          raise ValueError("No valid data remaining for Pareto plot.")

                     fig = make_subplots(specs=[[{"secondary_y": True}]])
                     bar_color = 'cornflowerblue'
                     fig.add_trace(go.Bar( x=df_sorted[x_col_actual], y=df_sorted[y_col_sales], name=trans_sales_qty, marker_color=bar_color ), secondary_y=False)
                     categories = sorted(df_sorted[category_col_actual].unique())
                     colors = px.colors.qualitative.Plotly
                     last_point = None
                     for i, category in enumerate(categories):
                         cat_data = df_sorted[df_sorted[category_col_actual] == category]
                         if not cat_data.empty:
                             segment_data = cat_data.copy()
                             if last_point is not None:
                                 segment_data = pd.concat([last_point, segment_data], ignore_index=True).sort_values(y_col_cumul)
                             line_color = colors[i % len(colors)]
                             fig.add_trace(go.Scatter( x=segment_data[x_col_actual], y=segment_data[y_col_cumul], name=f"{category}", mode='lines+markers', yaxis="y2", line=dict(color=line_color, dash='dash', width=2), marker=dict(size=5), showlegend=True ), secondary_y=True)
                             last_point = cat_data.iloc[-1:]
                     y2_title = get_translation_streamlit(current_lang, 'fig6.y2_axis.title', 'Cumulative %')
                     fig.update_layout( title=title, xaxis_title=x_title_trans, yaxis_title=trans_sales_qty, yaxis2=dict( title=y2_title, range=[0, 105], overlaying='y', side='right', showgrid=False ), legend_title_text=get_translation_streamlit(current_lang, 'common.categories', 'Categories'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) )
                 else:
                     logger.error(f"[{fig_key}] Missing required columns. Need X='{x_col_actual}', Sales='{y_col_sales}', Cumul='{y_col_cumul}', Category='{category_col_actual}'")
                     raise ValueError(f"Missing cols fig6. Check logs for details.")

            elif fig_key == 'fig7':
                 y_col_qty = find_column(df, [trans_qty_sold, 'sales_quantity', 'Quantity Sold'], title, exclude=x_col_actual)
                 if not y_col_qty: raise ValueError(f"Could not find Y-axis (Quantity Sold) column for fig7.")
                 x_col_price = find_column(df, [trans_sale_price, 'salePrice', 'Sale Price'], title, exclude=y_col_qty)
                 if not x_col_price: raise ValueError(f"Could not find X-axis (Sale Price) column for fig7 (different from Y='{y_col_qty}').")
                 col_name_hover = find_column(df, [trans_product, 'name', 'Product Name'], title, exclude=x_col_price)
                 hover_cols = [col_name_hover] if col_name_hover else None
                 x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col_price]); trendline_opt = "lowess" if x_is_numeric else None
                 if not x_is_numeric: logger.warning(f"[{fig_key}] X-axis ('{x_col_price}') not numeric. Trendline disabled.")
                 df_plot = df.copy(); x_col_plot = x_col_price
                 fig = px.scatter(df_plot, x=x_col_plot, y=y_col_qty, title=title,
                                  labels={x_col_plot: x_title_trans, y_col_qty: y_title_trans},
                                  trendline=trendline_opt, hover_data=hover_cols)

            elif fig_key == 'fig9':
                 y_col_stock = find_column(df, [trans_current_stock, 'current_stock'], title)
                 if x_col_actual and y_col_stock: # x_col_actual is product name
                      fig = px.bar(df, x=x_col_actual, y=y_col_stock, title=title, labels={x_col_actual: x_title_trans, y_col_stock: y_title_trans})
                      fig.update_layout(xaxis={'categoryorder':'total ascending'})
                 else: raise ValueError(f"Missing cols fig9. Name='{x_col_actual}', Stock='{y_col_stock}'")

            elif fig_key == 'fig10':
                 y_col_days = find_column(df, [trans_days, 'days_since_last_sale'], title);
                 col_cat = find_column(df, [trans_period, 'days_category', 'Stagnancy Period'], title)
                 hover_data_list = []
                 col_id_hover = find_column(df, [trans_id_hover, 'product_id', 'product_id_str', 'ID'], title);
                 col_last_sale_hover = find_column(df, [trans_last_sale_hover, 'last_sale_date_str', 'Last Sale'], title);
                 col_stock_hover = find_column(df, [trans_stock, 'current_stock', 'current_stock_str', 'Stock'], title)
                 main_cols = {x_col_actual, y_col_days, col_cat}
                 if col_id_hover and col_id_hover not in main_cols: hover_data_list.append(col_id_hover)
                 if col_last_sale_hover and col_last_sale_hover not in main_cols: hover_data_list.append(col_last_sale_hover)
                 if col_stock_hover and col_stock_hover not in main_cols: hover_data_list.append(col_stock_hover)
                 if x_col_actual and y_col_days and col_cat:
                     fig = px.bar(df, x=x_col_actual, y=y_col_days, color=col_cat, title=title,
                                  labels={x_col_actual: x_title_trans, y_col_days: y_title_trans, col_cat: trans_period},
                                  hover_data=hover_data_list if hover_data_list else None )
                     fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, xaxis={'categoryorder':'total descending'}, hovermode="x unified");
                     try: y_val = float(180); fig.add_hline(y=y_val, line_dash="dot", annotation_text="حد 6 أشهر", annotation_position="top right", line_color="red")
                     except: pass
                 else: raise ValueError(f"Missing cols fig10. Name='{x_col_actual}', Days='{y_col_days}', Cat='{col_cat}'")

            # --- START: fig11 with Markers on Forecast ---
            elif fig_key == 'fig11':
                 col_value = find_column(df, ['value', 'actual', 'forecast'], title); col_type = find_column(df, ['type'], title); col_lower = find_column(df, ['lower_ci'], title); col_upper = find_column(df, ['upper_ci'], title)
                 if x_col_actual and col_value and col_type:
                     df[x_col_actual] = pd.to_datetime(df[x_col_actual]); df = df.sort_values(x_col_actual); df_actual = df[df[col_type] == 'actual']; df_forecast = df[df[col_type] == 'forecast']
                     if not df_actual.empty: fig.add_trace(go.Scatter(x=df_actual[x_col_actual], y=df_actual[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig11.series.actual','Actual Sales'), marker=dict(color='rgba(0, 116, 217, 0.8)', size=5), line=dict(color='rgba(0, 116, 217, 0.8)', width=2)))
                     if not df_forecast.empty:
                         connected_forecast_df = df_forecast;
                         if not df_actual.empty: last_actual_row = df_actual.iloc[-1]; connection_point_data = {x_col_actual: last_actual_row[x_col_actual], col_value: last_actual_row[col_value]};
                         if col_lower: connection_point_data[col_lower] = np.nan
                         if col_upper: connection_point_data[col_upper] = np.nan; connected_forecast_df = pd.concat([pd.DataFrame([connection_point_data]), df_forecast], ignore_index=True)
                         # --- ADDING MARKERS HERE ---
                         fig.add_trace(go.Scatter(
                             x=connected_forecast_df[x_col_actual],
                             y=connected_forecast_df[col_value],
                             mode='lines+markers', # Changed to include markers
                             name=get_translation_streamlit(current_lang, 'fig11.series.forecast','Forecasted Sales'),
                             line=dict(color='rgba(255, 65, 54, 0.9)', dash='dash', width=2),
                             marker=dict(size=5) # Optional: Adjust marker size
                         ))
                         # --- END MARKER ADDITION ---
                         if col_lower and col_upper and col_lower in connected_forecast_df.columns and col_upper in connected_forecast_df.columns:
                             ci_valid_df = connected_forecast_df.iloc[1:].dropna(subset=[col_lower, col_upper]);
                             if not ci_valid_df.empty: fig.add_trace(go.Scatter(x=ci_valid_df[x_col_actual].tolist()+ci_valid_df[x_col_actual].tolist()[::-1], y=ci_valid_df[col_upper].tolist()+ci_valid_df[col_lower].tolist()[::-1], fill='toself', fillcolor='rgba(255, 65, 54, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=get_translation_streamlit(current_lang, 'fig11.series.ci','95% CI'), showlegend=True, hoverinfo='skip'))
                     fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                     all_dates = pd.to_datetime(df[x_col_actual].unique());
                     if len(all_dates) > 0: min_date, max_date = all_dates.min(), all_dates.max(); delta = (max_date - min_date).days if max_date > min_date else 1; margin = timedelta(days=max(7, delta*0.05)); fig.update_xaxes(range=[min_date - margin, max_date + margin])
                 else: raise ValueError(f"Missing cols fig11. Date='{x_col_actual}', Val='{col_value}', Type='{col_type}'")
            # --- END: fig11 with Markers on Forecast ---

            elif fig_key == 'fig12':
                 col_value = find_column(df, ['value'], title); col_type = find_column(df, ['type'], title); col_lower = find_column(df, ['lower_ci'], title); col_upper = find_column(df, ['upper_ci'], title)
                 if x_col_actual and col_value and col_type:
                     try: df['sort_key'] = pd.to_datetime(df[x_col_actual], format='%Y-%m')
                     except: df['sort_key'] = df[x_col_actual];
                     df = df.sort_values('sort_key'); df_actual = df[df[col_type] == 'actual']; df_forecast = df[df[col_type] == 'forecast']
                     if not df_actual.empty: fig.add_trace(go.Scatter(x=df_actual[x_col_actual], y=df_actual[col_value], mode='lines+markers', name=get_translation_streamlit(current_lang, 'fig12.series.actual','Actual Avg'), line=dict(color='royalblue')))
                     if not df_forecast.empty:
                         connected_forecast_df = df_forecast;
                         if not df_actual.empty: last_actual_row = df_actual.iloc[-1]; connection_point_data = {x_col_actual: last_actual_row[x_col_actual], col_value: last_actual_row[col_value], 'sort_key': last_actual_row['sort_key']};
                         if col_lower: connection_point_data[col_lower] = np.nan
                         if col_upper: connection_point_data[col_upper] = np.nan; connected_forecast_df = pd.concat([pd.DataFrame([connection_point_data]), df_forecast], ignore_index=True).sort_values('sort_key')
                         # Also add markers to fig12 forecast line? (Optional)
                         fig.add_trace(go.Scatter(
                             x=connected_forecast_df[x_col_actual],
                             y=connected_forecast_df[col_value],
                             mode='lines+markers', # Optional: Add markers here too
                             name=get_translation_streamlit(current_lang, 'fig12.series.forecast','Forecasted Avg'),
                             line=dict(color='firebrick', dash='dash'),
                             marker=dict(size=5) # Optional marker size
                         ))
                         if col_lower and col_upper and col_lower in connected_forecast_df.columns and col_upper in connected_forecast_df.columns:
                             ci_valid_df = connected_forecast_df.iloc[1:].dropna(subset=[col_lower, col_upper]);
                             if not ci_valid_df.empty: fig.add_trace(go.Scatter(x=ci_valid_df[x_col_actual].tolist()+ci_valid_df[x_col_actual].tolist()[::-1], y=ci_valid_df[col_upper].tolist()+ci_valid_df[col_lower].tolist()[::-1], fill='toself', fillcolor='rgba(255, 100, 100, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=get_translation_streamlit(current_lang, 'fig12.series.ci','95% CI'), showlegend=True, hoverinfo='skip'))
                     fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans, hovermode="x unified", xaxis={'type':'category'})
                 else: raise ValueError(f"Missing cols fig12. Date='{x_col_actual}', Val='{col_value}', Type='{col_type}'")

            elif fig_key == 'fig13':
                 y_col_name = find_column(df, [trans_product, 'name'], title) # Y axis
                 col_id = find_column(df, ['product_id'], title) # Hover
                 hover_cols = [col_id] if col_id else None
                 if x_col_actual and y_col_name: # x_col_actual is quantity sold
                      fig = px.bar(df, x=x_col_actual, y=y_col_name, orientation='h', title=title, labels={x_col_actual: x_title_trans, y_col_name: y_title_trans}, color=x_col_actual, color_continuous_scale=px.colors.sequential.Viridis, hover_data=hover_cols) # Changed scale
                      fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title=x_title_trans, yaxis_title=y_title_trans)
                 else: raise ValueError(f"Missing cols fig13. Qty='{x_col_actual}', Name='{y_col_name}'")

            elif fig_key == 'fig14': # Use st.dataframe
                 st.subheader(title); st.dataframe(df, use_container_width=True); return None

            elif fig_key == 'fig15':
                  y_col_name = find_column(df, [trans_product, 'name'], title) # Y axis
                  col_id = find_column(df, ['product_id'], title) # Hover
                  hover_cols = [col_id] if col_id else None
                  if x_col_actual and y_col_name: # x_col_actual is quantity sold
                       fig = px.bar(df, x=x_col_actual, y=y_col_name, orientation='h', title=title, labels={x_col_actual: x_title_trans, y_col_name: y_title_trans}, color=x_col_actual, color_continuous_scale=px.colors.sequential.Viridis, hover_data=hover_cols)
                       fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title=x_title_trans, yaxis_title=y_title_trans)
                  else: raise ValueError(f"Missing cols fig15. Qty='{x_col_actual}', Name='{y_col_name}'")

            elif fig_key == 'fig17':
                 y_col_amount = find_column(df, [trans_outstanding, 'outstanding_amount'], title) # Y axis
                 if x_col_actual and y_col_amount: # x_col_actual should be supplier ID
                     logger.info(f"[{fig_key}] Lang: {current_lang}, Found X: '{x_col_actual}', Found Y: '{y_col_amount}'")
                     logger.info(f"[{fig_key}] Lang: {current_lang}, Original DType of Y column '{y_col_amount}': {df[y_col_amount].dtype}")

                     try:
                         df[y_col_amount] = pd.to_numeric(df[y_col_amount], errors='coerce')
                         if df[y_col_amount].isnull().any():
                             nan_count = df[y_col_amount].isnull().sum()
                             logger.warning(f"[{fig_key}] Found {nan_count} NaN(s) in Y column '{y_col_amount}' after numeric conversion. Filling with 0.")
                             df[y_col_amount] = df[y_col_amount].fillna(0)
                         logger.info(f"[{fig_key}] Lang: {current_lang}, DType of Y column AFTER conversion: {df[y_col_amount].dtype}")
                     except Exception as e_conv:
                         logger.error(f"[{fig_key}] Failed to convert Y column '{y_col_amount}' to numeric: {e_conv}")
                         raise ValueError(f"Y-axis column '{y_col_amount}' could not be converted to numeric.")

                     df[x_col_actual] = df[x_col_actual].astype(str)
                     total_outstanding = df[y_col_amount].sum()
                     total_outstanding_str = f"{total_outstanding:,.2f}" if isinstance(total_outstanding, (int, float, np.number)) else "N/A"
                     fig = px.bar(df, x=x_col_actual, y=y_col_amount, color=y_col_amount, # Color by numeric Y
                                  title=title, labels={x_col_actual: x_title_trans, y_col_amount: y_title_trans},
                                  color_continuous_scale="Inferno") # Use continuous scale
                     fig.update_coloraxes(colorbar_title=y_title_trans)
                     fig.update_layout(xaxis_title=x_title_trans, yaxis_title=y_title_trans,
                                       xaxis={'categoryorder': 'total descending', 'type': 'category'})
                     # --- Get translated annotation text ---
                     # Ensure you add 'fig17.annotation.total' to your en.json and ar.json files
                     total_outstanding_label = get_translation_streamlit(current_lang, 'fig17.annotation.total', 'Total Outstanding')
                     fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98,
                                        text=f"{total_outstanding_label}: {total_outstanding_str}",
                                        showarrow=False, font=dict(size=14), bgcolor="rgba(211,211,211,0.7)",
                                        bordercolor="black", borderwidth=1, borderpad=4, align="right")
                 else:
                     logger.error(f"[{fig_key}] Failed to find required columns. X found: '{x_col_actual}', Y found: '{y_col_amount}'")
                     raise ValueError(f"Missing required columns for fig17. Check logs for find_column details.")

            else: # Fallback for unhandled fig_keys with DataFrame
                logger.warning(f"No specific logic for {fig_key}. Generic plot.")
                y_col_generic = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != x_col_actual), None)
                if y_col_generic: fig.add_trace(go.Scatter(x=df[x_col_actual], y=df[y_col_generic], mode='lines+markers', name=y_col_generic)); fig.update_layout(yaxis_title=y_col_generic)
                else: return fig.update_layout(annotations=[{'text': "No Y-axis", 'showarrow': False}])

        # --- Handle Dictionary Data (Pies - fig16) ---
        elif isinstance(data, dict):
             if chart_type == 'pie':
                 if fig_key == 'fig16': # Specific handling
                     logger.debug(f"[{fig_key}] Handling fig16 pies.")
                     color_map = data.get('color_mapping', {}); rev_title_trans = get_translation_streamlit(current_lang, 'fig16.series.revenue', "Revenue"); st.subheader(rev_title_trans)
                     rev_data = data.get('revenue_data');
                     if rev_data and isinstance(rev_data, list) and rev_data:
                         df_rev = pd.DataFrame(rev_data)
                         if 'label' in df_rev.columns and 'value' in df_rev.columns: fig_rev = px.pie(df_rev, names='label', values='value', title=f"{title} - {rev_title_trans}", color='label', color_discrete_map=color_map); fig_rev.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value'); st.plotly_chart(fig_rev, use_container_width=True)
                         else: st.info(f"Revenue data for '{title}' missing columns.")
                     else: st.info(f"No revenue data for '{title}'.")
                     prof_title_trans = get_translation_streamlit(current_lang, 'fig16.series.profit', "Profit"); st.subheader(prof_title_trans)
                     prof_data = data.get('profit_data');
                     if prof_data and isinstance(prof_data, list) and prof_data:
                         df_prof = pd.DataFrame(prof_data)
                         if 'label' in df_prof.columns and 'value' in df_prof.columns: fig_prof = px.pie(df_prof, names='label', values='value', title=f"{title} - {prof_title_trans}", color='label', color_discrete_map=color_map); fig_prof.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value'); st.plotly_chart(fig_prof, use_container_width=True)
                         else: st.info(f"Profit data for '{title}' missing columns.")
                     else: st.info(f"No profit data for '{title}'.")
                     return None # Handled directly
                 else: # Handle other pies (like fig2)
                     labels, values = None, None
                     if 'labels' in data and 'values' in data: labels, values = data['labels'], data['values']
                     if labels is not None and values is not None and len(labels) == len(values): fig = px.pie(names=labels, values=values, title=title); fig.update_traces(textposition='inside', textinfo='percent+label')
                     else: return fig.update_layout(annotations=[{'text': "Invalid Pie Data", 'showarrow': False}])
             else: return fig.update_layout(annotations=[{'text': "Unsupported Dict Data", 'showarrow': False}])
        # ---------------------------------------------------------

        # Add description if figure exists
        if fig and description:
             fig.update_layout(annotations=[dict(text=description, showarrow=False, xref='paper', yref='paper', x=0.5, y=-0.15, xanchor='center', yanchor='top', align='center')])

        # Final check if fig has data
        if fig and not fig.data:
            logger.warning(f"[{fig_key}] Figure created, but no data traces added.")
            return fig.update_layout(annotations=[{'text': get_translation_streamlit(current_lang,'error.no_series_plotted',"No data series could be plotted"), 'showarrow': False}])

        logger.info(f"Successfully created figure for key: {fig_key}")
        return fig

    except ValueError as ve: st.error(f"Data Error creating '{fig_key}' ('{title}'): {ve}"); logger.error(f"ValueError creating '{fig_key}': {ve}", exc_info=False); return go.Figure().update_layout(title=title, annotations=[{'text': f"Data Error: Check Logs", 'showarrow': False}])
    except Exception as e: st.error(f"Critical error creating '{fig_key}' ('{title}'): {e}"); logger.critical(f"Critical error creating '{fig_key}': {e}", exc_info=True); return go.Figure().update_layout(title=title, annotations=[{'text': f"Plotting Error: Check Logs", 'showarrow': False}])
# --- End Plotting Function ---


# --- Main UI (Tabs Layout) ---
st.title("📊 Client Analytics Dashboard (Tabs V3.13)") # Version Bump
st.markdown("Select a client and language to view their dashboards.")

col_select1, col_select2 = st.columns([1, 1])
with col_select1:
    available_clients = get_available_clients(RESULTS_DIR)
    if not available_clients: st.error("No client data found."); st.stop()
    selected_client = st.selectbox("Select Client ID:", available_clients, index=0, key="client_sel_rep313") # Changed key
with col_select2:
    lang_options = SUPPORTED_LANGUAGES
    default_lang_index = lang_options.index('ar') if 'ar' in lang_options else 0
    selected_lang = st.selectbox("Select Language:", lang_options, index=default_lang_index, key="lang_sel_rep313") # Changed key

if selected_client:
    st.divider()
    st.header(f"Dashboard for Client: {selected_client} (Lang: {selected_lang})", anchor=False)
    logger.info(f"Displaying dashboard V3.13 - Client: {selected_client}, Lang: {selected_lang}")

    # --- Tabs Layout ---
    tab_titles = {
        "daily": get_translation_streamlit(selected_lang, "frequencies.daily", "Daily"),
        "weekly": get_translation_streamlit(selected_lang, "frequencies.weekly", "Weekly"),
        "monthly": get_translation_streamlit(selected_lang, "frequencies.monthly", "Monthly"),
        "quarterly": get_translation_streamlit(selected_lang, "frequencies.quarterly", "Quarterly")
    }
    tab_daily, tab_weekly, tab_monthly, tab_quarterly = st.tabs([
        f"📅 {tab_titles['daily']}", f"🗓️ {tab_titles['weekly']}",
        f"🈷️ {tab_titles['monthly']}", f"📊 {tab_titles['quarterly']}"
    ])

    frequencies = ["daily", "weekly", "monthly", "quarterly"]
    tabs = [tab_daily, tab_weekly, tab_monthly, tab_quarterly]

    for freq, tab in zip(frequencies, tabs):
        with tab:
            st.subheader(f"{tab_titles[freq]} Reports")
            with st.spinner(f"Loading {freq} data..."):
                data_response = fetch_data_from_api(selected_client, freq, selected_lang)

            if data_response:
                figures_drawn = 0
                figure_keys = sorted(data_response.keys(), key=lambda x: int(x.replace('fig','')) if x.startswith('fig') and x[3:].isdigit() else 99)
                num_cols = 2
                cols = st.columns(num_cols)
                col_index = 0

                for fig_key in figure_keys:
                    chart_data_json = data_response.get(fig_key)
                    if chart_data_json:
                        with cols[col_index % num_cols]:
                            try:
                                plotly_fig = create_plotly_figure_replica(fig_key, chart_data_json, selected_lang)
                                if plotly_fig:
                                    st.plotly_chart(plotly_fig, use_container_width=True)
                                    figures_drawn += 1
                                    col_index += 1
                                elif chart_data_json.get('metadata', {}).get('chart_type') == 'table' and fig_key != 'fig14': # fig14 handled directly by st.dataframe
                                    # Other potential tables could be handled here if needed
                                    logger.info(f"Skipping non-fig14 table rendering for {fig_key}")
                                    # Count it as handled if needed, or skip counting
                                elif fig_key == 'fig16': # Handled directly by st.subheader/st.plotly_chart
                                    figures_drawn += 1 # Count as handled
                                    col_index += 1 # Move to next col
                                else:
                                     # This case covers fig14 (which returns None) and potential errors
                                     logger.warning(f"create_plotly_figure_replica returned None or unhandled type for {fig_key} (Freq: {freq}).")
                            except Exception as e_plot:
                                st.error(f"Failed to display {fig_key}: {e_plot}")
                                logger.error(f"Error rendering {fig_key} (Freq: {freq}): {e_plot}", exc_info=True)
                                col_index += 1 # Ensure index moves even on error
                    else:
                        logger.info(f"No data dict for {fig_key} (Freq: {freq})")

                if figures_drawn == 0:
                    st.info(f"No visualizations available for {freq} frequency for client {selected_client} (lang={selected_lang}).")
                    logger.info(f"No figures drawn for freq '{freq}', client {selected_client}, lang {selected_lang}.")
            # API fetch failed message handled by fetch_data_from_api
    # --- End Tabs Layout ---

else:
    st.info("Please select a Client ID.")

# --- Sidebar ---
st.sidebar.header("App Information")
st.sidebar.info(f"API Base URL: `{API_BASE_URL}`")
st.sidebar.info(f"Client Data Source: (Via API)")
st.sidebar.info(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
st.sidebar.caption(f"Dashboard Version: Tabs V3.13") # Version Bump

logger.info("--- Streamlit dashboard V3.13 script execution finished ---")