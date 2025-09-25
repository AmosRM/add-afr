from highspy import highs
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pulp import *
from datetime import datetime, date
import io
import zipfile
from ETL import etl_long_to_wide

def safe_float_convert(value):
    """
    Safely convert a value to float, handling comma-separated numbers.
    """
    if value is None or value == '':
        return 0.0
    try:
        # If it's already a number, return it as float
        if isinstance(value, (int, float, np.number)):
            return float(value)
        
        # If it's a string, remove commas and convert
        if isinstance(value, str):
            cleaned_value = value.replace(',', '').strip()
            return float(cleaned_value)
        
        # Fallback - try direct conversion
        return float(value)
    except (ValueError, TypeError):
        # If conversion fails, return 0.0 as fallback
        return 0.0

# Page configuration
st.set_page_config(
    page_title="Amos - ENERGYNEST",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Thermal Storage Optimization System - DA & aFRR Markets")

# Only show description and guidance when optimization hasn't started
if not st.session_state.get('run_clicked', False):
    st.markdown("""
    This application optimizes thermal storage operations with flexible objectives:
    - **Cost Minimization:** Traditional approach to minimize total energy costs
    - **Renewable Maximization:** Prioritize maximizing thermal energy from electricity (decarbonization focus)
    - **Day-Ahead Market:** Charging during low electricity prices and using stored energy during high prices.
    - **aFRR Market:** Committing capacity to the aFRR market using either a static or dynamic (ML-driven) bid strategy.
    - Considering grid charges, thermal demand, and market restrictions.
    """)

    # Add helpful guidance
    st.info("👈 **Getting Started:** Use the sidebar to configure the optimization mode, data sources, and system parameters, then run the optimization below.")

@st.cache_data
def precompute_afrr_auction(
    _df_afrr_capacity, afrr_bid_strategy, static_bid_price, _df_afrr_bids,
    _df_peak, holiday_set, static_hochlast_intervals, bid_mw
):
    """
    Performs the aFRR capacity auction pre-computation.
    This function is cached to prevent re-computation when only UI elements are changed.
    """
    if _df_afrr_capacity is None:
        return None, None

    afrr_blocks = _df_afrr_capacity.copy()
    
    # --- SIMPLIFIED PRICE COLUMN LOGIC (FIX) ---
    # This function now ONLY handles capacity data, which should have a 'price' column.
    price_col_candidates = [col for col in afrr_blocks.columns if 'price' in col.lower()]
    if not price_col_candidates:
        st.error("❌ aFRR Capacity Auction data file must contain a column with 'price' in its name.")
        return None, None
    price_col_name = price_col_candidates[0]
    # --- END FIX ---

    # Handle static vs dynamic bid prices
    if afrr_bid_strategy == 'Static Bid':
        afrr_blocks['our_bid'] = static_bid_price
    else:  # Dynamic Bids
        if _df_afrr_bids is None:
            st.error("❌ Dynamic bid strategy selected, but no bid data was provided or loaded.")
            return None, None
        afrr_blocks = pd.merge(afrr_blocks, _df_afrr_bids, left_index=True, right_index=True, how='left')
        if 'Bid Price' not in afrr_blocks.columns:
            st.error("❌ Dynamic bid file must contain a 'Bid Price' column.")
            return None, None
        afrr_blocks['Bid Price'] = afrr_blocks['Bid Price'].ffill().bfill()
        if afrr_blocks['Bid Price'].isna().any():
            afrr_blocks['Bid Price'] = afrr_blocks['Bid Price'].fillna(0)
        afrr_blocks.rename(columns={'Bid Price': 'our_bid'}, inplace=True)

    afrr_blocks["won_price"] = afrr_blocks[price_col_name] >= afrr_blocks['our_bid']

    hlf_time_cols_for_afrr = [col for col in _df_peak.columns if col != 'date'] if _df_peak is not None else []
    is_15min_data = len(afrr_blocks) > 30000  # Heuristic for 15-min vs 4h data
    afrr_blocks["is_hochlast"] = False

    if is_15min_data:
        # For 15-minute data, we need to check each 4-hour block as a unit
        afrr_blocks['block_id'] = afrr_blocks.index.floor('4H')
        
        # First, determine HLF status for each 15-min interval
        hlf_status = {}
        for idx, row in afrr_blocks.iterrows():
            interval_in_day = (idx.hour * 4) + (idx.minute // 15)
            is_holiday = idx.strftime('%Y-%m-%d') in holiday_set
            is_static_hochlast = interval_in_day in static_hochlast_intervals
            is_dynamic_hochlast = False
            
            if _df_peak is not None and hlf_time_cols_for_afrr:
                date_str = idx.strftime('%Y-%m-%d')
                day_peak_data = _df_peak[_df_peak['date'] == date_str]
                if not day_peak_data.empty and interval_in_day < len(hlf_time_cols_for_afrr):
                    col_name = hlf_time_cols_for_afrr[interval_in_day]
                    if col_name in day_peak_data.columns:
                        is_dynamic_hochlast = bool(day_peak_data[col_name].iloc[0])
            
            hlf_status[idx] = (is_static_hochlast or is_dynamic_hochlast) and not is_holiday
        
        # Now check if ANY interval in each 4-hour block has HLF
        block_has_hlf = {}
        for block_id in afrr_blocks['block_id'].unique():
            block_intervals = afrr_blocks[afrr_blocks['block_id'] == block_id].index
            block_has_hlf[block_id] = any(hlf_status.get(idx, False) for idx in block_intervals)
        
        # Apply the block-level HLF status to all intervals in each block
        for idx, row in afrr_blocks.iterrows():
            afrr_blocks.loc[idx, "is_hochlast"] = block_has_hlf[row['block_id']]
        
        afrr_blocks["won"] = afrr_blocks["won_price"] & (~afrr_blocks["is_hochlast"])
        
        block_won = afrr_blocks.groupby('block_id')['won'].all()
        block_bid_price = afrr_blocks.groupby('block_id')['our_bid'].first()
        block_revenue = block_won * block_bid_price * bid_mw * 4

        afrr_15min_mask = afrr_blocks["won"].copy()
        if afrr_15min_mask.index.tz is not None:
            afrr_15min_mask.index = afrr_15min_mask.index.tz_localize(None)

        afrr_won_blocks = pd.DataFrame({'won': block_won, 'cap_payment': block_revenue})
        afrr_won_blocks = afrr_won_blocks[afrr_won_blocks['won']]

    else:  # 4-hour blocks
        for idx, row in afrr_blocks.iterrows():
            block_start_hour = idx.hour
            block_intervals = [
                ((block_start_hour + h_offset) * 4 + (m_offset // 15)) % 96
                for h_offset in range(4) 
                for m_offset in [0, 15, 30, 45]
            ]
            is_holiday = idx.strftime('%Y-%m-%d') in holiday_set
            has_static_hochlast = any(interval in static_hochlast_intervals for interval in block_intervals)
            has_dynamic_hochlast = False
            if _df_peak is not None and hlf_time_cols_for_afrr:
                date_str = idx.strftime('%Y-%m-%d')
                day_peak_data = _df_peak[_df_peak['date'] == date_str]
                if not day_peak_data.empty:
                    for interval_in_day in block_intervals:
                        if interval_in_day < len(hlf_time_cols_for_afrr):
                            col_name = hlf_time_cols_for_afrr[interval_in_day]
                            if col_name in day_peak_data.columns and bool(day_peak_data[col_name].iloc[0]):
                                has_dynamic_hochlast = True
                                break
            afrr_blocks.loc[idx, "is_hochlast"] = (has_static_hochlast or has_dynamic_hochlast) and not is_holiday

        afrr_blocks["won"] = afrr_blocks["won_price"] & (~afrr_blocks["is_hochlast"])
        afrr_blocks["cap_payment"] = afrr_blocks["won"] * afrr_blocks['our_bid'] * bid_mw * 4
        afrr_won_blocks = afrr_blocks[afrr_blocks["won"]]

        afrr_15min_mask_list = []
        for idx, row in afrr_blocks.iterrows():
            afrr_15min_mask_list.extend([row['won']] * 16)

        full_range_index = pd.date_range(
            start=afrr_blocks.index.min().date(), 
            end=afrr_blocks.index.max().date() + pd.Timedelta(days=1), 
            freq="15T", 
            tz=afrr_blocks.index.tz
        )[:-1]
        
        if len(afrr_15min_mask_list) == len(full_range_index):
            afrr_15min_mask = pd.Series(afrr_15min_mask_list, index=full_range_index)
            if afrr_15min_mask.index.tz is not None:
                afrr_15min_mask.index = afrr_15min_mask.index.tz_localize(None)
        else:
            afrr_15min_mask = pd.Series()

    return afrr_won_blocks, afrr_15min_mask

@st.cache_data
def precompute_afrr_up_auction(
    _df_afrr_up_capacity, static_bid_price, _df_peak, holiday_set, static_hochlast_intervals, bid_mw
):
    """
    Performs the aFRR Up capacity auction pre-computation.
    This function is cached to prevent re-computation when only UI elements are changed.
    """
    if _df_afrr_up_capacity is None:
        return None, None

    afrr_up_blocks = _df_afrr_up_capacity.copy()
    
    # --- SIMPLIFIED PRICE COLUMN LOGIC (FIX) ---
    # This function now ONLY handles capacity data, which should have a 'price' column.
    price_col_candidates = [col for col in afrr_up_blocks.columns if 'price' in col.lower()]
    if not price_col_candidates:
        st.error("❌ aFRR Up Capacity Auction data file must contain a column with 'price' in its name.")
        return None, None
    price_col_name = price_col_candidates[0]
    # --- END FIX ---

    # For aFRR Up, we use static bid strategy only for now
    afrr_up_blocks['our_bid'] = static_bid_price

    afrr_up_blocks["won_price"] = afrr_up_blocks[price_col_name] >= afrr_up_blocks['our_bid']

    hlf_time_cols_for_afrr = [col for col in _df_peak.columns if col != 'date'] if _df_peak is not None else []
    is_15min_data = len(afrr_up_blocks) > 30000  # Heuristic for 15-min vs 4h data
    afrr_up_blocks["is_hochlast"] = False

    if is_15min_data:
        # For 15-minute data, we need to check each 4-hour block as a unit
        afrr_up_blocks['block_id'] = afrr_up_blocks.index.floor('4H')
        
        # First, determine HLF status for each 15-min interval
        hlf_status = {}
        for idx, row in afrr_up_blocks.iterrows():
            interval_in_day = (idx.hour * 4) + (idx.minute // 15)
            is_holiday = idx.strftime('%Y-%m-%d') in holiday_set
            is_static_hochlast = interval_in_day in static_hochlast_intervals
            is_dynamic_hochlast = False
            
            if _df_peak is not None and hlf_time_cols_for_afrr:
                date_str = idx.strftime('%Y-%m-%d')
                day_peak_data = _df_peak[_df_peak['date'] == date_str]
                if not day_peak_data.empty and interval_in_day < len(hlf_time_cols_for_afrr):
                    col_name = hlf_time_cols_for_afrr[interval_in_day]
                    if col_name in day_peak_data.columns:
                        is_dynamic_hochlast = bool(day_peak_data[col_name].iloc[0])
            
            hlf_status[idx] = (is_static_hochlast or is_dynamic_hochlast) and not is_holiday
        
        # Now check if ANY interval in each 4-hour block has HLF
        block_has_hlf = {}
        for block_id in afrr_up_blocks['block_id'].unique():
            block_intervals = afrr_up_blocks[afrr_up_blocks['block_id'] == block_id].index
            block_has_hlf[block_id] = any(hlf_status.get(idx, False) for idx in block_intervals)
        
        # Apply the block-level HLF status to all intervals in each block
        for idx, row in afrr_up_blocks.iterrows():
            afrr_up_blocks.loc[idx, "is_hochlast"] = block_has_hlf[row['block_id']]
        
        afrr_up_blocks["won"] = afrr_up_blocks["won_price"] & (~afrr_up_blocks["is_hochlast"])
        
        block_won = afrr_up_blocks.groupby('block_id')['won'].all()
        block_bid_price = afrr_up_blocks.groupby('block_id')['our_bid'].first()
        block_revenue = block_won * block_bid_price * bid_mw * 4

        afrr_up_15min_mask = afrr_up_blocks["won"].copy()
        if afrr_up_15min_mask.index.tz is not None:
            afrr_up_15min_mask.index = afrr_up_15min_mask.index.tz_localize(None)

        afrr_up_won_blocks = pd.DataFrame({'won': block_won, 'cap_payment': block_revenue})
        afrr_up_won_blocks = afrr_up_won_blocks[afrr_up_won_blocks['won']]

    else:  # 4-hour blocks
        for idx, row in afrr_up_blocks.iterrows():
            block_start_hour = idx.hour
            block_intervals = [
                ((block_start_hour + h_offset) * 4 + (m_offset // 15)) % 96
                for h_offset in range(4) 
                for m_offset in [0, 15, 30, 45]
            ]
            is_holiday = idx.strftime('%Y-%m-%d') in holiday_set
            has_static_hochlast = any(interval in static_hochlast_intervals for interval in block_intervals)
            has_dynamic_hochlast = False
            if _df_peak is not None and hlf_time_cols_for_afrr:
                date_str = idx.strftime('%Y-%m-%d')
                day_peak_data = _df_peak[_df_peak['date'] == date_str]
                if not day_peak_data.empty:
                    for interval_in_day in block_intervals:
                        if interval_in_day < len(hlf_time_cols_for_afrr):
                            col_name = hlf_time_cols_for_afrr[interval_in_day]
                            if col_name in day_peak_data.columns and bool(day_peak_data[col_name].iloc[0]):
                                has_dynamic_hochlast = True
                                break
            afrr_up_blocks.loc[idx, "is_hochlast"] = (has_static_hochlast or has_dynamic_hochlast) and not is_holiday

        afrr_up_blocks["won"] = afrr_up_blocks["won_price"] & (~afrr_up_blocks["is_hochlast"])
        afrr_up_blocks["cap_payment"] = afrr_up_blocks["won"] * afrr_up_blocks['our_bid'] * bid_mw * 4
        afrr_up_won_blocks = afrr_up_blocks[afrr_up_blocks["won"]]

        afrr_up_15min_mask_list = []
        for idx, row in afrr_up_blocks.iterrows():
            afrr_up_15min_mask_list.extend([row['won']] * 16)

        full_range_index = pd.date_range(
            start=afrr_up_blocks.index.min().date(), 
            end=afrr_up_blocks.index.max().date() + pd.Timedelta(days=1), 
            freq="15T", 
            tz=afrr_up_blocks.index.tz
        )[:-1]
        
        if len(afrr_up_15min_mask_list) == len(full_range_index):
            afrr_up_15min_mask = pd.Series(afrr_up_15min_mask_list, index=full_range_index)
            if afrr_up_15min_mask.index.tz is not None:
                afrr_up_15min_mask.index = afrr_up_15min_mask.index.tz_localize(None)
        else:
            afrr_up_15min_mask = pd.Series()

    return afrr_up_won_blocks, afrr_up_15min_mask

@st.cache_data
def extract_afrr_clearing_prices(_df_afrr_energy):
    """
    Extract clearing prices for aFRR energy market evaluation.
    Returns a Series indexed by datetime with clearing prices.
    """
    if _df_afrr_energy is None:
        return None
    
    afrr_data = _df_afrr_energy.copy()
    
    if 'Revenue' in afrr_data.columns and 'Date' in afrr_data.columns:
        afrr_data['datetime'] = pd.to_datetime(afrr_data['Date'])
        afrr_data = afrr_data.set_index('datetime')
        clearing_prices = afrr_data['Revenue'].apply(safe_float_convert)
    else:
        price_col_candidates = [col for col in afrr_data.columns if 'price' in col.lower()]
        if not price_col_candidates:
            st.error("❌ Could not find a 'Revenue' or 'price' column in the aFRR Energy data.")
            return None
        price_col_name = price_col_candidates[0]
        
        if len(afrr_data) > 30000:
            clearing_prices = afrr_data[price_col_name]
        else:
            clearing_prices_list = []
            for idx, row in afrr_data.iterrows():
                clearing_prices_list.extend([row[price_col_name]] * 16)
            full_range_index = pd.date_range(
                start=afrr_data.index.min().date(),
                end=afrr_data.index.max().date() + pd.Timedelta(days=1),
                freq="15T",
                tz=afrr_data.index.tz
            )[:-1]
            if len(clearing_prices_list) == len(full_range_index):
                clearing_prices = pd.Series(clearing_prices_list, index=full_range_index)
            else:
                return None
    
    if clearing_prices.index.tz is not None:
        clearing_prices.index = clearing_prices.index.tz_localize(None)
    
    return clearing_prices

@st.cache_data
def extract_afrr_activation_profile(_df_afrr_energy):
    """
    Extract activation percentages for aFRR energy market evaluation.
    Returns a Series indexed by datetime with activation percentages.
    """
    if _df_afrr_energy is None:
        return None
    
    afrr_data = _df_afrr_energy.copy()
    
    if 'Activation' in afrr_data.columns and 'Date' in afrr_data.columns:
        afrr_data['datetime'] = pd.to_datetime(afrr_data['Date'])
        afrr_data = afrr_data.set_index('datetime')
        activation_profile = afrr_data['Activation'] * 100
        if activation_profile.index.tz is not None:
            activation_profile.index = activation_profile.index.tz_localize(None)
        return activation_profile
    
    return None

@st.cache_data
def extract_afrr_up_clearing_prices(_df_afrr_up_energy):
    """
    Extract clearing prices for aFRR Up energy market evaluation.
    Returns a Series indexed by datetime with clearing prices.
    """
    if _df_afrr_up_energy is None:
        return None
    
    # For now, use the same extraction logic as Down market
    # In practice, Up and Down may have different price structures
    return extract_afrr_clearing_prices(_df_afrr_up_energy)

@st.cache_data
def extract_afrr_up_activation_profile(_df_afrr_up_energy):
    """
    Extract activation percentages for aFRR Up energy market evaluation.
    Returns a Series indexed by datetime with activation percentages.
    """
    if _df_afrr_up_energy is None:
        return None
    
    # For now, use the same extraction logic as Down market
    # In practice, Up and Down may have different activation patterns
    return extract_afrr_activation_profile(_df_afrr_up_energy)

# Sidebar for parameters
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio(
    "Select Price Data Source",
    ("Use Built-in EPEX 2024 Data", "Upload File", "Fetch from EnAppSys API"),
    index=0,
    help="Choose the source for Day-Ahead electricity prices."
)

uploaded_file = None
api_config = None
use_builtin_data = False

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload electricity price data (CSV)", type=['csv'])

    transform_data = st.sidebar.checkbox(
        "Transform price data (long to wide format)",
        value=True,
        help="Check this if your price data has one row per timestamp. Uncheck if your data is already wide (date, 00:00, 00:15...)."
    )
elif data_source == "Fetch from EnAppSys API":
    st.sidebar.subheader("🔌 EnAppSys API Configuration")
    api_type = st.sidebar.selectbox("API Type", ("chart", "bulk"), help="Chart API for specific chart codes, Bulk API for data types")
    with st.sidebar.expander("🔐 API Credentials", expanded=False):
        api_username = st.text_input("Username", type="default", help="Your EnAppSys username")
        api_password = st.text_input("Password", type="password", help="Your EnAppSys password")

    if api_type == "chart":
        chart_code = st.sidebar.text_input("Chart Code", value="de/elec/pricing/daprices", help="e.g., 'de/elec/pricing/daprices' for German day-ahead prices")
    else:
        bulk_type = st.sidebar.selectbox("Data Type", ("NL_SOLAR_FORECAST", "DE_WIND_FORECAST", "FR_DEMAND_FORECAST", "GB_DEMAND_FORECAST"), help="Select the type of bulk data to fetch")
        entities = st.sidebar.text_input("Entities", value="ALL", help="Comma-separated list of entities or 'ALL' for all available")
        chart_code = None

    col1, col2 = st.sidebar.columns(2)
    with col1:
        api_start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date(), help="Start date for data fetch")
    with col2:
        api_end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-31").date(), help="End date for data fetch")

    api_resolution = st.sidebar.selectbox("Resolution", ("qh", "hourly", "daily", "weekly", "monthly"), index=0, help="Data resolution/frequency (qh = quarter hourly)")
    api_timezone = st.sidebar.selectbox("Timezone", ("CET", "WET", "EET", "UTC"), index=0, help="Timezone for the data")
    api_currency = st.sidebar.selectbox("Currency", ("EUR", "GBP", "USD"), index=0, help="Currency for price data")

    if not api_username or not api_password:
        st.sidebar.warning("⚠️ Please enter your API credentials to proceed")
        api_config = None
    else:
        api_config = {
            'api_type': api_type, 'credentials': {'user': api_username, 'pass': api_password},
            'start_date': api_start_date.strftime('%d/%m/%Y 00:00'), 'end_date': api_end_date.strftime('%d/%m/%Y 23:59'),
            'resolution': api_resolution, 'timezone': api_timezone, 'currency': api_currency
        }
        if api_type == "chart":
            api_config['chart_code'] = chart_code
        else:
            api_config['bulk_type'] = bulk_type
            api_config['entities'] = entities
    transform_data = True
else:
    use_builtin_data = True
    st.sidebar.info("📊 Using built-in EPEX 2024 price data")
    st.sidebar.markdown("**Dataset:** `idprices-epex2024.csv`")
    transform_data = True

st.sidebar.header("⚖️ Optimization Mode")
optimization_mode = st.sidebar.radio(
    "Select Market Participation",
    ("DA Market Only", "DA + aFRR Market"),
    help="Choose 'DA Only' for standard cost minimization or 'DA + aFRR' to include ancillary service revenue."
)

afrr_capacity_file = None
afrr_up_capacity_file = None
afrr_energy_file = None
afrr_dynamic_bids_file = None
use_builtin_afrr = False
afrr_bid_price = 36.0
afrr_bid_mw = 2.0
afrr_up_bid_price = 50.0
afrr_up_bid_mw = 2.0
afrr_up_energy_bid_base = 100.0
afrr_up_max_discount = 20.0
afrr_energy_bid_base = 36.0

# Individual checkboxes for aFRR components - only show when aFRR is selected
enable_afrr_capacity = False
enable_afrr_energy = False
enable_afrr_up_capacity = False

if optimization_mode == "DA + aFRR Market":
    st.sidebar.subheader("⚡ aFRR Market Components")

    # Individual checkboxes for aFRR components
    enable_afrr_capacity = st.sidebar.checkbox(
        "Enable aFRR Capacity Market",
        value=True,
        help="Include aFRR capacity auction bidding for ancillary service revenue"
    )

    enable_afrr_energy = st.sidebar.checkbox(
        "Enable aFRR Energy Market",
        value=True,
        help="Include aFRR energy market participation with SOC-based bidding"
    )

    enable_afrr_up_capacity = st.sidebar.checkbox(
        "Enable aFRR Up Capacity Market",
        value=False,
        help="Include aFRR Up capacity auction bidding for upward regulation reserve"
    )

    # Show aFRR data section if any component is enabled
    if enable_afrr_capacity or enable_afrr_energy or enable_afrr_up_capacity:
        st.sidebar.subheader("⚡ aFRR Market Data")
        if data_source == "Use Built-in EPEX 2024 Data":
            st.sidebar.info("📊 Using built-in aFRR 2024 data")
            if enable_afrr_capacity:
                st.sidebar.markdown("**Capacity Auction (Down):** `aFRRprices.csv`")
            if enable_afrr_up_capacity:
                st.sidebar.markdown("**Capacity Auction (Up):** `aFRR_2024Predictions.csv`")
            if enable_afrr_energy:
                st.sidebar.markdown("**Energy Market:** `aFRRenergylight.csv`")
            use_builtin_afrr = True
        else:
            if enable_afrr_capacity:
                afrr_capacity_file = st.sidebar.file_uploader("Upload aFRR Down Capacity Auction Data (CSV)", type=['csv'], help="Upload a CSV with 4-hour capacity clearing prices (e.g., aFRRprices.csv).")
            if enable_afrr_up_capacity:
                afrr_up_capacity_file = st.sidebar.file_uploader("Upload aFRR Up Capacity Auction Data (CSV)", type=['csv'], help="Upload a CSV with 4-hour Up capacity clearing prices (e.g., aFRR_2024Predictions.csv).")
            if enable_afrr_energy:
                afrr_energy_file = st.sidebar.file_uploader("Upload aFRR Energy Market Data (CSV)", type=['csv'], help="Upload a CSV with 15-min energy revenue and activation data (e.g., aFRRenergylight.csv).")
            use_builtin_afrr = False

with st.sidebar.expander("⚙️ Advanced System Parameters"):
    st.markdown("Define the physical properties of the thermal storage asset.")
    Δt = st.number_input("Time Interval (hours)", value=0.25, min_value=0.1, max_value=1.0, step=0.05)
    Pmax_el = st.number_input("Max Electrical Power (MW)", value=2.0, min_value=0.1, max_value=200.0, step=0.1)
    Pmax_th = st.number_input("Max Thermal Power (MW)", value=2.0, min_value=0.1, max_value=200.0, step=0.1)
    Smax = st.number_input("Max Storage Capacity (MWh)", value=8.0, min_value=1.0, max_value=1000.0, step=0.5)
    SOC_min = st.number_input("Min Storage Level (MWh)", value=0.0, min_value=0.0, max_value=5.0, step=0.5)
    η = st.number_input("Charging Efficiency", value=0.95, min_value=0.7, max_value=1.0, step=0.05)
    self_discharge_daily = st.number_input("Self-Discharge Rate (% per day)", value=3.0, min_value=0.0, max_value=20.0, step=0.1, help="Daily percentage of stored energy lost due to standing thermal losses.")
    boiler_efficiency_pct = st.number_input("Gas Boiler Efficiency (%)", value=90.0, min_value=50.0, max_value=100.0, step=1.0, help="Efficiency of the gas boiler in converting gas fuel to thermal energy.")
    boiler_efficiency = boiler_efficiency_pct / 100.0

    st.markdown("---")
    enable_power_curve = st.checkbox("Enable Power-Capacity Curve", value=True, help="Model charging/discharging power limits based on State of Charge.")
    if enable_power_curve:
        charge_taper_soc_pct = 75
        charge_power_at_full_pct = 30
        discharge_taper_soc_pct = 25
        discharge_power_at_empty_pct = 30

        st.markdown("#### Charge Curve")
        st.write(f"Taper starts at: **{charge_taper_soc_pct}%** SOC")
        st.write(f"Power at 100% SOC: **{charge_power_at_full_pct}%**")
        st.markdown("#### Discharge Curve")
        st.write(f"Taper ends at: **{discharge_taper_soc_pct}%** SOC")
        st.write(f"Power at 0% SOC: **{discharge_power_at_empty_pct}%**")

with st.sidebar.expander("⚖️ Economic & Bidding Parameters"):
    optimization_objective = st.radio(
        "**Optimization Objective**",
        ("Minimize Cost", "Maximize Thermal from Electricity")
    )

    price_cap_for_max_thermal = 150.0
    if optimization_objective == "Maximize Thermal from Electricity":
        price_cap_for_max_thermal = st.number_input(
            "Max Charging Price (€/MWh)", 
            value=150.0, min_value=0.0, step=5.0,
            help="When maximizing thermal from electricity, avoid charging if DA price exceeds this value."
        )
    
    C_grid = st.number_input("Grid Charges (€/MWh)", value=30.0, min_value=0.0, max_value=100.0, step=1.0)

    C_gas = st.number_input("Gas Price (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)
    terminal_value = st.number_input("Terminal Value (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0, help="Estimated value of energy remaining in storage at the end of the optimization period.")
    st.markdown("---")
    st.markdown("**Market Capacity Allocation**")
    da_max_capacity = st.number_input(
        "Max DA Market Capacity (MW)", 
        value=2.0,  # Default to full capacity
        min_value=0.0, 
        max_value=Pmax_el, 
        step=0.1,
        help="Maximum power to commit to Day-Ahead market. Remaining capacity (up to system max) will be reserved for aFRR energy market participation."
    )
    da_max_soc_pct = st.number_input(
        "Max DA Market SOC (%)",
        value=100.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="Maximum State of Charge the Day-Ahead market can plan for. The remaining capacity is reserved for aFRR energy."
    )
    # aFRR parameters - only show when aFRR is selected
    if optimization_mode == "DA + aFRR Market":
        # aFRR Capacity Market parameters
        if enable_afrr_capacity:
            st.markdown("---")
            st.markdown("**aFRR Capacity Market**")
            afrr_bid_mw = st.number_input("Our aFRR Bid Size (MW)", value=2.0, min_value=0.1, max_value=100.0, step=0.1, help="The amount of power capacity to bid. Must be <= Max Electrical Power.")

            afrr_bid_strategy = st.radio(
                "aFRR Bid Strategy",
                ("Static Bid", "Dynamic Bids (from CSV)"),
                help="Choose a single fixed bid price, or upload a CSV with time-varying bid prices."
            )
            if afrr_bid_strategy == "Static Bid":
                afrr_bid_price = st.number_input("Our aFRR Bid Price (€/MW)", value=36.0, min_value=0.0, step=1.0, help="Our fixed bid price. We win any block where the clearing price is >= this value.")
            else: # Dynamic Bids
                afrr_dynamic_bids_file = st.file_uploader(
                    "Upload Your Dynamic Bid Prices (CSV)",
                    type=['csv'],
                    help="Upload a long-format CSV with 'Date (CET)' and 'Bid Price' columns."
                )

        # aFRR Up Capacity Market parameters
        if enable_afrr_up_capacity:
            st.markdown("---")
            st.markdown("**aFRR Up Capacity Market**")
            afrr_up_bid_mw = st.number_input("Our aFRR Up Bid Size (MW)", value=2.0, min_value=0.1, max_value=100.0, step=0.1, help="The amount of upward regulation power capacity to bid. Must be <= Max Electrical Power.")
            afrr_up_bid_price = st.number_input("Our aFRR Up Bid Price (€/MW)", value=50.0, min_value=0.0, step=1.0, help="Our fixed bid price for upward regulation. We win any block where the clearing price is >= this value.")
            
            st.markdown("**aFRR Up Energy Market**")
            afrr_up_energy_bid_base = st.number_input(
                "aFRR Up Energy Base Bid (€/MWh)",
                value=100.0,
                min_value=0.0,
                step=1.0,
                help="Base bid price for aFRR Up energy. Discount applied based on SOC (higher SOC = lower bid = more likely to win)."
            )
            afrr_up_max_discount = st.number_input(
                "Max aFRR Up Discount (€/MWh)",
                value=20.0,
                min_value=0.0,
                step=1.0,
                help="Maximum discount applied when SOC is at 100%. Discount scales linearly with SOC percentage."
            )

        # aFRR Energy Market parameters
        if enable_afrr_energy:
            st.markdown("---")
            st.markdown("**aFRR Energy Market**")
            afrr_energy_bid_base = st.number_input(
                "aFRR Energy Base Bid (€/MWh)",
                value=36.0,
                min_value=(-50.0),
                step=1.0,
                help="Base bid price for aFRR energy. Premium will be applied based on SOC."
            )
            st.markdown("**SOC-based Premium Schedule**")
            st.caption("Premium increases with SOC to avoid winning when battery is full")
            
            # Create editable premium table
            col1, col2 = st.columns([6, 1])
            with col1:
                # Initialize default premium table if not in session state
                if 'soc_premium_table' not in st.session_state:
                    st.session_state.soc_premium_table = {
                        0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 0.0,
                        4.0: 0.0, 7.5: 30.0, 8.0: 10000.0
                    }
                
                # Create editable dataframe
                premium_df = pd.DataFrame(
                    list(st.session_state.soc_premium_table.items()),
                    columns=['SOC (MWh)', 'Premium (€/MWh)']
                )
                
                # Use st.data_editor for editable table
                edited_premium_df = st.data_editor(
                    premium_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    key="premium_table_editor"
                )
                
                # Update session state when table is edited
                if not edited_premium_df.equals(premium_df):
                    new_premium_table = {}
                    for _, row in edited_premium_df.iterrows():
                        soc = float(row['SOC (MWh)'])
                        premium = float(row['Premium (€/MWh)'])
                        new_premium_table[soc] = premium
                    st.session_state.soc_premium_table = new_premium_table
                
                # Use the current premium table from session state
                soc_premium_table = st.session_state.soc_premium_table
            
            with col2:
                if st.button("Reset"):
                    st.session_state.soc_premium_table = {
                        0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 10.0,
                        4.0: 40.0, 7.5: 300.0, 8.0: 10000.0
                    }
                    st.rerun()
                
    else:
        # Default premium table when aFRR energy is not enabled
        soc_premium_table = {
            0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 10.0,
            4.0: 40.0, 7.5: 300.0, 8.0: 10000.0
        }

with st.sidebar.expander("🔥 Thermal Demand Configuration"):
    demand_option = st.radio("Select Demand Source", ('Constant Demand', 'Upload Demand Profile'), help="Choose a fixed, constant demand or upload a CSV file with a time-varying demand profile.")
    D_th = None
    demand_file = None
    if demand_option == 'Constant Demand':
        D_th = st.number_input("Thermal Demand (MW)", value=1.0, min_value=0.0, max_value=20.0, step=0.1)
    else:
        demand_file = st.file_uploader("Upload customer demand data (CSV)", type=['csv'])

with st.sidebar.expander("📈 Peak Period Restrictions"):
    peak_period_option = st.radio(
        "Define Peak Periods",
        ("Use Built-in Example Data", "Manual Selection", "Upload CSV File"),
        index=0,
        help="Choose how to define peak restriction periods. Built-in example uses Example_Peak Restriktions.csv, manual selection uses fixed checkboxes, or upload your own CSV for date-specific schedules."
    )
    hochlast_intervals_static = set()
    peak_period_file = None
    use_builtin_peak = False

    if peak_period_option == "Use Built-in Example Data":
        use_builtin_peak = True
        st.sidebar.info("📊 Using built-in HLF example data")
        st.sidebar.markdown("**Dataset:** `Example_Peak Restriktions.csv`")
    elif peak_period_option == "Manual Selection":
        hochlast_morning = st.checkbox("Morning Peak (8-10 AM)", value=True)
        hochlast_evening = st.checkbox("Evening Peak (6-8 PM)", value=True)
        if hochlast_morning:
            hochlast_intervals_static.update(range(32, 40))
        if hochlast_evening:
            hochlast_intervals_static.update(range(72, 80))
    else:
        peak_period_file = st.file_uploader("Upload Peak Period CSV", type=['csv'], help="Upload a CSV with 'Date (CET)' and 'Is HLF' (1 for restricted, 0 for not).")

with st.sidebar.expander("🗓️ Holiday Configuration"):
    default_holidays = ['2024-01-01', '2024-03-29', '2024-04-01', '2024-05-01', '2024-05-09', '2024-05-10', '2024-05-20', '2024-05-30', '2024-05-31', '2024-10-01', '2024-10-04', '2024-11-01', '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30', '2024-12-31']
    holiday_input = st.text_area("Holiday Dates (one per line, YYYY-MM-DD)", value='\n'.join(default_holidays), height=150)
    holiday_dates = [date.strip() for date in holiday_input.split('\n') if date.strip()]
    holiday_set = set(holiday_dates)

with st.sidebar.expander("💾 Cache Management"):
    cached_items = []
    if 'cached_df_price' in st.session_state: cached_items.append("✅ Price Data")
    if 'cached_df_demand' in st.session_state: cached_items.append("✅ Demand Data")
    if 'cached_df_peak' in st.session_state: cached_items.append("✅ Peak Restriction Data")
    if 'cached_df_afrr_capacity' in st.session_state: cached_items.append("✅ aFRR Capacity Data")
    if 'cached_df_afrr_energy' in st.session_state: cached_items.append("✅ aFRR Energy Data")
    if 'cached_df_afrr_bids' in st.session_state: cached_items.append("✅ aFRR Dynamic Bids")

    if cached_items:
        st.write("**Cached Data:**"); [st.write(item) for item in cached_items]
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Cache", help="Clear all cached data and force refresh"):
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('cached_')]
                for key in keys_to_remove: del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("🔄 Refresh Data", help="Clear cache and reload current configuration"):
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('cached_')]
                for key in keys_to_remove: del st.session_state[key]
                st.rerun()
    else:
        st.write("No cached data"); st.caption("Data will be automatically cached after first fetch/upload")

# --- MAIN LOGIC ---
if uploaded_file is not None or api_config is not None or use_builtin_data:
    config_key = f"file_{uploaded_file.name}_{hash(uploaded_file.getvalue())}" if uploaded_file else (f"api_{hash(str(sorted(api_config.items())))}" if api_config else "builtin_epex2024")

    if 'cached_config_key' in st.session_state and 'cached_df_price' in st.session_state and st.session_state['cached_config_key'] == config_key:
        df_price = st.session_state['cached_df_price']
        if not st.session_state.get('run_clicked', False):
            st.success("✅ Using cached price data from previous fetch!")
    else:
        # ... (rest of price data loading logic remains the same)
        df_price = None
        if transform_data:
            if not st.session_state.get('run_clicked', False):
                st.info("New data source detected. Running ETL transformation...")
            with st.spinner("Transforming data from long to wide format..."):
                try:
                    if uploaded_file is not None:
                        df_price = etl_long_to_wide(input_source=uploaded_file, datetime_column_name='Date (CET)', value_column_name='Day Ahead Price')
                    elif api_config is not None:
                        df_price = etl_long_to_wide(use_api=True, api_config=api_config)
                    elif use_builtin_data:
                        df_price = etl_long_to_wide(input_source="idprices-epex2024.csv", datetime_column_name='Date (CET)', value_column_name='Day Ahead Price')
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Price ETL transformation successful!")
                    if df_price is not None:
                        st.session_state['cached_df_price'] = df_price.copy()
                        st.session_state['cached_config_key'] = config_key
                except Exception as e:
                    st.error(f"❌ ETL process failed: {e}"); st.stop()
        else:
            try:
                if uploaded_file is not None: df_price = pd.read_csv(uploaded_file)
                elif use_builtin_data: df_price = pd.read_csv("idprices-epex2024.csv")
                else: st.error("❌ API data must be transformed."); st.stop()
                if df_price is not None:
                    st.session_state['cached_df_price'] = df_price.copy()
                    st.session_state['cached_config_key'] = config_key
            except Exception as e: st.error(f"❌ Failed to load the price CSV file: {e}"); st.stop()


    # ... (Demand and Peak data loading logic remains the same)
    df_demand = None
    if demand_option == 'Upload Demand Profile':
        if demand_file is None: st.warning("Please upload a Customer Demand CSV file."); st.stop()
        demand_config_key = f"demand_{demand_file.name}_{hash(demand_file.getvalue())}"
        if 'cached_demand_config_key' in st.session_state and 'cached_df_demand' in st.session_state and st.session_state['cached_demand_config_key'] == demand_config_key:
            df_demand = st.session_state['cached_df_demand']
            if not st.session_state.get('run_clicked', False):
                st.success("✅ Using cached demand data!")
        else:
            with st.spinner("Transforming demand data..."):
                try:
                    df_demand = etl_long_to_wide(input_source=demand_file, datetime_column_name='Date (CET)', value_column_name='MW-th')
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Demand ETL transformation successful!")
                    st.session_state['cached_df_demand'] = df_demand.copy()
                    st.session_state['cached_demand_config_key'] = demand_config_key
                except Exception as e: st.error(f"❌ Demand file processing failed: {e}"); st.stop()

    df_peak = None
    if peak_period_option == 'Use Built-in Example Data':
        peak_config_key = "peak_builtin_example"
        if 'cached_peak_config_key' in st.session_state and 'cached_df_peak' in st.session_state and st.session_state['cached_peak_config_key'] == peak_config_key:
            df_peak = st.session_state['cached_df_peak']
            if not st.session_state.get('run_clicked', False):
                st.success("✅ Using cached built-in peak restriction data!")
        else:
            with st.spinner("Loading built-in peak restriction example data..."):
                try:
                    df_peak = etl_long_to_wide(input_source="Example_Peak Restriktions.csv", datetime_column_name='Date (CET)', value_column_name='Is HLF')
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Built-in peak restriction data loaded successfully!")
                    st.session_state['cached_df_peak'] = df_peak.copy()
                    st.session_state['cached_peak_config_key'] = peak_config_key
                except Exception as e:
                    st.error(f"❌ Failed to load built-in peak restriction file: {e}")
                    st.stop()
    elif peak_period_option == 'Upload CSV File':
        if peak_period_file is None: st.warning("Please upload a Peak Period CSV file."); st.stop()
        peak_config_key = f"peak_{peak_period_file.name}_{hash(peak_period_file.getvalue())}"
        if 'cached_peak_config_key' in st.session_state and 'cached_df_peak' in st.session_state and st.session_state['cached_peak_config_key'] == peak_config_key:
            df_peak = st.session_state['cached_df_peak']
            if not st.session_state.get('run_clicked', False):
                st.success("✅ Using cached peak restriction data!")
        else:
            with st.spinner("Analyzing peak restriction file..."):
                try:
                    peak_period_file.seek(0)
                    lines = peak_period_file.read().decode('utf-8-sig').splitlines()
                    header_row_index = -1
                    for i, line in enumerate(lines):
                        if 'Date (CET)' in line and 'Is HLF' in line: header_row_index = i; break
                    if header_row_index == -1: st.error("❌ Invalid Peak Restriction File: Could not find 'Date (CET)' and 'Is HLF' columns."); st.stop()
                    clean_csv_in_memory = io.StringIO('\n'.join(lines[header_row_index:]))
                    df_peak = etl_long_to_wide(input_source=clean_csv_in_memory, datetime_column_name='Date (CET)', value_column_name='Is HLF')
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Peak restriction data cleaned and ETL successful!")
                    st.session_state['cached_df_peak'] = df_peak.copy()
                    st.session_state['cached_peak_config_key'] = peak_config_key
                except Exception as e: st.error(f"❌ A critical error occurred while processing the peak restriction file: {e}"); st.stop()

    # SEPARATE aFRR DATA LOADING (FIX) ---
    df_afrr_capacity = None
    df_afrr_up_capacity = None
    df_afrr_energy = None
    df_afrr_up_energy = None
    df_afrr_bids = None

    # Load Capacity Data if aFRR is selected and capacity is enabled
    if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity:
        if use_builtin_afrr:
            capacity_config_key = "afrr_capacity_builtin"
            if 'cached_afrr_capacity_config_key' in st.session_state and st.session_state.get('cached_afrr_capacity_config_key') == capacity_config_key:
                df_afrr_capacity = st.session_state.get('cached_df_afrr_capacity')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ Using cached aFRR capacity data!")
            else:
                try:
                    df_afrr_capacity = pd.read_csv("aFRRprices.csv")
                    df_afrr_capacity['datetime'] = pd.to_datetime(df_afrr_capacity['Date (CET)'])
                    df_afrr_capacity = df_afrr_capacity.set_index('datetime')
                    st.session_state['cached_df_afrr_capacity'] = df_afrr_capacity.copy()
                    st.session_state['cached_afrr_capacity_config_key'] = capacity_config_key
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Built-in aFRR capacity data loaded!")
                except Exception as e: st.error(f"❌ Failed to load built-in aFRRprices.csv: {e}")
        elif afrr_capacity_file:
            capacity_config_key = f"afrr_capacity_{afrr_capacity_file.name}_{hash(afrr_capacity_file.getvalue())}"
            # ... caching logic for uploaded capacity file ...
            try:
                df_afrr_capacity = pd.read_csv(afrr_capacity_file)
                df_afrr_capacity['datetime'] = pd.to_datetime(df_afrr_capacity['Date (CET)'])
                df_afrr_capacity = df_afrr_capacity.set_index('datetime')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ aFRR capacity data loaded!")
            except Exception as e: st.error(f"❌ Failed to process aFRR capacity file: {e}")
        else:
            st.warning("Please provide aFRR Capacity Auction data.")

    # Load aFRR Up Capacity Data if aFRR Up is selected
    if optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity:
        if use_builtin_afrr:
            up_capacity_config_key = "afrr_up_capacity_builtin"
            if 'cached_afrr_up_capacity_config_key' in st.session_state and st.session_state.get('cached_afrr_up_capacity_config_key') == up_capacity_config_key:
                df_afrr_up_capacity = st.session_state.get('cached_df_afrr_up_capacity')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ Using cached aFRR Up capacity data!")
            else:
                try:
                    df_afrr_up_capacity = pd.read_csv("aFRR_2024Predictions.csv")
                    df_afrr_up_capacity['datetime'] = pd.to_datetime(df_afrr_up_capacity['Date (CET)'])
                    df_afrr_up_capacity = df_afrr_up_capacity.set_index('datetime')
                    st.session_state['cached_df_afrr_up_capacity'] = df_afrr_up_capacity.copy()
                    st.session_state['cached_afrr_up_capacity_config_key'] = up_capacity_config_key
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Built-in aFRR Up capacity data loaded!")
                except Exception as e: st.error(f"❌ Failed to load built-in aFRR_2024Predictions.csv: {e}")
        elif afrr_up_capacity_file:
            up_capacity_config_key = f"afrr_up_capacity_{afrr_up_capacity_file.name}_{hash(afrr_up_capacity_file.getvalue())}"
            # ... caching logic for uploaded up capacity file ...
            try:
                df_afrr_up_capacity = pd.read_csv(afrr_up_capacity_file)
                df_afrr_up_capacity['datetime'] = pd.to_datetime(df_afrr_up_capacity['Date (CET)'])
                df_afrr_up_capacity = df_afrr_up_capacity.set_index('datetime')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ aFRR Up capacity data loaded!")
            except Exception as e: st.error(f"❌ Failed to process aFRR Up capacity file: {e}")
        else:
            st.warning("Please provide aFRR Up Capacity Auction data.")

    # Load Energy Data if aFRR is selected and energy is enabled
    if optimization_mode == "DA + aFRR Market" and enable_afrr_energy:
        if use_builtin_afrr:
            energy_config_key = "afrr_energy_builtin"
            if 'cached_afrr_energy_config_key' in st.session_state and st.session_state.get('cached_afrr_energy_config_key') == energy_config_key:
                df_afrr_energy = st.session_state.get('cached_df_afrr_energy')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ Using cached aFRR energy data!")
            else:
                try:
                    df_afrr_energy = pd.read_csv("aFRRenergylight.csv")
                    st.session_state['cached_df_afrr_energy'] = df_afrr_energy.copy()
                    st.session_state['cached_afrr_energy_config_key'] = energy_config_key
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ Built-in aFRR energy data loaded!")
                except Exception as e: st.error(f"❌ Failed to load built-in aFRRenergylight.csv: {e}")
        elif afrr_energy_file:
            energy_config_key = f"afrr_energy_{afrr_energy_file.name}_{hash(afrr_energy_file.getvalue())}"
            # ... caching logic for uploaded energy file ...
            try:
                df_afrr_energy = pd.read_csv(afrr_energy_file)
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ aFRR energy data loaded!")
            except Exception as e: st.error(f"❌ Failed to process aFRR energy file: {e}")
        else:
            st.warning("Please provide aFRR Energy Market data.")

    # Load aFRR Up Energy Data if aFRR Up capacity is enabled
    df_afrr_up_energy = None
    if optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity:
        if use_builtin_afrr:
            up_energy_config_key = "afrr_up_energy_builtin"
            if 'cached_afrr_up_energy_config_key' in st.session_state and st.session_state.get('cached_afrr_up_energy_config_key') == up_energy_config_key:
                df_afrr_up_energy = st.session_state.get('cached_df_afrr_up_energy')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ Using cached aFRR Up energy data!")
            else:
                try:
                    # Try to load dedicated Up energy file, fallback to symmetric prices from Down market
                    try:
                        df_afrr_up_energy = pd.read_csv("aFRRUpEnergyData.csv")
                        if not st.session_state.get('run_clicked', False):
                            st.success("✅ Built-in aFRR Up energy data loaded!")
                    except FileNotFoundError:
                        # Fallback: Use Down energy data as proxy for Up market
                        if df_afrr_energy is not None:
                            df_afrr_up_energy = df_afrr_energy.copy()
                            if not st.session_state.get('run_clicked', False):
                                st.info("ℹ️ Using aFRR Down energy data as proxy for Up market (symmetric pricing)")
                        else:
                            st.warning("⚠️ No aFRR Up energy data available and no Down energy data to use as fallback")
                            df_afrr_up_energy = None
                    
                    if df_afrr_up_energy is not None:
                        st.session_state['cached_df_afrr_up_energy'] = df_afrr_up_energy.copy()
                        st.session_state['cached_afrr_up_energy_config_key'] = up_energy_config_key
                except Exception as e: 
                    st.error(f"❌ Failed to load aFRR Up energy data: {e}")
                    df_afrr_up_energy = None
        else:
            # For uploaded files, we'd need a separate uploader - for now, use Down data as fallback
            if df_afrr_energy is not None:
                df_afrr_up_energy = df_afrr_energy.copy()
                if not st.session_state.get('run_clicked', False):
                    st.info("ℹ️ Using aFRR Down energy data as proxy for Up market")
            else:
                st.warning("Please provide aFRR Up Energy Market data or enable Down energy market.")

    # Load Dynamic Bids Data if aFRR is selected, capacity is enabled, and dynamic strategy is selected
    if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity and afrr_bid_strategy == 'Dynamic Bids (from CSV)':
        if afrr_dynamic_bids_file is None:
            st.warning("Dynamic bid strategy selected. Please upload your bid price CSV file."); st.stop()
        else:
            bids_config_key = f"afrr_bids_{afrr_dynamic_bids_file.name}_{hash(afrr_dynamic_bids_file.getvalue())}"
            if 'cached_afrr_bids_config_key' in st.session_state and st.session_state.get('cached_afrr_bids_config_key') == bids_config_key:
                df_afrr_bids = st.session_state.get('cached_df_afrr_bids')
                if not st.session_state.get('run_clicked', False):
                    st.success("✅ Using cached aFRR dynamic bid data!")
            else:
                try:
                    df_afrr_bids = pd.read_csv(afrr_dynamic_bids_file)
                    if 'Date (CET)' not in df_afrr_bids.columns or 'Bid Price' not in df_afrr_bids.columns:
                        st.error("❌ Dynamic bids file must contain 'Date (CET)' and 'Bid Price' columns.")
                        st.stop()
                    df_afrr_bids['datetime'] = pd.to_datetime(df_afrr_bids['Date (CET)'])
                    df_afrr_bids = df_afrr_bids.set_index('datetime')
                    st.session_state['cached_df_afrr_bids'] = df_afrr_bids.copy()
                    st.session_state['cached_afrr_bids_config_key'] = bids_config_key
                    if not st.session_state.get('run_clicked', False):
                        st.success("✅ aFRR dynamic bid data loaded!")
                except Exception as e:
                    st.error(f"❌ Failed to process aFRR dynamic bids file: {e}")
                    st.stop()

    if df_price is not None:
        try:
            # ... (Data filtering and preparation logic remains the same)
            st.sidebar.header("🗓️ Date Range Filter")
            df_price['date_obj'] = pd.to_datetime(df_price['date'])
            min_date, max_date = df_price['date_obj'].min().date(), df_price['date_obj'].max().date()
            col1, col2 = st.sidebar.columns(2)
            with col1: start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2: end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            if start_date > end_date: st.sidebar.error("Error: Start date cannot be after end date."); st.stop()
            mask = (df_price['date_obj'] >= pd.to_datetime(start_date)) & (df_price['date_obj'] <= pd.to_datetime(end_date))
            df_filtered = df_price.loc[mask].drop(columns=['date_obj'])
            df_processed = df_filtered
            if df_demand is not None:
                df_processed = pd.merge(df_filtered, df_demand, on='date', how='left', suffixes=('_price', '_demand'))
                if len(df_processed) == 0: st.error("❌ No matching dates found between price and demand data."); st.stop()
                # Fill missing demand values with the constant demand value
                demand_cols = [col for col in df_processed.columns if col.endswith('_demand')]
                for col in demand_cols:
                    df_processed[col] = df_processed[col].fillna(value=0)
            if df_peak is not None:
                peak_time_cols = [col for col in df_peak.columns if col != 'date']
                rename_map = {col: f"{col}_hlf" for col in peak_time_cols}
                df_peak_renamed = df_peak.rename(columns=rename_map)
                df_processed = pd.merge(df_processed, df_peak_renamed, on='date', how='left')
                hlf_time_cols = [col for col in df_processed.columns if col.endswith('_hlf')]
                if hlf_time_cols: df_processed[hlf_time_cols] = df_processed[hlf_time_cols].fillna(0)
            else: hlf_time_cols = []
            if not st.session_state.get('run_clicked', False):
                st.success(f"✅ Ready to analyze {len(df_processed)} days of data.")
            with st.spinner("Cleaning data..."):
                for col in df_processed.columns:
                    if col != 'date': df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan).interpolate(method='linear', limit_direction='both').fillna(df_processed[col].median())
            if not st.session_state.get('run_clicked', False):
                st.success("✅ Data cleaning completed")

            # --- Pre-computation happens here, but UI feedback is conditional ---
            afrr_won_blocks, afrr_15min_mask = None, None
            afrr_up_won_blocks, afrr_up_15min_mask = None, None
            
            if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity and df_afrr_capacity is not None:
                afrr_won_blocks, afrr_15min_mask = precompute_afrr_auction(
                    _df_afrr_capacity=df_afrr_capacity, afrr_bid_strategy=afrr_bid_strategy,
                    static_bid_price=afrr_bid_price, _df_afrr_bids=df_afrr_bids,
                    _df_peak=df_peak, holiday_set=holiday_set,
                    static_hochlast_intervals=hochlast_intervals_static, bid_mw=afrr_bid_mw
                )
            
            if optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity and df_afrr_up_capacity is not None:
                afrr_up_won_blocks, afrr_up_15min_mask = precompute_afrr_up_auction(
                    _df_afrr_up_capacity=df_afrr_up_capacity, static_bid_price=afrr_up_bid_price,
                    _df_peak=df_peak, holiday_set=holiday_set,
                    static_hochlast_intervals=hochlast_intervals_static, bid_mw=afrr_up_bid_mw
                )
            
            # --- Mutual Exclusivity Logic for aFRR Down/Up Overlaps ---
            if (afrr_won_blocks is not None and not afrr_won_blocks.empty and 
                afrr_up_won_blocks is not None and not afrr_up_won_blocks.empty):
                
                # Find overlapping blocks by comparing indices
                down_blocks = set(afrr_won_blocks.index)
                up_blocks = set(afrr_up_won_blocks.index)
                overlapping_blocks = down_blocks.intersection(up_blocks)
                
                # For each overlapping block, choose the one with higher revenue
                for block_time in overlapping_blocks:
                    down_revenue = afrr_won_blocks.loc[block_time, 'cap_payment']
                    up_revenue = afrr_up_won_blocks.loc[block_time, 'cap_payment']
                    
                    if down_revenue >= up_revenue:
                        # Remove from Up blocks
                        afrr_up_won_blocks = afrr_up_won_blocks.drop(block_time)
                        # Update the 15-min mask for Up
                        if afrr_up_15min_mask is not None:
                            block_start = block_time
                            block_end = block_time + pd.Timedelta(hours=4)
                            mask_range = (afrr_up_15min_mask.index >= block_start) & (afrr_up_15min_mask.index < block_end)
                            afrr_up_15min_mask.loc[mask_range] = False
                    else:
                        # Remove from Down blocks
                        afrr_won_blocks = afrr_won_blocks.drop(block_time)
                        # Update the 15-min mask for Down
                        if afrr_15min_mask is not None:
                            block_start = block_time
                            block_end = block_time + pd.Timedelta(hours=4)
                            mask_range = (afrr_15min_mask.index >= block_start) & (afrr_15min_mask.index < block_end)
                            afrr_15min_mask.loc[mask_range] = False

            afrr_clearing_prices_series = None
            afrr_activation_profile_series = None
            if optimization_mode == "DA + aFRR Market" and enable_afrr_energy and df_afrr_energy is not None:
                afrr_clearing_prices_series = extract_afrr_clearing_prices(df_afrr_energy)
                afrr_activation_profile_series = extract_afrr_activation_profile(df_afrr_energy)

            afrr_up_clearing_prices_series = None
            afrr_up_activation_profile_series = None
            if optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity and df_afrr_up_energy is not None:
                afrr_up_clearing_prices_series = extract_afrr_up_clearing_prices(df_afrr_up_energy)
                afrr_up_activation_profile_series = extract_afrr_up_activation_profile(df_afrr_up_energy)

            # --- Conditionally display the pre-run info ---
            if not st.session_state.get('run_clicked', False):
                pre_run_container = st.container()
                with pre_run_container:
                    if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity:
                        st.header("⚡ aFRR Capacity Auction Pre-computation")
                        if afrr_won_blocks is not None and not afrr_won_blocks.empty:
                            st.success(f"✅ Pre-computation complete. Found {len(afrr_won_blocks)} won aFRR blocks.")
                        elif afrr_won_blocks is not None:
                            st.info("ℹ️ Pre-computation complete. No aFRR blocks were won based on the current bid strategy.")
                        else:
                            st.warning("⚠️ Could not perform aFRR pre-computation. Check aFRR capacity data sources.")
                    
                    if optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity:
                        st.header("⚡ aFRR Up Capacity Auction Pre-computation")
                        if afrr_up_won_blocks is not None and not afrr_up_won_blocks.empty:
                            st.success(f"✅ Pre-computation complete. Found {len(afrr_up_won_blocks)} won aFRR Up blocks.")
                        elif afrr_up_won_blocks is not None:
                            st.info("ℹ️ Pre-computation complete. No aFRR Up blocks were won based on the current bid strategy.")
                        else:
                            st.warning("⚠️ Could not perform aFRR Up pre-computation. Check aFRR Up capacity data sources.")
                    
                    if optimization_mode == "DA + aFRR Market" and enable_afrr_energy:
                        st.header("⚡ aFRR Energy Market Analysis")
                        if afrr_clearing_prices_series is not None:
                            st.success(f"✅ Extracted {len(afrr_clearing_prices_series)} aFRR energy clearing price points.")
                        
                        if afrr_activation_profile_series is not None:
                            st.success(f"✅ Extracted {len(afrr_activation_profile_series)} activation profile points.")
                            avg_activation = afrr_activation_profile_series.mean()
                            st.info(f"📊 Average activation rate: {avg_activation:.1f}%")
                        else:
                            st.info("📊 Using default 100% activation profile as no activation data was found.")

            if df_demand is not None:
                price_time_cols = [col for col in df_processed.columns if col.endswith('_price')]
                demand_time_cols = [col for col in df_processed.columns if col.endswith('_demand')]
            else:
                price_time_cols = [col for col in df_processed.columns if col != 'date' and not col.endswith('_hlf')]
                demand_time_cols = []

            def build_da_only_model(prices, demand_profile, soc0, η_self, boiler_eff, 
                                    peak_restrictions=None, is_holiday=False, 
                                    afrr_commitment_mask=None, afrr_up_commitment_mask=None,
                                    enable_curve=False, curve_params=None, da_capacity_limit=None, da_soc_limit_mwh=None,
                                    optimization_objective="Minimize Cost", price_cap=150.0):
                """
                Builds the PuLP model for Day-Ahead optimization only.
                DA can see the real SOC but cannot charge beyond da_soc_limit_mwh.
                """
                T = len(prices)
                
                if optimization_objective == "Minimize Cost":
                    model = LpProblem("DA_Cost_Minimization", LpMinimize)
                else:  # Maximize Thermal from Electricity
                    model = LpProblem("DA_Maximize_Electric_Thermal", LpMaximize)
                da_cap = da_capacity_limit if da_capacity_limit is not None else Pmax_el
                effective_smax_da = da_soc_limit_mwh if da_soc_limit_mwh is not None else Smax
                
                # Decision variables
                p_el_da = LpVariable.dicts("p_el_da", range(T), lowBound=0, upBound=da_cap)
                p_th = LpVariable.dicts("p_th", range(T), lowBound=0, upBound=Pmax_th)
                p_gas = LpVariable.dicts("p_gas", range(T), lowBound=0)
                
                # SOC can go up to the actual Smax (for tracking purposes)
                # but we'll constrain charging to not exceed da_soc_limit_mwh
                soc = LpVariable.dicts("soc", range(T), lowBound=SOC_min, upBound=Smax)
                
                # Binary variables to track if we can charge (SOC < limit)
                can_charge = LpVariable.dicts("can_charge", range(T), cat='Binary')
                
                # Big M for binary constraints
                M = 1000.0
                
                # Hourly block constraints for DA market
                for hour_idx, t in enumerate(range(0, T, 4)):
                    if t + 3 < T:
                        model += p_el_da[t+1] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int1"
                        model += p_el_da[t+2] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int2"
                        model += p_el_da[t+3] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int3"
               
                # Objective function
                if optimization_objective == "Minimize Cost":
                    # EXISTING: Minimize costs
                    da_costs = lpSum([(prices[t] + C_grid) * p_el_da[t] * Δt for t in range(T)])
                    gas_costs = lpSum([(C_gas / boiler_eff) * p_gas[t] * Δt for t in range(T)])
                    model += da_costs + gas_costs - terminal_value * soc[T-1]
                else:
                    # NEW: Maximize thermal energy from electricity
                    total_electric_thermal = lpSum([p_th[t] * Δt for t in range(T)])
                    
                    # Add a smarter cost penalty to break ties and avoid high prices
                    cost_penalty = 0
                    for t in range(T):
                        # Use a very high penalty if price exceeds the cap
                        if prices[t] > price_cap:
                            penalty_multiplier = 10000.0 # A large number to make charging extremely unattractive
                        else:
                            penalty_multiplier = 0.001 # The original small penalty
                        
                        cost_penalty += penalty_multiplier * (prices[t] + C_grid) * p_el_da[t] * Δt

                    model += total_electric_thermal - cost_penalty
                
                # Power curve parameters (if enabled)
                if enable_curve and curve_params is not None:
                    # Calculate curve parameters based on ACTUAL Smax (not DA limit)
                    s_cut_charge_frac = curve_params['charge_start_pct'] / 100.0
                    p_min_charge_frac = curve_params['charge_end_pct'] / 100.0
                    s_cut_discharge_frac = curve_params['discharge_start_pct'] / 100.0
                    p_min_discharge_frac = curve_params['discharge_end_pct'] / 100.0
                    
                    soc_charge_knee = s_cut_charge_frac * Smax
                    soc_discharge_knee = s_cut_discharge_frac * Smax
                    
                    if (Smax - soc_charge_knee) > 1e-6:
                        power_at_full = p_min_charge_frac * Pmax_el
                        m_charge = (power_at_full - Pmax_el) / (Smax - soc_charge_knee)
                        b_charge = Pmax_el - m_charge * soc_charge_knee
                    else:
                        m_charge, b_charge = None, None
                        
                    if (soc_discharge_knee - SOC_min) > 1e-6:
                        power_at_empty = p_min_discharge_frac * Pmax_th
                        m_discharge = (Pmax_th - power_at_empty) / (soc_discharge_knee - SOC_min)
                        b_discharge = power_at_empty - m_discharge * SOC_min
                    else:
                        m_discharge, b_discharge = None, None
                
                # Main constraints
                for t in range(T):
                    # Thermal balance
                    model += p_th[t] + p_gas[t] == demand_profile[t]
                    
                    # Binary constraint: can_charge = 1 if prev_soc < da_soc_limit
                    if t == 0:
                        # For first period, compare initial SOC directly
                        if soc0 < effective_smax_da:
                            model += can_charge[t] == 1
                        else:
                            model += can_charge[t] == 0
                    else:
                        # For subsequent periods, use binary constraints with previous SOC variable
                        model += soc[t-1] <= effective_smax_da + M * (1 - can_charge[t])
                        model += soc[t-1] >= effective_smax_da - M * can_charge[t] + 0.001
                    
                    # If we can't charge (SOC >= limit), then p_el_da must be 0
                    model += p_el_da[t] <= M * can_charge[t]
                    
                    # DA market restrictions (HLF, aFRR capacity commitments)
                    is_da_restricted = False
                    if not is_holiday:
                        if t in hochlast_intervals_static:
                            is_da_restricted = True
                        elif peak_restrictions is not None and len(peak_restrictions) > t and peak_restrictions[t] == 1:
                            is_da_restricted = True
                    
                    if afrr_commitment_mask and len(afrr_commitment_mask) > t and afrr_commitment_mask[t]:
                        is_da_restricted = True
                    
                    if is_da_restricted:
                        model += p_el_da[t] == 0
                    
                    if optimization_objective == "Maximize Thermal from Electricity" and prices[t] > price_cap:
                        model += p_el_da[t] == 0
                    
                    # Apply power curves if enabled (based on actual SOC)
                    if enable_curve and curve_params is not None:
                        if m_charge is not None:
                            model += p_el_da[t] <= Pmax_el
                            if t == 0:
                                model += p_el_da[t] <= m_charge * soc0 + b_charge
                            else:
                                model += p_el_da[t] <= m_charge * soc[t-1] + b_charge
                        if m_discharge is not None:
                            model += p_th[t] <= Pmax_th
                            if t == 0:
                                model += p_th[t] <= m_discharge * soc0 + b_discharge
                            else:
                                model += p_th[t] <= m_discharge * soc[t-1] + b_discharge
                    
                    # aFRR Up constraint: Ensure sufficient SOC reserve for upward regulation
                    if afrr_up_commitment_mask and len(afrr_up_commitment_mask) > t and afrr_up_commitment_mask[t]:
                        # Check if we're at the start of a 4-hour block (every 16 intervals = 4 hours)
                        block_start = (t // 16) * 16  # Find start of current 4-hour block
                        if t >= block_start:  # Only enforce during the committed block
                            # Required energy reserve for the full 4-hour block commitment
                            # Account for thermal output efficiency (no conversion loss for direct discharge)
                            required_energy_reserve = afrr_up_bid_mw * 4.0  # MWh needed for 4-hour block
                            
                            # Apply constraint based on time step
                            if t == 0:
                                # For first interval, check initial SOC
                                model += soc0 >= required_energy_reserve, f"MinSOC_aFRR_up_{t}_init"
                            else:
                                # For subsequent intervals, ensure previous SOC can support commitment
                                model += soc[t-1] >= required_energy_reserve, f"MinSOC_aFRR_up_{t}"
                    
                    # SOC dynamics
                    if t == 0:
                        model += soc[t] == soc0 * η_self + η * p_el_da[t] * Δt - p_th[t] * Δt
                    else:
                        model += soc[t] == soc[t-1] * η_self + η * p_el_da[t] * Δt - p_th[t] * Δt
                
                return model, p_el_da, p_th, p_gas, soc

            def simulate_afrr_energy(da_plan, soc0, daily_clearing_prices, daily_activation_profile, η_self, 
                                     enable_curve=False, curve_params=None, Pmax_el=2.0, Smax=8.0, 
                                     SOC_min=0.0, η=0.95, Δt=0.25, C_grid=30.0, C_gas=65.0, 
                                     boiler_efficiency=0.9, soc_premium_table=None, 
                                     afrr_energy_bid_base=36.0):
                """
                Simulates aFRR energy market participation on top of a fixed DA plan.
                aFRR can use the FULL battery capacity (8 MWh).
                """
                T = len(da_plan['p_el_da'])
                p_el_afrr = [0.0] * T
                final_soc_trajectory = [0.0] * T

                net_cost = 0.0
                savings_vs_gas = 0.0

                current_soc = soc0

                # Setup power curve parameters if enabled
                max_charge_power = Pmax_el
                if enable_curve and curve_params is not None:
                    s_cut_charge_frac = curve_params['charge_start_pct'] / 100.0
                    p_min_charge_frac = curve_params['charge_end_pct'] / 100.0
                    soc_charge_knee = s_cut_charge_frac * Smax

                    if (Smax - soc_charge_knee) > 1e-6:
                        power_at_full = p_min_charge_frac * Pmax_el
                        m_charge = (power_at_full - Pmax_el) / (Smax - soc_charge_knee)
                        b_charge = Pmax_el - m_charge * soc_charge_knee
                    else:
                        m_charge, b_charge = None, None
                else:
                    m_charge, b_charge = None, None

                for t in range(T):
                    p_el_afrr[t] = 0.0  # Reset for this interval

                    # 1. Apply power curve to determine actual max charging power at current SOC
                    if enable_curve and m_charge is not None:
                        # The power curve limits charging based on SOC
                        if current_soc < soc_charge_knee:
                            max_charge_at_soc = Pmax_el
                        else:
                            max_charge_at_soc = min(Pmax_el, m_charge * current_soc + b_charge)
                        max_charge_at_soc = max(0, max_charge_at_soc)  # Ensure non-negative
                    else:
                        max_charge_at_soc = Pmax_el

                    # 2. Determine available capacity considering both DA commitment and power curve
                    already_charging_da = da_plan['p_el_da'][t]
                    remaining_capacity = max_charge_at_soc - already_charging_da
                    remaining_capacity = max(0, remaining_capacity)  # Ensure non-negative

                    bid_size_mw = np.floor(remaining_capacity)

                    # 3. Check SOC limit - aFRR can charge up to full Smax (8 MWh)
                    max_additional_energy = (Smax - current_soc) / (η * Δt)
                    bid_size_mw = min(bid_size_mw, np.floor(max_additional_energy))

                    # 4. Decide if we can and want to bid
                    can_bid = bid_size_mw >= 1.0
                    we_win_bid = False

                    if can_bid:
                        # Determine bid premium based on current SOC
                        premium = 0.0
                        soc_levels = sorted([float(k) for k in soc_premium_table.keys()])
                        for i in range(len(soc_levels) - 1):
                            if soc_levels[i] <= current_soc <= soc_levels[i+1]:
                                soc_range = soc_levels[i+1] - soc_levels[i]
                                if soc_range > 0:
                                    weight = (current_soc - soc_levels[i]) / soc_range
                                    premium = (1 - weight) * float(soc_premium_table[soc_levels[i]]) + weight * float(soc_premium_table[soc_levels[i+1]])
                                else:
                                    premium = float(soc_premium_table[soc_levels[i]])
                                break
                        if current_soc >= soc_levels[-1]:
                            premium = float(soc_premium_table[soc_levels[-1]])

                        effective_bid = float(afrr_energy_bid_base) + premium
                        clearing_price = safe_float_convert(daily_clearing_prices[t]) if daily_clearing_prices is not None else 0.0

                        if clearing_price >= effective_bid:
                            we_win_bid = True

                    # 5. If we won, calculate the activated power based on our integer bid
                    if we_win_bid:
                        activation_frac = (safe_float_convert(daily_activation_profile[t]) / 100.0) if daily_activation_profile is not None else 1.0
                        called_power = bid_size_mw * activation_frac

                        if called_power > 0.01:
                            power_charged_afrr = called_power
                            p_el_afrr[t] = power_charged_afrr

                            # Calculate financial impact for this interval
                            net_cost_per_mwh = C_grid - clearing_price
                            net_cost_for_interval = net_cost_per_mwh * power_charged_afrr * Δt
                            net_cost += net_cost_for_interval

                            # Calculate savings vs. using gas for the same thermal energy
                            thermal_via_afrr = power_charged_afrr * Δt * η
                            gas_alternative_cost = thermal_via_afrr * (C_gas / boiler_efficiency)
                            savings_vs_gas += (gas_alternative_cost - net_cost_for_interval)

                    # 6. Update SOC - discharge comes from entire SOC
                    total_charging = da_plan['p_el_da'][t] + p_el_afrr[t]
                    current_soc = current_soc * η_self + η * total_charging * Δt - da_plan['p_th'][t] * Δt
                    current_soc = max(SOC_min, min(Smax, current_soc))
                    final_soc_trajectory[t] = current_soc

                return {
                    "p_el_afrr": p_el_afrr,
                    "final_soc_trajectory": final_soc_trajectory,
                    "net_cost": net_cost,
                    "savings_vs_gas": savings_vs_gas
                }

            def simulate_afrr_up_energy(da_plan, soc0, daily_clearing_prices, daily_activation_profile, η_self, 
                                       enable_curve=False, curve_params=None, Pmax_th=2.0, Smax=8.0, 
                                       SOC_min=0.0, η=0.95, Δt=0.25, C_grid=30.0, C_gas=65.0, 
                                       boiler_efficiency=0.9, afrr_up_energy_bid_base=50.0, afrr_up_max_discount=20.0):
                """
                Simulates aFRR Up energy market participation on top of a fixed DA plan.
                aFRR Up provides upward regulation by discharging when called.
                The bid price = base_bid - discount, where discount increases as SOC increases.
                """
                T = len(da_plan['p_el_da'])
                p_th_afrr_up = [0.0] * T  # Additional thermal output from aFRR Up
                final_soc_trajectory = [0.0] * T

                net_cost = 0.0  # Net cost impact (should be negative = revenue)
                savings_vs_gas = 0.0

                current_soc = soc0

                # Setup power curve parameters if enabled
                max_discharge_power = Pmax_th
                if enable_curve and curve_params is not None:
                    s_cut_discharge_frac = curve_params['discharge_start_pct'] / 100.0
                    p_min_discharge_frac = curve_params['discharge_end_pct'] / 100.0
                    soc_discharge_knee = s_cut_discharge_frac * Smax

                    if soc_discharge_knee > 1e-6:
                        power_at_empty = p_min_discharge_frac * Pmax_th
                        m_discharge = (Pmax_th - power_at_empty) / soc_discharge_knee
                        b_discharge = power_at_empty
                    else:
                        m_discharge, b_discharge = None, None
                else:
                    m_discharge, b_discharge = None, None

                for t in range(T):
                    p_th_afrr_up[t] = 0.0  # Reset for this interval

                    # 1. Apply power curve to determine actual max discharging power at current SOC
                    if enable_curve and m_discharge is not None:
                        if current_soc <= soc_discharge_knee:
                            max_discharge_at_soc = m_discharge * current_soc + b_discharge
                        else:
                            max_discharge_at_soc = Pmax_th
                        max_discharge_at_soc = max(0, max_discharge_at_soc)  # Ensure non-negative
                    else:
                        max_discharge_at_soc = Pmax_th

                    # 2. Determine available discharge capacity considering DA thermal output
                    already_discharging_da = da_plan['p_th'][t]
                    remaining_discharge_capacity = max_discharge_at_soc - already_discharging_da
                    remaining_discharge_capacity = max(0, remaining_discharge_capacity)  # Ensure non-negative

                    bid_size_mw = np.floor(remaining_discharge_capacity)

                    # 3. Check SOC limit - ensure we have enough energy to discharge
                    max_additional_energy = (current_soc - SOC_min) / Δt
                    bid_size_mw = min(bid_size_mw, np.floor(max_additional_energy))

                    # 4. Decide if we can and want to bid
                    can_bid = bid_size_mw >= 1.0
                    we_win_bid = False

                    if can_bid:
                        # Determine bid discount based on current SOC
                        # Higher SOC = more discount = lower bid price = more likely to win
                        soc_fraction = current_soc / Smax
                        discount = soc_fraction * afrr_up_max_discount  # Configurable max discount when fully charged
                        
                        effective_bid = float(afrr_up_energy_bid_base) - discount
                        clearing_price = safe_float_convert(daily_clearing_prices[t]) if daily_clearing_prices is not None else 0.0

                        if clearing_price >= effective_bid:
                            we_win_bid = True

                    # 5. If we won, calculate the activated power based on our integer bid
                    if we_win_bid:
                        activation_frac = (safe_float_convert(daily_activation_profile[t]) / 100.0) if daily_activation_profile is not None else 1.0
                        called_power = bid_size_mw * activation_frac

                        if called_power > 0.01:
                            power_discharged_afrr_up = called_power
                            p_th_afrr_up[t] = power_discharged_afrr_up

                            # Calculate financial impact for this interval
                            # Revenue from selling energy at clearing price
                            revenue_for_interval = clearing_price * power_discharged_afrr_up * Δt
                            net_cost -= revenue_for_interval  # Negative cost = revenue

                            # Calculate savings vs. generating thermal with gas
                            thermal_via_afrr_up = power_discharged_afrr_up * Δt
                            gas_alternative_cost = thermal_via_afrr_up * (C_gas / boiler_efficiency)
                            savings_vs_gas += gas_alternative_cost  # Pure savings since we get paid to discharge

                    # 6. Update SOC - additional discharge from aFRR Up
                    total_charging = da_plan['p_el_da'][t]
                    total_discharging = da_plan['p_th'][t] + p_th_afrr_up[t]
                    current_soc = current_soc * η_self + η * total_charging * Δt - total_discharging * Δt
                    current_soc = max(SOC_min, min(Smax, current_soc))
                    final_soc_trajectory[t] = current_soc

                return {
                    "p_th_afrr_up": p_th_afrr_up,
                    "final_soc_trajectory": final_soc_trajectory,
                    "net_cost": net_cost,
                    "savings_vs_gas": savings_vs_gas
                }

            η_self = (1 - self_discharge_daily / 100) ** (Δt / 24)
            
            curve_params = None
            if enable_power_curve:
                curve_params = {
                    'charge_start_pct': charge_taper_soc_pct,
                    'charge_end_pct': charge_power_at_full_pct,
                    'discharge_start_pct': discharge_taper_soc_pct,
                    'discharge_end_pct': discharge_power_at_empty_pct
                }


            if st.button("🚀 Run Optimization", type="primary"):
                st.session_state.run_clicked = True  # Set flag to hide pre-run UI
                if 'results' in st.session_state: del st.session_state['results']
                if 'all_trades' in st.session_state: del st.session_state['all_trades']
                if 'gas_baseline' in st.session_state: del st.session_state['gas_baseline']

                progress_bar = st.progress(0); status_text = st.empty()
                soc0 = float(SOC_min)
                results, all_trades, all_baselines = [], [], []
                da_soc_limit_mwh = Smax * (da_max_soc_pct / 100.0)

                for idx, (_, row) in enumerate(df_processed.iterrows()):
                    progress_bar.progress((idx + 1) / len(df_processed))
                    day = row['date']; status_text.text(f"Processing day {idx + 1}/{len(df_processed)}: {day}")
                    prices = row[price_time_cols].values
                    demand_profile = np.full(len(prices), D_th) if demand_option == 'Constant Demand' else row[demand_time_cols].values
                    T = len(prices)

                    # --- Data Preparation for the Day ---
                    # aFRR capacity commitment mask
                    daily_afrr_cap_revenue = 0
                    blocked_intervals_for_day = [False] * T
                    if optimization_mode == 'DA + aFRR Market' and afrr_15min_mask is not None:
                        day_dt = pd.to_datetime(day); day_dt = day_dt.tz_localize(None) if day_dt.tz is not None else day_dt
                        day_start = day_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                        try:
                            daily_mask = afrr_15min_mask[(afrr_15min_mask.index >= day_start) & (afrr_15min_mask.index <= day_end)]
                            if not daily_mask.empty:
                                mask_values = daily_mask.values
                                num_to_copy = min(len(mask_values), T)
                                blocked_intervals_for_day[:num_to_copy] = [bool(x) for x in mask_values[:num_to_copy]]
                        except Exception as e: st.warning(f"aFRR mask extraction failed for {day}: {e}")
                        
                        if afrr_won_blocks is not None:
                            daily_won_blocks = afrr_won_blocks[(afrr_won_blocks.index >= day_start) & (afrr_won_blocks.index < day_end)]
                            daily_afrr_cap_revenue = daily_won_blocks["cap_payment"].sum()
                    
                    # aFRR Up capacity commitment mask
                    daily_afrr_up_cap_revenue = 0
                    afrr_up_blocked_intervals_for_day = [False] * T
                    if optimization_mode == 'DA + aFRR Market' and afrr_up_15min_mask is not None:
                        try:
                            daily_up_mask = afrr_up_15min_mask[(afrr_up_15min_mask.index >= day_start) & (afrr_up_15min_mask.index <= day_end)]
                            if not daily_up_mask.empty:
                                up_mask_values = daily_up_mask.values
                                num_to_copy = min(len(up_mask_values), T)
                                afrr_up_blocked_intervals_for_day[:num_to_copy] = [bool(x) for x in up_mask_values[:num_to_copy]]
                        except Exception as e: st.warning(f"aFRR Up mask extraction failed for {day}: {e}")
                        
                        if afrr_up_won_blocks is not None:
                            daily_up_won_blocks = afrr_up_won_blocks[(afrr_up_won_blocks.index >= day_start) & (afrr_up_won_blocks.index < day_end)]
                            daily_afrr_up_cap_revenue = daily_up_won_blocks["cap_payment"].sum()

                    # Gas baseline and holiday status
                    gas_baseline_daily = (sum(demand_profile) * Δt * C_gas) / boiler_efficiency
                    all_baselines.append(gas_baseline_daily)
                    is_holiday = day in holiday_set
                    peak_restrictions_for_day = row[hlf_time_cols].values if (df_peak is not None and hlf_time_cols) else None
                    
                    # aFRR energy market data
                    daily_clearing_prices = np.zeros(T)
                    if afrr_clearing_prices_series is not None:
                        day_dt = pd.to_datetime(day).tz_localize(None)
                        day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                        daily_prices_series = afrr_clearing_prices_series[(afrr_clearing_prices_series.index >= day_start) & (afrr_clearing_prices_series.index <= day_end)]
                        if not daily_prices_series.empty:
                            source_data = daily_prices_series.values
                            num_to_copy = min(T, len(source_data))
                            daily_clearing_prices[:num_to_copy] = source_data[:num_to_copy]
                    
                    daily_afrr_activation_profile = np.full(T, 100.0)
                    if afrr_activation_profile_series is not None:
                        day_dt = pd.to_datetime(day).tz_localize(None)
                        day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                        daily_activation_series = afrr_activation_profile_series[(afrr_activation_profile_series.index >= day_start) & (afrr_activation_profile_series.index <= day_end)]
                        if not daily_activation_series.empty:
                           source_data = daily_activation_series.values
                           num_to_copy = min(T, len(source_data))
                           daily_afrr_activation_profile[:num_to_copy] = source_data[:num_to_copy]
                    
                    # aFRR Up energy market data
                    daily_up_clearing_prices = np.zeros(T)
                    if afrr_up_clearing_prices_series is not None:
                        day_dt = pd.to_datetime(day).tz_localize(None)
                        day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                        daily_up_prices_series = afrr_up_clearing_prices_series[(afrr_up_clearing_prices_series.index >= day_start) & (afrr_up_clearing_prices_series.index <= day_end)]
                        if not daily_up_prices_series.empty:
                            up_source_data = daily_up_prices_series.values
                            num_to_copy = min(T, len(up_source_data))
                            daily_up_clearing_prices[:num_to_copy] = up_source_data[:num_to_copy]
                    
                    daily_afrr_up_activation_profile = np.full(T, 100.0)
                    if afrr_up_activation_profile_series is not None:
                        day_dt = pd.to_datetime(day).tz_localize(None)
                        day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                        daily_up_activation_series = afrr_up_activation_profile_series[(afrr_up_activation_profile_series.index >= day_start) & (afrr_up_activation_profile_series.index <= day_end)]
                        if not daily_up_activation_series.empty:
                            up_source_data = daily_up_activation_series.values
                            num_to_copy = min(T, len(up_source_data))
                            daily_afrr_up_activation_profile[:num_to_copy] = up_source_data[:num_to_copy]
                    
                    # --- STAGE 1: Day-Ahead Market Optimization ---
                    model, p_el_da_vars, p_th_vars, p_gas_vars, soc_vars = build_da_only_model(
                        prices, demand_profile, soc0, η_self, boiler_efficiency,
                        peak_restrictions_for_day, is_holiday, blocked_intervals_for_day, afrr_up_blocked_intervals_for_day,
                        enable_curve=enable_power_curve, curve_params=curve_params, da_capacity_limit=da_max_capacity, 
                        da_soc_limit_mwh=da_soc_limit_mwh,
                        optimization_objective=optimization_objective,
                        price_cap=price_cap_for_max_thermal
                    )
                    status = model.solve(HiGHS(msg=False))

                    if status == 1:
                        # Extract DA plan from the optimization results
                        da_plan = {
                            'p_el_da': [p_el_da_vars[t].value() for t in range(T)],
                            'p_th': [p_th_vars[t].value() for t in range(T)],
                            'p_gas': [p_gas_vars[t].value() for t in range(T)],
                        }
                        
                        # --- STAGE 2: aFRR Energy Market Simulation ---
                        sim_results = None
                        if optimization_mode == "DA + aFRR Market" and enable_afrr_energy:
                            sim_results = simulate_afrr_energy(
                                da_plan, soc0, daily_clearing_prices, daily_afrr_activation_profile, η_self,
                                enable_curve=enable_power_curve, curve_params=curve_params,
                                Pmax_el=Pmax_el, Smax=Smax, SOC_min=SOC_min, η=η, Δt=Δt,
                                C_grid=C_grid, C_gas=C_gas, boiler_efficiency=boiler_efficiency,
                                soc_premium_table=soc_premium_table, afrr_energy_bid_base=afrr_energy_bid_base
                            )

                        else:
                            # If aFRR energy is disabled, create a zeroed-out result
                            initial_soc_trajectory = [soc0] + [soc_vars[t].value() for t in range(T)]
                            sim_results = {
                                "p_el_afrr": [0.0] * T,
                                "final_soc_trajectory": initial_soc_trajectory[1:], # remove initial soc
                                "net_cost": 0.0,
                                "savings_vs_gas": 0.0
                            }
                        
                        # --- STAGE 2b: aFRR Up Energy Market Simulation ---
                        sim_up_results = None
                        if (optimization_mode == "DA + aFRR Market" and enable_afrr_up_capacity and 
                            daily_afrr_up_cap_revenue > 0):
                            # Use SOC after Down energy simulation if available
                            soc_after_down = sim_results['final_soc_trajectory'][-1] if sim_results else soc0
                            
                            sim_up_results = simulate_afrr_up_energy(
                                da_plan, soc_after_down, daily_up_clearing_prices, daily_afrr_up_activation_profile, η_self,
                                enable_curve=enable_power_curve, curve_params=curve_params,
                                Pmax_th=Pmax_th, Smax=Smax, SOC_min=SOC_min, η=η, Δt=Δt,
                                C_grid=C_grid, C_gas=C_gas, boiler_efficiency=boiler_efficiency,
                                afrr_up_energy_bid_base=afrr_up_energy_bid_base, afrr_up_max_discount=afrr_up_max_discount
                            )
                        else:
                            # If aFRR Up energy is disabled, create a zeroed-out result
                            final_soc_after_down = sim_results['final_soc_trajectory'] if sim_results else [soc0] * T
                            sim_up_results = {
                                "p_th_afrr_up": [0.0] * T,
                                "final_soc_trajectory": final_soc_after_down,
                                "net_cost": 0.0,
                                "savings_vs_gas": 0.0
                            }
                        
                        # --- STAGE 3: Aggregation & Reporting ---
                        elec_cost_da = sum((prices[t] + C_grid) * da_plan['p_el_da'][t] * Δt for t in range(T))
                        gas_cost = sum(C_gas * (da_plan['p_gas'][t] / boiler_efficiency) * Δt for t in range(T))
                        afrr_energy_net_cost = sim_results['net_cost']
                        afrr_energy_revenue = sim_results['savings_vs_gas'] # This is the "value" metric
                        afrr_up_energy_net_cost = sim_up_results['net_cost'] if sim_up_results else 0.0
                        afrr_up_energy_revenue = sim_up_results['savings_vs_gas'] if sim_up_results else 0.0
                        
                        reported_cash_flow_cost = (elec_cost_da + gas_cost + afrr_energy_net_cost + afrr_up_energy_net_cost 
                                                 - daily_afrr_cap_revenue - daily_afrr_up_cap_revenue)
                        savings = gas_baseline_daily - reported_cash_flow_cost
                        
                        elec_energy_da = sum([p * Δt for p in da_plan['p_el_da']])
                        elec_energy_afrr = sum([p * Δt for p in sim_results['p_el_afrr']])
                        elec_energy = elec_energy_da + elec_energy_afrr
                        gas_fuel_energy = sum([(p / boiler_efficiency) * Δt for p in da_plan['p_gas']])
                        
                        # Populate detailed trade records for the day
                        for t in range(T):
                            interval_hour, interval_min = divmod(t * 15, 60); time_str = f"{interval_hour:02d}:{interval_min:02d}:00"
                            gas_cost_interval = C_gas * (da_plan['p_gas'][t] / boiler_efficiency) * Δt
                            elec_cost_da_interval = (prices[t] + C_grid) * da_plan['p_el_da'][t] * Δt
                            
                            afrr_net_cost_interval = 0
                            if sim_results['p_el_afrr'][t] > 0.01:
                                clearing_price = safe_float_convert(daily_clearing_prices[t]) if daily_clearing_prices is not None else 0.0
                                afrr_net_cost_interval = (C_grid - clearing_price) * sim_results['p_el_afrr'][t] * Δt
                            
                            total_elec_power = da_plan['p_el_da'][t] + sim_results['p_el_afrr'][t]
                            is_static_restricted = t in hochlast_intervals_static and not is_holiday
                            is_dynamic_restricted = (peak_restrictions_for_day is not None and len(peak_restrictions_for_day) > t and peak_restrictions_for_day[t] == 1 and not is_holiday)
                            
                            trade_record = {
                                'date': day, 'time': time_str, 'interval': t, 'da_price': prices[t],
                                'total_elec_cost': prices[t] + C_grid, 'p_el_heater': total_elec_power,
                                'p_el_da': da_plan['p_el_da'][t], 'p_el_afrr': sim_results['p_el_afrr'][t],
                                'p_th_discharge': da_plan['p_th'][t], 'p_gas_backup': da_plan['p_gas'][t],
                                'soc': (sim_up_results['final_soc_trajectory'][t] if sim_up_results and sim_up_results['final_soc_trajectory']
                                       else sim_results['final_soc_trajectory'][t] if sim_results else soc0),
                                'elec_cost_interval': elec_cost_da_interval + afrr_net_cost_interval,
                                'gas_cost_interval': gas_cost_interval,
                                'total_cost_interval': elec_cost_da_interval + afrr_net_cost_interval + gas_cost_interval,
                                'is_hochlast': is_static_restricted or is_dynamic_restricted,
                                'is_holiday': is_holiday, 'is_charging': total_elec_power > 0.01,
                                'is_discharging': da_plan['p_th'][t] > 0.01,
                                'using_gas': da_plan['p_gas'][t] > 0.01, 'demand_th': demand_profile[t],
                                'is_in_afrr_market': blocked_intervals_for_day[t] if enable_afrr_capacity else False,
                                'is_in_afrr_up_market': afrr_up_blocked_intervals_for_day[t] if enable_afrr_up_capacity else False,
                                'afrr_energy_won': 1 if sim_results['p_el_afrr'][t] > 0.01 else 0,
                                'afrr_up_energy_won': 1 if (sim_up_results and sim_up_results['p_th_afrr_up'][t] > 0.01) else 0,
                                'p_th_afrr_up': sim_up_results['p_th_afrr_up'][t] if sim_up_results else 0
                            }
                            all_trades.append(trade_record)
                        
                        results.append({
                            "day": day, "cost": reported_cash_flow_cost, "savings": savings, 
                            "soc_end": (sim_up_results['final_soc_trajectory'][-1] if sim_up_results and sim_up_results['final_soc_trajectory']
                                       else sim_results['final_soc_trajectory'][-1] if sim_results and sim_results['final_soc_trajectory']
                                       else soc0), 
                            "elec_energy": elec_energy, "gas_energy": gas_fuel_energy, "is_holiday": is_holiday, 
                            "gas_baseline_daily": gas_baseline_daily, "elec_cost_da": elec_cost_da, "gas_cost": gas_cost,
                            "afrr_cap_revenue": daily_afrr_cap_revenue, "afrr_up_cap_revenue": daily_afrr_up_cap_revenue, 
                            "afrr_energy_revenue": afrr_energy_revenue, "afrr_up_energy_revenue": afrr_up_energy_revenue
                        })
                        
                        # --- STAGE 4: Update SOC for the next day's planning ---
                        # Use final SOC from Up energy simulation if available, otherwise Down simulation
                        soc0 = (sim_up_results['final_soc_trajectory'][-1] if sim_up_results and sim_up_results['final_soc_trajectory']
                               else sim_results['final_soc_trajectory'][-1] if sim_results and sim_results['final_soc_trajectory']
                               else soc0)
                    
                    else: # If optimization fails
                        # Fallback to gas-only operation for the day
                        fallback_gas_cost = gas_baseline_daily 
                        results.append({
                            "day": day, "cost": fallback_gas_cost, "savings": 0, "soc_end": soc0, 
                            "elec_energy": 0, "gas_energy": sum(demand_profile) * Δt, "is_holiday": is_holiday, 
                            "gas_baseline_daily": gas_baseline_daily, "elec_cost_da": 0, "gas_cost": fallback_gas_cost,
                            "afrr_cap_revenue": 0, "afrr_up_cap_revenue": 0, "afrr_energy_revenue": 0, "afrr_up_energy_revenue": 0
                        })
                        # Add empty trade records for the failed day
                        for t in range(T):
                             all_trades.append({
                                'date': day, 'time': f"{t*15//60:02d}:{t*15%60:02d}:00", 'interval': t, 'da_price': prices[t],
                                'total_elec_cost': prices[t] + C_grid, 'p_el_heater': 0, 'p_el_da': 0, 'p_el_afrr': 0,
                                'p_th_discharge': 0, 'p_gas_backup': demand_profile[t], 'soc': soc0, 
                                'elec_cost_interval': 0, 'gas_cost_interval': (demand_profile[t] * Δt * C_gas) / boiler_efficiency,
                                'total_cost_interval': (demand_profile[t] * Δt * C_gas) / boiler_efficiency,
                                'is_hochlast': False, 'is_holiday': is_holiday, 'is_charging': False, 'is_discharging': False,
                                'using_gas': True, 'demand_th': demand_profile[t], 'is_in_afrr_market': False, 'afrr_energy_won': 0
                            })

                progress_bar.progress(1.0); status_text.text("✅ Optimization completed!")
                st.session_state['results'] = results; st.session_state['all_trades'] = all_trades
                st.session_state['gas_baseline_total'] = sum(all_baselines) if all_baselines else 0
        except Exception as e: st.error(f"❌ An error occurred during optimization: {str(e)}"); st.stop()

        if 'results' in st.session_state and st.session_state['results']:
            results, all_trades, gas_baseline_total = st.session_state['results'], st.session_state['all_trades'], st.session_state['gas_baseline_total']
            results_df = pd.DataFrame(results); results_df['date'] = pd.to_datetime(results_df['day'])
            trades_df = pd.DataFrame(all_trades)

            col1, col2 = st.columns([3, 1])
            with col1: st.header("Results Summary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    del st.session_state['results'], st.session_state['all_trades'], st.session_state['gas_baseline_total']
                    st.session_state.run_clicked = False  # Reset flag to show pre-run info again
                    st.rerun()

            # --- KPI Calculations ---
            total_savings = sum([r['savings'] for r in results])
            total_afrr_cap_revenue = sum([r.get('afrr_cap_revenue', 0) for r in results]) if enable_afrr_capacity else 0
            total_afrr_up_cap_revenue = sum([r.get('afrr_up_cap_revenue', 0) for r in results]) if enable_afrr_up_capacity else 0
            total_afrr_energy_revenue = sum([r.get('afrr_energy_revenue', 0) for r in results]) if enable_afrr_energy else 0
            total_afrr_up_energy_revenue = sum([r.get('afrr_up_energy_revenue', 0) for r in results]) if enable_afrr_up_capacity else 0
            total_afrr_revenue = total_afrr_cap_revenue + total_afrr_up_cap_revenue + total_afrr_energy_revenue + total_afrr_up_energy_revenue
            
            total_gas_only_price = gas_baseline_total
            new_total_cost = total_gas_only_price - total_savings
            savings_pct = (total_savings / total_gas_only_price) * 100 if total_gas_only_price > 0 else 0

            # --- CORRECTED LOGIC: Calculate percentage of thermal demand serviced by electricity ---
            total_elec_energy = sum([r['elec_energy'] for r in results])
            total_gas_fuel_energy = sum([r['gas_energy'] for r in results])

            thermal_from_elec_total = total_elec_energy * η
            thermal_from_gas_total = total_gas_fuel_energy * boiler_efficiency
            
            # Calculate total thermal demand from the trades data
            total_thermal_demand = sum([trade['demand_th'] * Δt for trade in all_trades])
            
            # Percentage of total thermal demand that was met by electricity
            elec_percentage = (thermal_from_elec_total / total_thermal_demand) * 100 if total_thermal_demand > 0 else 0

            # --- Display KPIs ---
            kpi_cols_1 = st.columns(4)
            kpi_cols_1[0].metric("Days Analyzed", len(results))
            
            # Show consistent metrics with highlighting based on optimization objective
            if optimization_objective == "Minimize Cost":
                kpi_cols_1[1].metric("Total Savings vs Gas Boiler", f"€{total_savings:,.0f}", 
                                    delta="PRIMARY OBJECTIVE")
            else:
                kpi_cols_1[1].metric("Total Savings vs Gas Boiler", f"€{total_savings:,.0f}")
            
            kpi_cols_1[2].metric("Total Gas Only Price", f"€{total_gas_only_price:,.0f}")
            kpi_cols_1[3].metric("New Total Cost", f"€{new_total_cost:,.0f}",delta=f"{savings_pct:.1f}%")
            
            kpi_cols_2 = st.columns(3)
            if optimization_objective == "Maximize Thermal from Electricity":
                kpi_cols_2[0].metric("Thermal from Electricity", f"{elec_percentage:.1f}%", 
                                    delta="PRIMARY OBJECTIVE")
            else:
                kpi_cols_2[0].metric("Thermal from Electricity", f"{elec_percentage:.1f}%")
            if enable_afrr_capacity or enable_afrr_energy:
                kpi_cols_2[1].metric("Total aFRR Revenue", f"€{total_afrr_revenue:,.0f}")
            else:
                kpi_cols_2[1].metric("Total aFRR Revenue", "N/A (Disabled)")

            # Calculate average daily savings for visualization
            avg_savings = total_savings / len(results) if len(results) > 0 else 0

            best_day = max(results, key=lambda x: x['savings']); worst_day = min(results, key=lambda x: x['savings'])
            col1, col2 = st.columns(2)
            with col1: st.success(f"**Best day:** {best_day['day']} (€{best_day['savings']:.2f} saved)")
            with col2: st.warning(f"**Worst day:** {worst_day['day']} (€{worst_day['savings']:.2f} saved)")

            
            fig1 = px.line(results_df, x='date', y='savings', title='Daily Savings Over Time', labels={'savings': 'Savings (€)', 'date': 'Date'})
            fig1.add_hline(y=avg_savings, line_dash="dash", annotation_text=f"Average: €{avg_savings:.2f}")
            st.plotly_chart(fig1, use_container_width=True)
            results_df['cumulative_savings'] = results_df['savings'].cumsum()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['elec_energy'], mode='lines', name='Electricity Input', fill='tonexty'))
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['gas_energy'], mode='lines', name='Gas Fuel Input', fill='tozeroy'))
            fig3.update_layout(title='Daily Energy Input Mix', xaxis_title='Date', yaxis_title='Energy (MWh)')
            st.plotly_chart(fig3, use_container_width=True)
            
            monthly_df = results_df.copy()
            monthly_df['date'] = pd.to_datetime(monthly_df['day'])
            monthly_df['month'] = monthly_df['date'].dt.to_period('M').astype(str)
            
            # --- CORRECTED ATTRIBUTION LOGIC ---
            # Calculate the savings from the DA market as the total savings minus the direct aFRR contributions.
            # This prevents double-counting the benefit of aFRR energy.
            monthly_df['DA_Savings'] = monthly_df['savings'] - monthly_df['afrr_cap_revenue'] - monthly_df.get('afrr_up_cap_revenue', 0) - monthly_df['afrr_energy_revenue'] - monthly_df.get('afrr_up_energy_revenue', 0)


            # Build columns for monthly summary based on enabled components
            summary_columns = ['DA_Savings']
            y_columns = ['DA_Savings']
            rename_dict = {}

            if enable_afrr_capacity:
                summary_columns.append('afrr_cap_revenue')
                y_columns.append('aFRR Down Capacity Revenue')
                rename_dict['afrr_cap_revenue'] = 'aFRR Down Capacity Revenue'

            if enable_afrr_up_capacity:
                summary_columns.append('afrr_up_cap_revenue')
                y_columns.append('aFRR Up Capacity Revenue')
                rename_dict['afrr_up_cap_revenue'] = 'aFRR Up Capacity Revenue'

            if enable_afrr_energy:
                summary_columns.append('afrr_energy_revenue')
                y_columns.append('aFRR Energy Revenue')
                rename_dict['afrr_energy_revenue'] = 'aFRR Energy Revenue'

            # Add aFRR Up energy revenue if Up capacity is enabled (implies Up energy)
            if enable_afrr_up_capacity and 'afrr_up_energy_revenue' in monthly_df.columns:
                summary_columns.append('afrr_up_energy_revenue')
                y_columns.append('aFRR Up Energy Revenue')
                rename_dict['afrr_up_energy_revenue'] = 'aFRR Up Energy Revenue'

            monthly_summary = monthly_df.groupby('month')[summary_columns].sum().reset_index()
            monthly_summary.rename(columns=rename_dict, inplace=True)

            fig_monthly = px.bar(monthly_summary, x='month', y=y_columns, title='Monthly Revenue & Savings Stack', height=500)
            fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Total Value (€)', legend_title='Revenue Stream')
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            if optimization_objective == "Maximize Thermal from Electricity":
                st.header("Breakeven Analysis")
                st.markdown("""
                This analysis determines the maximum Day-Ahead (DA) price at which charging the storage is still cheaper than using the gas boiler, **considering revenues from ancillary services (aFRR)**.
                """)

                # SMART ESTIMATE BASED ON INITIAL SIMULATION RESULTS
                st.subheader("Quick Estimate (Based on Last Run)")
                
                try:
                    # Calculate total DA energy and aFRR revenues from the completed simulation
                    total_elec_energy_da = trades_df['p_el_da'].sum() * Δt
                    total_afrr_revenue = (results_df['afrr_cap_revenue'].sum() + 
                                        results_df.get('afrr_up_cap_revenue', pd.Series(0)).sum() + 
                                        results_df['afrr_energy_revenue'].sum() + 
                                        results_df.get('afrr_up_energy_revenue', pd.Series(0)).sum())

                    if total_elec_energy_da > 0.01:
                        avg_afrr_subsidy = total_afrr_revenue / total_elec_energy_da
                    else:
                        avg_afrr_subsidy = 0

                    # The effective cost of producing 1 MWh of thermal energy from gas
                    cost_th_from_gas = C_gas / boiler_efficiency
                    
                    # The breakeven DA price, now including the aFRR subsidy
                    breakeven_price_analytical = (cost_th_from_gas * η) + avg_afrr_subsidy

                    st.metric(
                        label="Estimated Breakeven DA Price",
                        value=f"€{breakeven_price_analytical:.2f} / MWh",
                        help="This smart estimate includes the average aFRR revenue as a 'subsidy' to the electricity cost, based on the simulation you just ran."
                    )
                    
                    with st.expander("See Calculation Details"):
                        st.write(f"- Cost of Thermal from Gas: `€{C_gas:.2f} / {boiler_efficiency:.2f} = €{cost_th_from_gas:.2f} / MWh_th`")
                        st.write(f"- Total aFRR Revenue from Run: `€{total_afrr_revenue:,.2f}`")
                        st.write(f"- Total DA Charging from Run: `{total_elec_energy_da:,.2f} MWh`")
                        st.write(f"- Average aFRR Subsidy: `€{total_afrr_revenue:,.2f} / {total_elec_energy_da:,.2f} MWh = €{avg_afrr_subsidy:.2f} / MWh`")
                        st.write(f"- Breakeven Formula: `(Gas_Cost_th * η) + Subsidy`")
                        st.write(f"- Result: `(€{cost_th_from_gas:.2f} * {η:.2f}) + €{avg_afrr_subsidy:.2f} = €{breakeven_price_analytical:.2f}`")

                except (ZeroDivisionError, KeyError) as e:
                    st.warning("Could not calculate a smart estimate. Run the main optimization first or check results.")
                    breakeven_price_analytical = 100 # Fallback for simulation

                # --- 2. PRECISE SIMULATION (IMPROVED) ---
                with st.expander("📈 Precise Result (Full Simulation)", expanded=False):
                    st.markdown("""
                    Click below to run an iterative simulation that accounts for all dynamics to find the precise breakeven price. This is computationally intensive.
                    """)

                    # This helper function remains the same as it is the "gold standard"
                    memo = {} 
                    def run_simulation_for_price(price_cap, da_soc_limit_mwh):
                        if price_cap in memo:
                            return memo[price_cap]

                        local_soc0 = float(SOC_min)
                        daily_results = []
                        
                        for _, row in df_processed.iterrows():
                            day = row['date']
                            prices = row[price_time_cols].values
                            demand_profile = np.full(len(prices), D_th) if demand_option == 'Constant Demand' else row[demand_time_cols].values
                            T = len(prices)
                            
                            daily_afrr_cap_revenue = 0
                            blocked_intervals_for_day = [False] * T
                            if optimization_mode == 'DA + aFRR Market' and afrr_15min_mask is not None:
                                day_dt = pd.to_datetime(day).tz_localize(None)
                                day_start = day_dt.replace(hour=0, minute=0)
                                day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                                try:
                                    daily_mask = afrr_15min_mask[(afrr_15min_mask.index >= day_start) & (afrr_15min_mask.index <= day_end)]
                                    if not daily_mask.empty:
                                        mask_values = daily_mask.values; num_to_copy = min(len(mask_values), T)
                                        blocked_intervals_for_day[:num_to_copy] = [bool(x) for x in mask_values[:num_to_copy]]
                                except Exception: pass
                                if afrr_won_blocks is not None:
                                    daily_won_blocks = afrr_won_blocks[(afrr_won_blocks.index >= day_start) & (afrr_won_blocks.index < day_end)]
                                    daily_afrr_cap_revenue = daily_won_blocks["cap_payment"].sum()
                            
                            # aFRR Up capacity commitment mask
                            daily_afrr_up_cap_revenue = 0
                            afrr_up_blocked_intervals_for_day = [False] * T
                            if optimization_mode == 'DA + aFRR Market' and afrr_up_15min_mask is not None:
                                try:
                                    daily_up_mask = afrr_up_15min_mask[(afrr_up_15min_mask.index >= day_start) & (afrr_up_15min_mask.index <= day_end)]
                                    if not daily_up_mask.empty:
                                        up_mask_values = daily_up_mask.values
                                        num_to_copy = min(len(up_mask_values), T)
                                        afrr_up_blocked_intervals_for_day[:num_to_copy] = [bool(x) for x in up_mask_values[:num_to_copy]]
                                except Exception: pass
                                if afrr_up_won_blocks is not None:
                                    daily_up_won_blocks = afrr_up_won_blocks[(afrr_up_won_blocks.index >= day_start) & (afrr_up_won_blocks.index < day_end)]
                                    daily_afrr_up_cap_revenue = daily_up_won_blocks["cap_payment"].sum()

                            gas_baseline_daily = (sum(demand_profile) * Δt * C_gas) / boiler_efficiency
                            is_holiday = day in holiday_set
                            peak_restrictions_for_day = row[hlf_time_cols].values if (df_peak is not None and hlf_time_cols) else None
                            
                            daily_clearing_prices = np.zeros(T)
                            if afrr_clearing_prices_series is not None:
                                day_dt = pd.to_datetime(day).tz_localize(None); day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                                daily_prices_series = afrr_clearing_prices_series[(afrr_clearing_prices_series.index >= day_start) & (afrr_clearing_prices_series.index <= day_end)]
                                if not daily_prices_series.empty:
                                    source_data = daily_prices_series.values; num_to_copy = min(T, len(source_data))
                                    daily_clearing_prices[:num_to_copy] = source_data[:num_to_copy]
                            
                            daily_afrr_activation_profile = np.full(T, 100.0)
                            if afrr_activation_profile_series is not None:
                                day_dt = pd.to_datetime(day).tz_localize(None); day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                                daily_activation_series = afrr_activation_profile_series[(afrr_activation_profile_series.index >= day_start) & (afrr_activation_profile_series.index <= day_end)]
                                if not daily_activation_series.empty:
                                   source_data = daily_activation_series.values; num_to_copy = min(T, len(source_data))
                                   daily_afrr_activation_profile[:num_to_copy] = source_data[:num_to_copy]
                            
                            # aFRR Up energy market data
                            daily_up_clearing_prices = np.zeros(T)
                            if afrr_up_clearing_prices_series is not None:
                                day_dt = pd.to_datetime(day).tz_localize(None); day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                                daily_up_prices_series = afrr_up_clearing_prices_series[(afrr_up_clearing_prices_series.index >= day_start) & (afrr_up_clearing_prices_series.index <= day_end)]
                                if not daily_up_prices_series.empty:
                                    up_source_data = daily_up_prices_series.values; num_to_copy = min(T, len(up_source_data))
                                    daily_up_clearing_prices[:num_to_copy] = up_source_data[:num_to_copy]
                            
                            daily_afrr_up_activation_profile = np.full(T, 100.0)
                            if afrr_up_activation_profile_series is not None:
                                day_dt = pd.to_datetime(day).tz_localize(None); day_start = day_dt; day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                                daily_up_activation_series = afrr_up_activation_profile_series[(afrr_up_activation_profile_series.index >= day_start) & (afrr_up_activation_profile_series.index <= day_end)]
                                if not daily_up_activation_series.empty:
                                    up_source_data = daily_up_activation_series.values; num_to_copy = min(T, len(up_source_data))
                                    daily_afrr_up_activation_profile[:num_to_copy] = up_source_data[:num_to_copy]

                            model, p_el_da_vars, p_th_vars, p_gas_vars, soc_vars = build_da_only_model(
                                prices, demand_profile, local_soc0, η_self, boiler_efficiency,
                                peak_restrictions_for_day, is_holiday, blocked_intervals_for_day, afrr_up_blocked_intervals_for_day,
                                enable_curve=enable_power_curve, curve_params=curve_params, da_capacity_limit=da_max_capacity,
                                da_soc_limit_mwh=da_soc_limit_mwh, # <<< ADD THIS MISSING PARAMETER
                                optimization_objective="Maximize Thermal from Electricity", price_cap=price_cap
                            )
                            status = model.solve(HiGHS(msg=False))

                            if status == 1:
                                da_plan = {'p_el_da': [v.value() for v in p_el_da_vars.values()], 'p_th': [v.value() for v in p_th_vars.values()], 'p_gas': [v.value() for v in p_gas_vars.values()]}
                                if optimization_mode == "DA + aFRR Market" and enable_afrr_energy:
                                    sim_results = simulate_afrr_energy(
                                        da_plan, local_soc0, daily_clearing_prices, daily_afrr_activation_profile, η_self,
                                        enable_curve=enable_power_curve, curve_params=curve_params,
                                        Pmax_el=Pmax_el, Smax=Smax, SOC_min=SOC_min, η=η, Δt=Δt,
                                        C_grid=C_grid, C_gas=C_gas, boiler_efficiency=boiler_efficiency,
                                        soc_premium_table=soc_premium_table, afrr_energy_bid_base=afrr_energy_bid_base
                                    )
                                    final_soc, afrr_energy_net_cost = sim_results['final_soc_trajectory'][-1], sim_results['net_cost']
                                else:
                                    final_soc, afrr_energy_net_cost = soc_vars[T-1].value(), 0.0
                                elec_cost_da = sum((prices[t] + C_grid) * da_plan['p_el_da'][t] * Δt for t in range(T))
                                gas_cost = sum(C_gas * (da_plan['p_gas'][t] / boiler_efficiency) * Δt for t in range(T))
                                reported_cash_flow_cost = elec_cost_da + gas_cost + afrr_energy_net_cost - daily_afrr_cap_revenue - daily_afrr_up_cap_revenue
                                savings = gas_baseline_daily - reported_cash_flow_cost
                                daily_results.append({"savings": savings, "soc_end": final_soc})
                                local_soc0 = final_soc
                            else:
                                daily_results.append({"savings": 0, "soc_end": local_soc0})

                        total_savings_for_run = sum(r['savings'] for r in daily_results)
                        memo[price_cap] = total_savings_for_run
                        return total_savings_for_run

                    if st.button("📈 Run Precise Breakeven Simulation"):
                        with st.spinner("Finding precise breakeven price... This may take several minutes."):
                            low_price = max(0, breakeven_price_analytical - 15)
                            high_price = breakeven_price_analytical + 15
                            best_breakeven_price = None
                            
                            savings_at_high = run_simulation_for_price(high_price, da_soc_limit_mwh)
                            if savings_at_high > 0:
                                st.warning(f"The system is still profitable at €{high_price:0f}/MWh (Savings: €{savings_at_high:,.0f}). The true breakeven point is higher.")
                                best_breakeven_price = high_price
                            else:
                                for i in range(5): # 5 iterations on a smaller range is plenty
                                    mid_price = (low_price + high_price) / 2
                                    savings = run_simulation_for_price(mid_price, da_soc_limit_mwh)
                                    if savings >= 0:
                                        low_price = mid_price
                                        best_breakeven_price = mid_price
                                    else:
                                        high_price = mid_price
                                
                            if best_breakeven_price is not None:
                                final_savings = run_simulation_for_price(best_breakeven_price, da_soc_limit_mwh)
                                st.metric(
                                    label="Simulated Breakeven DA Price",
                                    value=f"€{best_breakeven_price:.1f} / MWh",
                                    delta=f"~€{best_breakeven_price - breakeven_price_analytical:.1f} vs. estimate",
                                    help=f"The full simulation finds this price cap results in total savings of ~€{final_savings:,.0f}."
                                )
                            else:
                                st.error("Could not find a breakeven point. It's likely unprofitable even at a zero charging price.")

            st.header("Sample Period Analysis")
            with st.expander("🔍 Detailed Period Analysis"):
                trades_df['date_obj'] = pd.to_datetime(trades_df['date'])
                min_analysis_date = results_df['date'].min().date()
                max_analysis_date = results_df['date'].max().date()
                st.write("Select a date range for detailed operational analysis:")
                col1, col2 = st.columns(2)
                with col1: start_date_analysis = st.date_input("Start Date",value=min_analysis_date,min_value=min_analysis_date,max_value=max_analysis_date,key="analysis_start")
                with col2:
                    default_end_date = max(min_analysis_date, start_date_analysis)
                    end_date_analysis = st.date_input("End Date",value=default_end_date,min_value=start_date_analysis,max_value=max_analysis_date,key="analysis_end")
                if start_date_analysis > end_date_analysis: st.error("Analysis start date cannot be after the end date.")
                else:
                    mask = (trades_df['date_obj'].dt.date >= start_date_analysis) & (trades_df['date_obj'].dt.date <= end_date_analysis)
                    analysis_trades = trades_df[mask].copy()
                    if not analysis_trades.empty:
                        analysis_trades['datetime'] = pd.to_datetime(analysis_trades['date'] + ' ' + analysis_trades['time'])
                        analysis_trades = analysis_trades.sort_values('datetime')
                        fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Operations & Market Prices', 'State of Charge'), vertical_spacing=0.1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['da_price'], name='DA Price', line=dict(color='blue')), row=1, col=1, secondary_y=False)
                        
                        # Stacked area chart for charging sources
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_el_da'], name='Charging (DA)', stackgroup='one', mode='lines', line=dict(width=0.5, color='rgb(0, 176, 80)'), fill='tozeroy', fillcolor='rgba(0, 176, 80, 0.5)'), row=1, col=1, secondary_y=True)
                        if enable_afrr_energy:
                            fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_el_afrr'], name='Charging (aFRR)', stackgroup='one', mode='lines', line=dict(width=0.5, color='rgb(146, 208, 80)'), fill='tonexty', fillcolor='rgba(146, 208, 80, 0.5)'), row=1, col=1, secondary_y=True)
                        
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_th_discharge'], name='Discharging (DA)', line=dict(color='red', dash='dot')), row=1, col=1, secondary_y=True)
                        
                        # Add aFRR Up discharge if enabled and data exists
                        if enable_afrr_up_capacity and 'p_th_afrr_up' in analysis_trades.columns:
                            fig4.add_trace(go.Scatter(
                                x=analysis_trades['datetime'], 
                                y=analysis_trades['p_th_afrr_up'], 
                                name='Discharging (aFRR Up)', 
                                line=dict(color='orange', dash='dashdot')
                            ), row=1, col=1, secondary_y=True)
                        
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['demand_th'], name='Thermal Demand', line=dict(color='purple', dash='longdash')), row=1, col=1, secondary_y=True)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['soc'], name='SOC', line=dict(color='orange')), row=2, col=1)
                        
                        # Only show aFRR blocks if capacity market is enabled
                        if enable_afrr_capacity:
                            in_afrr_block = False; start_block_time = None
                            for index, row in analysis_trades.iterrows():
                                if row['is_in_afrr_market'] and not in_afrr_block:
                                    in_afrr_block = True; start_block_time = row['datetime']
                                elif not row['is_in_afrr_market'] and in_afrr_block:
                                    in_afrr_block = False; fig4.add_vrect(x0=start_block_time, x1=row['datetime'], annotation_text="aFRR Won", annotation_position="top left", fillcolor="grey", opacity=0.25, line_width=0, layer="below")
                            if in_afrr_block: fig4.add_vrect(x0=start_block_time, x1=analysis_trades['datetime'].iloc[-1], annotation_text="aFRR Won", annotation_position="top left", fillcolor="grey", opacity=0.25, line_width=0, layer="below")
                        fig4.update_layout(height=800, title_text=f"Detailed Analysis from {start_date_analysis} to {end_date_analysis}"); fig4.update_yaxes(title_text="Price (€/MWh)", row=1, col=1, secondary_y=False); fig4.update_yaxes(title_text="Power (MW)", row=1, col=1, secondary_y=True, showgrid=False); fig4.update_yaxes(title_text="Storage (MWh)", row=2, col=1)
                        st.plotly_chart(fig4, use_container_width=True)
                    else: st.warning("No data available for the selected date range.")

            st.header("💾 Download Results")
            if not trades_df.empty:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr('thermal_storage_trades.csv', trades_df.to_csv(index=False))
                    zip_file.writestr('thermal_storage_daily.csv', results_df.to_csv(index=False))
                    
                    params_text = (
                        f"Thermal Storage Optimization Parameters\n"
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"Demand Option: {demand_option}\n"
                        f"Market Participation:\n"
                        f"- Day-Ahead Market: Enabled\n"
                        f"- aFRR Capacity Market: {'Enabled' if enable_afrr_capacity else 'Disabled'}\n"
                        f"- aFRR Energy Market: {'Enabled' if enable_afrr_energy else 'Disabled'}\n\n"
                        f"System Parameters:\n"
                        f"- Time Interval: {Δt} hours\n"
                        f"- Max Electrical Power: {Pmax_el} MW\n"
                        f"- Max Thermal Power: {Pmax_th} MW\n"
                        f"- Max Storage Capacity: {Smax} MWh\n"
                        f"- Min Storage Level: {SOC_min} MWh\n"
                        f"- Charging Efficiency: {η}\n"
                        f"- Self-Discharge Rate: {self_discharge_daily} % per day\n"
                        f"- Gas Boiler Efficiency: {boiler_efficiency_pct} %\n\n"
                        f"Economic Parameters:\n"
                        f"- Grid Charges: {C_grid} €/MWh\n"
                        f"- Gas Price: {C_gas} €/MWh\n"
                        f"- Terminal Value: {terminal_value} €/MWh\n\n"
                        f"Market Capacity Allocation\n"
                        f"- Max DA Market Capacity: {da_max_capacity} MW\n"
                    )

                    if enable_afrr_capacity or enable_afrr_energy:
                        params_text += "aFRR Market Parameters:\n"
                        if enable_afrr_capacity:
                            params_text += f"- aFRR Capacity Bid Size: {afrr_bid_mw} MW\n"
                            params_text += f"- aFRR Capacity Bid Price: {afrr_bid_price} €/MW\n"
                            params_text += f"- Bid Strategy: {afrr_bid_strategy}\n"
                        if enable_afrr_energy:
                            params_text += f"- aFRR Energy Base Bid: {afrr_energy_bid_base} €/MWh\n"
                            params_text += f"- SOC Premium Table: {soc_premium_table}\n"
                        params_text += "\n"
                    if enable_power_curve:
                        params_text += (
                            f"Power-Capacity Curve (Enabled):\n"
                            f"- Charge Taper Start SOC: {charge_taper_soc_pct} %\n"
                            f"- Charge Power at 100% SOC: {charge_power_at_full_pct} % of Pmax\n"
                            f"- Discharge Taper Start SOC: {discharge_taper_soc_pct} %\n"
                            f"- Discharge Power at Min SOC: {discharge_power_at_empty_pct} % of Pmax\n\n"
                        )
                    params_text += (
                        f"Results Summary:\n"
                        f"- Days Analyzed: {len(results)}\n"
                        f"- Average Daily Savings: €{avg_savings:.2f} ({savings_pct:.1f}%)\n"
                        f"- Total Savings: €{total_savings:.2f}\n"
                        f"- Thermal Demand Serviced by Electricity: {elec_percentage:.1f}%\n"
                    )

                    if enable_afrr_capacity or enable_afrr_up_capacity or enable_afrr_energy:
                        params_text += f"- Total aFRR Revenue: €{total_afrr_revenue:.2f}\n"
                        if enable_afrr_capacity:
                            params_text += f"- aFRR Down Capacity Revenue: €{total_afrr_cap_revenue:.2f}\n"
                        if enable_afrr_up_capacity:
                            params_text += f"- aFRR Up Capacity Revenue: €{total_afrr_up_cap_revenue:.2f}\n"
                        if enable_afrr_energy:
                            params_text += f"- aFRR Energy Revenue: €{total_afrr_energy_revenue:.2f}\n"
                        if enable_afrr_up_capacity:
                            params_text += f"- aFRR Up Energy Revenue: €{total_afrr_up_energy_revenue:.2f}\n"
                    
                    zip_file.writestr('parameters_and_summary.txt', params_text)
                zip_buffer.seek(0)
                st.download_button(label="📥 Download All Results (ZIP)", data=zip_buffer.getvalue(), file_name=f"thermal_storage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
                col1, col2 = st.columns(2)
                with col1: st.download_button(label="📊 Download Detailed Trades (CSV)", data=trades_df.to_csv(index=False), file_name='thermal_storage_trades.csv', mime='text/csv')
                with col2: st.download_button(label="📅 Download Daily Summary (CSV)", data=results_df.to_csv(index=False), file_name='thermal_storage_daily.csv', mime='text/csv')
        else:
            st.info("🔍 Run optimization to see results and download options.")
else:
    st.info("👈 Please upload a CSV file or configure API access using the sidebar to begin.")
    with st.expander("📋 Data Source Guide"):
        st.markdown("""
        This app supports three data sources for Day-Ahead prices and several for other inputs.

        ---
        ### 📁 File Upload Formats

        #### 1. Price Data Format
        - **Long Format (default):** A CSV with a datetime column (e.g., `Date (CET)`) and a value column (e.g., `Day Ahead Price`). Ensure "Transform data" is checked.
        - **Wide Format:** A CSV with a 'date' column and 96 columns for each 15-min interval (`00:00:00`, etc.). Uncheck "Transform data".

        ---
        #### 2. Customer Demand Data Format (Optional)
        If you select "Upload Demand Profile", the file must be in **long format**.
        - Column 1: `Date (CET)` (datetime information)
        - Column 2: `MW-th` (thermal demand value)

        ---
        #### 3. Peak Period Restriction Data Format (Optional)
        If you select "Upload CSV File" for Peak Period Restrictions, the file must be in **long format**.
        - Column 1: `Date (CET)` (datetime information)
        - Column 2: `Is HLF` (1 for restricted, 0 for not)

        ---
        #### 4. aFRR Capacity Auction Data Format (Optional)
        If participating in the aFRR market, upload a **long format** CSV.
        - Column 1: `Date (CET)` (datetime information)
        - Column 2: A column containing "price" in its name (e.g., `aFRR Down Capacity Price`)

        ---
        #### 5. aFRR Energy Market Data Format (Optional)
        If participating in the aFRR market, upload a **long format** CSV.
        - Column 1: `Date` (datetime information)
        - Column 2: `Revenue` (the energy price in €/MWh)
        - Column 3: `Activation` (the activation rate as a decimal)

        ---
        #### 6. aFRR Dynamic Bid Price Data Format (Optional)
        If using the "Dynamic Bids" strategy, upload a **long format** CSV.
        - Column 1: `Date (CET)` (datetime information)
        - Column 2: `Bid Price` (your bid for that interval)
        """)

# Footer
st.markdown("---")