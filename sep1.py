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
st.markdown("""
This application optimizes thermal storage operations to minimize energy costs by:
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
            return None, None
        afrr_blocks = pd.merge(afrr_blocks, _df_afrr_bids, left_index=True, right_index=True, how='left')
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

# --- START: aFRR UI LOGIC ---
afrr_capacity_file = None
afrr_energy_file = None
afrr_dynamic_bids_file = None
use_builtin_afrr = False
afrr_bid_price = 36.0
afrr_bid_mw = 2.0
afrr_energy_bid_base = 36.0

# Individual checkboxes for aFRR components - only show when aFRR is selected
enable_afrr_capacity = False
enable_afrr_energy = False

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

    # Show aFRR data section if either component is enabled
    if enable_afrr_capacity or enable_afrr_energy:
        st.sidebar.subheader("⚡ aFRR Market Data")
        if data_source == "Use Built-in EPEX 2024 Data":
            st.sidebar.info("📊 Using built-in aFRR 2024 data")
            if enable_afrr_capacity:
                st.sidebar.markdown("**Capacity Auction:** `aFRRprices.csv`")
            if enable_afrr_energy:
                st.sidebar.markdown("**Energy Market:** `aFRRenergylight.csv`")
            use_builtin_afrr = True
        else:
            if enable_afrr_capacity:
                afrr_capacity_file = st.sidebar.file_uploader("Upload aFRR Capacity Auction Data (CSV)", type=['csv'], help="Upload a CSV with 4-hour capacity clearing prices (e.g., aFRRprices.csv).")
            if enable_afrr_energy:
                afrr_energy_file = st.sidebar.file_uploader("Upload aFRR Energy Market Data (CSV)", type=['csv'], help="Upload a CSV with 15-min energy revenue and activation data (e.g., aFRRenergylight.csv).")
            use_builtin_afrr = False
# --- END: aFRR UI LOGIC ---

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
    st.markdown("Set costs, prices, and strategic bid values.")
    C_grid = st.number_input("Grid Charges (€/MWh)", value=30.0, min_value=0.0, max_value=100.0, step=1.0)
    C_gas = st.number_input("Gas Price (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)
    terminal_value = st.number_input("Terminal Value (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0, help="Estimated value of energy remaining in storage at the end of the optimization period.")

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

        # aFRR Energy Market parameters
        if enable_afrr_energy:
            st.markdown("---")
            st.markdown("**aFRR Energy Market**")
            afrr_energy_bid_base = st.number_input(
                "aFRR Energy Base Bid (€/MWh)",
                value=36.0,
                min_value=0.0,
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
                        0.0: 0.0, 1.0: 0.0, 2.0: 0.0, 3.0: 10.0,
                        4.0: 40.0, 7.5: 300.0, 8.0: 10000.0
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
        st.success("✅ Using cached price data from previous fetch!")
    else:
        # ... (rest of price data loading logic remains the same)
        df_price = None
        if transform_data:
            st.info("New data source detected. Running ETL transformation...")
            with st.spinner("Transforming data from long to wide format..."):
                try:
                    if uploaded_file is not None:
                        df_price = etl_long_to_wide(input_source=uploaded_file, datetime_column_name='Date (CET)', value_column_name='Day Ahead Price')
                    elif api_config is not None:
                        df_price = etl_long_to_wide(use_api=True, api_config=api_config)
                    elif use_builtin_data:
                        df_price = etl_long_to_wide(input_source="idprices-epex2024.csv", datetime_column_name='Date (CET)', value_column_name='Day Ahead Price')
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
            st.success("✅ Using cached demand data!")
        else:
            with st.spinner("Transforming demand data..."):
                try:
                    df_demand = etl_long_to_wide(input_source=demand_file, datetime_column_name='Date (CET)', value_column_name='MW-th')
                    st.success("✅ Demand ETL transformation successful!")
                    st.session_state['cached_df_demand'] = df_demand.copy()
                    st.session_state['cached_demand_config_key'] = demand_config_key
                except Exception as e: st.error(f"❌ Demand file processing failed: {e}"); st.stop()

    df_peak = None
    if peak_period_option == 'Use Built-in Example Data':
        peak_config_key = "peak_builtin_example"
        if 'cached_peak_config_key' in st.session_state and 'cached_df_peak' in st.session_state and st.session_state['cached_peak_config_key'] == peak_config_key:
            df_peak = st.session_state['cached_df_peak']
            st.success("✅ Using cached built-in peak restriction data!")
        else:
            with st.spinner("Loading built-in peak restriction example data..."):
                try:
                    df_peak = etl_long_to_wide(input_source="Example_Peak Restriktions.csv", datetime_column_name='Date (CET)', value_column_name='Is HLF')
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
                    st.success("✅ Peak restriction data cleaned and ETL successful!")
                    st.session_state['cached_df_peak'] = df_peak.copy()
                    st.session_state['cached_peak_config_key'] = peak_config_key
                except Exception as e: st.error(f"❌ A critical error occurred while processing the peak restriction file: {e}"); st.stop()

    # SEPARATE aFRR DATA LOADING (FIX) ---
    df_afrr_capacity = None
    df_afrr_energy = None
    df_afrr_bids = None

    # Load Capacity Data if aFRR is selected and capacity is enabled
    if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity:
        if use_builtin_afrr:
            capacity_config_key = "afrr_capacity_builtin"
            if 'cached_afrr_capacity_config_key' in st.session_state and st.session_state.get('cached_afrr_capacity_config_key') == capacity_config_key:
                df_afrr_capacity = st.session_state.get('cached_df_afrr_capacity')
                st.success("✅ Using cached aFRR capacity data!")
            else:
                try:
                    df_afrr_capacity = pd.read_csv("aFRRprices.csv")
                    df_afrr_capacity['datetime'] = pd.to_datetime(df_afrr_capacity['Date (CET)'])
                    df_afrr_capacity = df_afrr_capacity.set_index('datetime')
                    st.session_state['cached_df_afrr_capacity'] = df_afrr_capacity.copy()
                    st.session_state['cached_afrr_capacity_config_key'] = capacity_config_key
                    st.success("✅ Built-in aFRR capacity data loaded!")
                except Exception as e: st.error(f"❌ Failed to load built-in aFRRprices.csv: {e}")
        elif afrr_capacity_file:
            capacity_config_key = f"afrr_capacity_{afrr_capacity_file.name}_{hash(afrr_capacity_file.getvalue())}"
            # ... caching logic for uploaded capacity file ...
            try:
                df_afrr_capacity = pd.read_csv(afrr_capacity_file)
                df_afrr_capacity['datetime'] = pd.to_datetime(df_afrr_capacity['Date (CET)'])
                df_afrr_capacity = df_afrr_capacity.set_index('datetime')
                st.success("✅ aFRR capacity data loaded!")
            except Exception as e: st.error(f"❌ Failed to process aFRR capacity file: {e}")
        else:
            st.warning("Please provide aFRR Capacity Auction data.")

    # Load Energy Data if aFRR is selected and energy is enabled
    if optimization_mode == "DA + aFRR Market" and enable_afrr_energy:
        if use_builtin_afrr:
            energy_config_key = "afrr_energy_builtin"
            if 'cached_afrr_energy_config_key' in st.session_state and st.session_state.get('cached_afrr_energy_config_key') == energy_config_key:
                df_afrr_energy = st.session_state.get('cached_df_afrr_energy')
                st.success("✅ Using cached aFRR energy data!")
            else:
                try:
                    df_afrr_energy = pd.read_csv("aFRRenergylight.csv")
                    st.session_state['cached_df_afrr_energy'] = df_afrr_energy.copy()
                    st.session_state['cached_afrr_energy_config_key'] = energy_config_key
                    st.success("✅ Built-in aFRR energy data loaded!")
                except Exception as e: st.error(f"❌ Failed to load built-in aFRRenergylight.csv: {e}")
        elif afrr_energy_file:
            energy_config_key = f"afrr_energy_{afrr_energy_file.name}_{hash(afrr_energy_file.getvalue())}"
            # ... caching logic for uploaded energy file ...
            try:
                df_afrr_energy = pd.read_csv(afrr_energy_file)
                st.success("✅ aFRR energy data loaded!")
            except Exception as e: st.error(f"❌ Failed to process aFRR energy file: {e}")
        else:
            st.warning("Please provide aFRR Energy Market data.")

    # Load Dynamic Bids Data if aFRR is selected, capacity is enabled, and dynamic strategy is selected
    if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity and afrr_bid_strategy == 'Dynamic Bids (from CSV)':
        if afrr_dynamic_bids_file is None:
            st.warning("Dynamic bid strategy selected. Please upload your bid price CSV file."); st.stop()

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
                df_processed = pd.merge(df_filtered, df_demand, on='date', how='inner', suffixes=('_price', '_demand'))
                if len(df_processed) == 0: st.error("❌ No matching dates found between price and demand data."); st.stop()
            if df_peak is not None:
                peak_time_cols = [col for col in df_peak.columns if col != 'date']
                rename_map = {col: f"{col}_hlf" for col in peak_time_cols}
                df_peak_renamed = df_peak.rename(columns=rename_map)
                df_processed = pd.merge(df_processed, df_peak_renamed, on='date', how='left')
                hlf_time_cols = [col for col in df_processed.columns if col.endswith('_hlf')]
                if hlf_time_cols: df_processed[hlf_time_cols] = df_processed[hlf_time_cols].fillna(0)
            else: hlf_time_cols = []
            st.success(f"✅ Ready to analyze {len(df_processed)} days of data.")
            with st.spinner("Cleaning data..."):
                for col in df_processed.columns:
                    if col != 'date': df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan).interpolate(method='linear', limit_direction='both').fillna(df_processed[col].median())
            st.success("✅ Data cleaning completed")

            afrr_15min_mask, afrr_won_blocks = None, None
            if optimization_mode == "DA + aFRR Market" and enable_afrr_capacity and df_afrr_capacity is not None:
                st.header("⚡ aFRR Capacity Auction Pre-computation")
                with st.spinner("Analyzing aFRR capacity bids (using cache if available)..."):
                    afrr_won_blocks, afrr_15min_mask = precompute_afrr_auction(
                        _df_afrr_capacity=df_afrr_capacity, afrr_bid_strategy=afrr_bid_strategy,
                        static_bid_price=afrr_bid_price, _df_afrr_bids=df_afrr_bids,
                        _df_peak=df_peak, holiday_set=holiday_set,
                        static_hochlast_intervals=hochlast_intervals_static, bid_mw=afrr_bid_mw
                    )
                    
                if afrr_won_blocks is not None and not afrr_won_blocks.empty:
                    st.success(f"✅ Pre-computation complete. Found {len(afrr_won_blocks)} won aFRR blocks.")
                elif afrr_won_blocks is not None:
                     st.success("✅ Pre-computation complete. No aFRR blocks were won.")
                else:
                    st.warning("⚠️ Could not perform aFRR pre-computation. Check aFRR capacity data sources.")

            afrr_clearing_prices_series = None
            afrr_activation_profile_series = None
            
            if optimization_mode == "DA + aFRR Market" and enable_afrr_energy and df_afrr_energy is not None:
                st.header("⚡ aFRR Energy Market Analysis")
                with st.spinner("Extracting aFRR energy prices and activation profile..."):
                    afrr_clearing_prices_series = extract_afrr_clearing_prices(df_afrr_energy)
                    afrr_activation_profile_series = extract_afrr_activation_profile(df_afrr_energy)
                    
                    if afrr_clearing_prices_series is not None:
                        st.success(f"✅ Extracted {len(afrr_clearing_prices_series)} clearing price points")
                    
                    if afrr_activation_profile_series is not None:
                        st.success(f"✅ Extracted {len(afrr_activation_profile_series)} activation profile points")
                        avg_activation = afrr_activation_profile_series.mean()
                        st.info(f"📊 Average activation rate: {avg_activation:.1f}%")
                    else:
                        st.info("📊 Using default 100% activation profile")

            if df_demand is not None:
                price_time_cols = [col for col in df_processed.columns if col.endswith('_price')]
                demand_time_cols = [col for col in df_processed.columns if col.endswith('_demand')]
            else:
                price_time_cols = [col for col in df_processed.columns if col != 'date' and not col.endswith('_hlf')]
                demand_time_cols = []
            

            def build_thermal_model(prices, demand_profile, soc0, η_self, boiler_eff, 
                                    peak_restrictions=None, is_holiday=False, afrr_commitment_mask=None,
                                    afrr_activation_profile=None, afrr_clearing_prices=None,
                                    soc_premium_table=None, afrr_energy_bid_base=36.0,
                                    enable_curve=False, curve_params=None):
                T = len(prices)
                model = LpProblem("Thermal_Storage_Optimization", LpMinimize)
                
                # Decision variables
                p_el_da = LpVariable.dicts("p_el_da", range(T), lowBound=0)
                p_th = LpVariable.dicts("p_th", range(T), lowBound=0, upBound=Pmax_th)
                p_gas = LpVariable.dicts("p_gas", range(T), lowBound=0)
                soc = LpVariable.dicts("soc", range(T), lowBound=SOC_min, upBound=Smax)
                
                # aFRR energy variables (no binary variable needed anymore)
                p_el_afrr = LpVariable.dicts("p_el_afrr", range(T), lowBound=0)

                # Iterate through the timeline in hourly chunks (4 intervals of 15 mins)
                for hour_idx, t in enumerate(range(0, T, 4)):
                    # Ensure we don't go out of bounds if T is not a multiple of 4
                    if t + 3 < T:
                        # Constrain the power in the next 3 intervals to be the same as the first one
                        model += p_el_da[t+1] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int1"
                        model += p_el_da[t+2] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int2"
                        model += p_el_da[t+3] == p_el_da[t], f"HourlyPower_DA_H{hour_idx}_Int3"
               
                # Objective function
                da_costs = lpSum([(prices[t] + C_grid) * p_el_da[t] * Δt for t in range(T)])
                gas_costs = lpSum([(C_gas / boiler_eff) * p_gas[t] * Δt for t in range(T)])
                
                # This term handles both costs (positive clearing price) and revenues (negative clearing price) and correctly includes grid charges.
                afrr_energy_costs = lpSum([
                    ( (safe_float_convert(afrr_clearing_prices[t]) if afrr_clearing_prices is not None and t < len(afrr_clearing_prices) else 0.0) + C_grid) 
                    * p_el_afrr[t] * Δt 
                    for t in range(T)
                ])
                
                model += da_costs + gas_costs + afrr_energy_costs - terminal_value * soc[T-1]
                
                # Power curve parameters if enabled
                if enable_curve and curve_params is not None:
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
                    else: m_charge, b_charge = None, None
                        
                    if (soc_discharge_knee - SOC_min) > 1e-6:
                        power_at_empty = p_min_discharge_frac * Pmax_th
                        m_discharge = (Pmax_th - power_at_empty) / (soc_discharge_knee - SOC_min)
                        b_discharge = power_at_empty - m_discharge * SOC_min
                    else: m_discharge, b_discharge = None, None
                
                # Main constraints
                for t in range(T):
                    # Thermal balance
                    model += p_th[t] + p_gas[t] == demand_profile[t]
                    
                    # Combined power constraint
                    model += p_el_da[t] + p_el_afrr[t] <= Pmax_el
                    
                    # AFRR ENERGY MARKET PARTICIPATION ---
                    if afrr_clearing_prices is not None and soc_premium_table is not None:
                        clearing_price = safe_float_convert(afrr_clearing_prices[t]) if t < len(afrr_clearing_prices) else 0.0
                        activation_frac = (safe_float_convert(afrr_activation_profile[t]) / 100.0) if afrr_activation_profile is not None and t < len(afrr_activation_profile) else 1.0

                        # estimated_soc = float(min(soc0 + t * 0.1, Smax)) # Keeping original simple estimate for linearity
                        if t == 0:
                            estimated_soc = float(soc0)
                        else:
                            # Use the actual SOC from previous interval
                            # This requires accessing the variable's value, which is tricky in LP
                            # A better approach is to use soc0 for the entire day's planning
                            estimated_soc = float(soc0)  # Use day-start SOC for all intervals
                        premium = 0.0
                        soc_levels = sorted([float(k) for k in soc_premium_table.keys()])
                        for i in range(len(soc_levels) - 1):
                            if soc_levels[i] <= estimated_soc <= soc_levels[i+1]:
                                soc_range = soc_levels[i+1] - soc_levels[i]
                                if soc_range > 0:
                                    weight = (estimated_soc - soc_levels[i]) / soc_range
                                    premium = (1 - weight) * float(soc_premium_table[soc_levels[i]]) + weight * float(soc_premium_table[soc_levels[i+1]])
                                else:
                                    premium = float(soc_premium_table[soc_levels[i]])
                                break
                        if estimated_soc >= soc_levels[-1]:
                            premium = float(soc_premium_table[soc_levels[-1]])
                        
                        # Effective bid increases with SOC to make us less competitive when storage is full
                        effective_bid = float(afrr_energy_bid_base) - premium
                        
                        # We "win" (get activated) if the market clearing price is less than or equal to our bid.
                        # This means the system will pay us (negative price) or we pay less than our max willingness.
                        if clearing_price <= effective_bid:
                            # MUST charge exactly at the activation rate
                            required_power = Pmax_el * activation_frac
                            model += p_el_afrr[t] == required_power  # EQUALITY constraint
                        else:
                            model += p_el_afrr[t] == 0
                    else:
                        # aFRR energy market is disabled
                        model += p_el_afrr[t] == 0
                    
                    # DA market restrictions
                    is_da_restricted = False
                    if not is_holiday:
                        if t in hochlast_intervals_static: is_da_restricted = True
                        elif peak_restrictions is not None and len(peak_restrictions) > t and peak_restrictions[t] == 1: is_da_restricted = True
                    
                    if afrr_commitment_mask and len(afrr_commitment_mask) > t and afrr_commitment_mask[t]:
                        is_da_restricted = True
                    
                    if is_da_restricted:
                        model += p_el_da[t] == 0
                    
                    # Apply power curves if enabled
                    if enable_curve and curve_params is not None:
                        prev_soc = soc[t-1] if t > 0 else soc0
                        total_charge = p_el_da[t] + p_el_afrr[t]
                        
                        if m_charge is not None:
                            model += total_charge <= Pmax_el, f"ChargeCurve_Flat_{t}"
                            model += total_charge <= m_charge * prev_soc + b_charge, f"ChargeCurve_Sloped_{t}"
                        
                        if m_discharge is not None:
                            model += p_th[t] <= Pmax_th, f"DischargeCurve_Flat_{t}"
                            model += p_th[t] <= m_discharge * prev_soc + b_discharge, f"DischargeCurve_Sloped_{t}"
                    
                    # SOC dynamics
                    total_charging = p_el_da[t] + p_el_afrr[t]
                    if t == 0:
                        model += soc[t] == soc0 * η_self + η * total_charging * Δt - p_th[t] * Δt
                    else:
                        model += soc[t] == soc[t-1] * η_self + η * total_charging * Δt - p_th[t] * Δt
                
                # The binary variable is no longer used, return None in its place
                return model, p_el_da, p_th, p_gas, soc, p_el_afrr, None

            if st.button("🚀 Run Optimization", type="primary"):
                if 'results' in st.session_state: del st.session_state['results']
                if 'all_trades' in st.session_state: del st.session_state['all_trades']
                if 'gas_baseline' in st.session_state: del st.session_state['gas_baseline']

                progress_bar = st.progress(0); status_text = st.empty()
                soc0 = float(SOC_min)
                results, all_trades, all_baselines = [], [], []
                η_self = (1 - self_discharge_daily / 100) ** (Δt / 24)
                
                curve_params = None
                if enable_power_curve:
                    curve_params = {
                        'charge_start_pct': charge_taper_soc_pct,
                        'charge_end_pct': charge_power_at_full_pct,
                        'discharge_start_pct': discharge_taper_soc_pct,
                        'discharge_end_pct': discharge_power_at_empty_pct
                    }

                for idx, (_, row) in enumerate(df_processed.iterrows()):
                    progress_bar.progress((idx + 1) / len(df_processed))
                    day = row['date']; status_text.text(f"Processing day {idx + 1}/{len(df_processed)}: {day}")
                    prices = row[price_time_cols].values
                    demand_profile = np.full(len(prices), D_th) if demand_option == 'Constant Demand' else row[demand_time_cols].values

                    daily_afrr_cap_revenue = 0
                    blocked_intervals_for_day = [False] * len(prices)
                    if optimization_mode == 'DA + aFRR Market' and afrr_15min_mask is not None:
                        day_dt = pd.to_datetime(day)
                        if day_dt.tz is not None: day_dt = day_dt.tz_localize(None)
                        day_start = day_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                        try:
                            daily_mask = afrr_15min_mask[(afrr_15min_mask.index >= day_start) & (afrr_15min_mask.index <= day_end)]
                            if not daily_mask.empty:
                                mask_values = daily_mask.values
                                if len(mask_values) >= len(prices): blocked_intervals_for_day = [bool(x) for x in mask_values[:len(prices)]]
                                else: blocked_intervals_for_day = [bool(x) for x in mask_values] + [False] * (len(prices) - len(mask_values))
                        except Exception as e: st.warning(f"aFRR mask extraction failed for {day}: {e}")
                        
                        if afrr_won_blocks is not None:
                            daily_won_blocks = afrr_won_blocks[(afrr_won_blocks.index >= day_start) & (afrr_won_blocks.index < day_end)]
                            daily_afrr_cap_revenue = daily_won_blocks["cap_payment"].sum()

                    if len(prices) != len(demand_profile): st.warning(f"Skipping day {day} due to mismatched data length."); continue

                    gas_baseline_daily = (sum(demand_profile) * Δt * C_gas) / boiler_efficiency
                    all_baselines.append(gas_baseline_daily)
                    is_holiday = day in holiday_set
                    peak_restrictions_for_day = row[hlf_time_cols].values if (df_peak is not None and hlf_time_cols) else None
                    
                    day_dt = pd.to_datetime(day).tz_localize(None)
                    day_start = day_dt
                    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

                    daily_clearing_prices = None
                    if afrr_clearing_prices_series is not None:
                        daily_clearing_prices = np.zeros(len(prices))
                        daily_prices_series = afrr_clearing_prices_series[(afrr_clearing_prices_series.index >= day_start) & (afrr_clearing_prices_series.index <= day_end)]
                        if not daily_prices_series.empty:
                            source_data = daily_prices_series.values
                            num_to_copy = min(len(prices), len(source_data))
                            daily_clearing_prices[:num_to_copy] = source_data[:num_to_copy]

                    daily_afrr_activation_profile = np.full(len(prices), 100.0)
                    if afrr_activation_profile_series is not None:
                        daily_activation_series = afrr_activation_profile_series[(afrr_activation_profile_series.index >= day_start) & (afrr_activation_profile_series.index <= day_end)]
                        if not daily_activation_series.empty:
                            source_data = daily_activation_series.values
                            num_to_copy = min(len(prices), len(source_data))
                            daily_afrr_activation_profile[:num_to_copy] = source_data[:num_to_copy]
                    
                    # The build_thermal_model function no longer returns afrr_energy_won
                    model, p_el_da, p_th, p_gas, soc, p_el_afrr, _ = build_thermal_model(
                        prices, demand_profile, soc0, η_self, boiler_efficiency,
                        peak_restrictions_for_day, is_holiday, blocked_intervals_for_day,
                        daily_afrr_activation_profile, daily_clearing_prices,
                        soc_premium_table if enable_afrr_energy else None,
                        afrr_energy_bid_base if enable_afrr_energy else 36.0,
                        enable_curve=enable_power_curve, curve_params=curve_params
                    )
                    status = model.solve(PULP_CBC_CMD(msg=False))
                    if status == 1:
                        soc_end = soc[len(prices)-1].value()
                        elec_cost_da = sum((prices[t] + C_grid) * p_el_da[t].value() * Δt for t in range(len(prices)))
                        gas_cost = sum(C_gas * (p_gas[t].value() / boiler_efficiency) * Δt for t in range(len(prices)))

                        # --- DETAILED aFRR Energy Savings Analysis ---
                        afrr_energy_details = []
                        for t in range(len(prices)):
                            if p_el_afrr[t].value() > 0.01:  # Only count when actually charging
                                clearing_price = safe_float_convert(daily_clearing_prices[t]) if daily_clearing_prices is not None else 0.0
                                power = p_el_afrr[t].value()
                                
                                # Cost to charge via aFRR (including grid charges)
                                afrr_cost_per_mwh = clearing_price + C_grid
                                thermal_via_afrr = power * Δt * η  # Thermal energy gained
                                cost_for_thermal = (afrr_cost_per_mwh * power * Δt) / η  # Cost per MWh thermal
                                
                                # Gas alternative cost for same thermal energy
                                gas_alternative_cost = thermal_via_afrr * (C_gas / boiler_efficiency)
                                
                                # Savings compared to gas
                                savings_vs_gas = gas_alternative_cost - (afrr_cost_per_mwh * power * Δt)
                                afrr_energy_details.append({
                                    'interval': t,
                                    'clearing_price': clearing_price,
                                    'power': power,
                                    'thermal_gained': thermal_via_afrr,
                                    'cost': afrr_cost_per_mwh * power * Δt,
                                    'gas_alternative': gas_alternative_cost,
                                    'savings': savings_vs_gas
                                })

                        # Total aFRR contribution to savings
                        afrr_energy_savings = sum([d['savings'] for d in afrr_energy_details])

                        # For reporting, show it as positive contribution to savings
                        afrr_energy_revenue = afrr_energy_savings  # This is what you want to show
                        
                        # Total cost includes DA, gas, and the aFRR energy cost, minus capacity revenue
                        # Note: afrr_energy_revenue is now the savings, so we subtract it from costs
                        afrr_energy_cost = sum([d['cost'] for d in afrr_energy_details]) if afrr_energy_details else 0
                        reported_cash_flow_cost = elec_cost_da + gas_cost + afrr_energy_cost - daily_afrr_cap_revenue
                        savings = gas_baseline_daily - reported_cash_flow_cost
                        
                        elec_energy_da = sum([p_el_da[t].value() * Δt for t in range(len(prices))])
                        elec_energy_afrr = sum([p_el_afrr[t].value() * Δt for t in range(len(prices))])
                        elec_energy = elec_energy_da + elec_energy_afrr
                        gas_fuel_energy = sum([(p_gas[t].value() / boiler_efficiency) * Δt for t in range(len(prices))])
                        
                        for t in range(len(prices)):
                            interval_hour, interval_min = divmod(t * 15, 60); time_str = f"{interval_hour:02d}:{interval_min:02d}:00"
                            gas_cost_interval_val = C_gas * (p_gas[t].value() / boiler_efficiency) * Δt
                            elec_cost_da_interval = (prices[t] + C_grid) * p_el_da[t].value() * Δt
                            # Calculate aFRR cost for this interval (consistent with the detailed analysis above)
                            afrr_net_cost_interval = 0
                            if p_el_afrr[t].value() > 0.01:
                                clearing_price = safe_float_convert(daily_clearing_prices[t]) if daily_clearing_prices is not None else 0.0
                                afrr_net_cost_interval = (clearing_price + C_grid) * p_el_afrr[t].value() * Δt
                            
                            total_elec_power = p_el_da[t].value() + p_el_afrr[t].value()
                            is_static_restricted = t in hochlast_intervals_static and not is_holiday
                            is_dynamic_restricted = (peak_restrictions_for_day is not None and len(peak_restrictions_for_day) > t and peak_restrictions_for_day[t] == 1 and not is_holiday)
                            is_restricted = is_static_restricted or is_dynamic_restricted
                            
                            trade_record = {
                                'date': day, 'time': time_str, 'interval': t, 'da_price': prices[t],
                                'total_elec_cost': prices[t] + C_grid, 'p_el_heater': total_elec_power,
                                'p_el_da': p_el_da[t].value(), 'p_el_afrr': p_el_afrr[t].value() if enable_afrr_energy else 0.0,
                                'p_th_discharge': p_th[t].value(), 'p_gas_backup': p_gas[t].value(),
                                'soc': soc[t].value(), 'elec_cost_interval': elec_cost_da_interval + afrr_net_cost_interval,
                                'gas_cost_interval': gas_cost_interval_val,
                                'total_cost_interval': elec_cost_da_interval + afrr_net_cost_interval + gas_cost_interval_val,
                                'is_hochlast': is_restricted, 'is_holiday': is_holiday,
                                'is_charging': total_elec_power > 0.01, 'is_discharging': p_th[t].value() > 0.01,
                                'using_gas': p_gas[t].value() > 0.01, 'demand_th': demand_profile[t],
                                'is_in_afrr_market': (blocked_intervals_for_day[t] if t < len(blocked_intervals_for_day) else False) if enable_afrr_capacity else False,
                                'afrr_energy_won': 1 if p_el_afrr[t].value() > 0.01 else 0 # Simplified win indicator
                            }
                            all_trades.append(trade_record)
                        
                        soc0 = float(soc_end)
                        
                        results.append({
                            "day": day, "cost": reported_cash_flow_cost, "savings": savings, "soc_end": soc_end, 
                            "elec_energy": elec_energy, "gas_energy": gas_fuel_energy, "is_holiday": is_holiday, 
                            "gas_baseline_daily": gas_baseline_daily, 
                            "elec_cost_da": elec_cost_da, "gas_cost": gas_cost,
                            "afrr_cap_revenue": daily_afrr_cap_revenue, "afrr_energy_revenue": afrr_energy_revenue
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
            with col1: st.header("📊 Results Summary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    del st.session_state['results'], st.session_state['all_trades'], st.session_state['gas_baseline_total']
                    st.rerun()

            # --- KPI Calculations ---
            total_savings = sum([r['savings'] for r in results])
            total_afrr_cap_revenue = sum([r.get('afrr_cap_revenue', 0) for r in results]) if enable_afrr_capacity else 0
            total_afrr_energy_revenue = sum([r.get('afrr_energy_revenue', 0) for r in results]) if enable_afrr_energy else 0
            total_afrr_revenue = total_afrr_cap_revenue + total_afrr_energy_revenue
            
            total_gas_only_price = gas_baseline_total
            new_total_cost = total_gas_only_price - total_savings
            savings_pct = (total_savings / total_gas_only_price) * 100 if total_gas_only_price > 0 else 0

            avg_savings = np.mean([r['savings'] for r in results])
            avg_elec = np.mean([r['elec_energy'] for r in results])
            avg_gas = np.mean([r['gas_energy'] for r in results])
            thermal_from_elec = avg_elec * η
            thermal_from_gas = avg_gas * boiler_efficiency
            total_thermal_delivered = thermal_from_elec + thermal_from_gas
            elec_percentage = (thermal_from_elec / total_thermal_delivered) * 100 if total_thermal_delivered > 0 else 0

            # --- Display KPIs ---
            kpi_cols_1 = st.columns(4)
            kpi_cols_1[0].metric("Days Analyzed", len(results))
            kpi_cols_1[1].metric("Total Savings vs Gas Boiler", f"€{total_savings:,.0f}")
            kpi_cols_1[2].metric("Total Gas Only Price", f"€{total_gas_only_price:,.0f}")
            kpi_cols_1[3].metric("New Total Cost", f"€{new_total_cost:,.0f}")

            kpi_cols_2 = st.columns(3)
            kpi_cols_2[0].metric("Thermal from Electricity", f"{elec_percentage:.1f}%")
            kpi_cols_2[2].metric("Savings", f"{savings_pct:.1f}%")
            if enable_afrr_capacity or enable_afrr_energy:
                kpi_cols_2[1].metric("Total aFRR Revenue", f"€{total_afrr_revenue:,.0f}")
            else:
                kpi_cols_2[1].metric("Total aFRR Revenue", "N/A (Disabled)")

            #########

            best_day = max(results, key=lambda x: x['savings']); worst_day = min(results, key=lambda x: x['savings'])
            col1, col2 = st.columns(2)
            with col1: st.success(f"**Best day:** {best_day['day']} (€{best_day['savings']:.2f} saved)")
            with col2: st.warning(f"**Worst day:** {worst_day['day']} (€{worst_day['savings']:.2f} saved)")

            st.header("📈 Visualizations")
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
            monthly_df['DA_Savings'] = monthly_df['savings'] - monthly_df['afrr_cap_revenue'] - monthly_df['afrr_energy_revenue']


            # Build columns for monthly summary based on enabled components
            summary_columns = ['DA_Savings']
            y_columns = ['DA_Savings']
            rename_dict = {}

            if enable_afrr_capacity:
                summary_columns.append('afrr_cap_revenue')
                y_columns.append('aFRR Capacity Revenue')
                rename_dict['afrr_cap_revenue'] = 'aFRR Capacity Revenue'

            if enable_afrr_energy:
                summary_columns.append('afrr_energy_revenue')
                y_columns.append('aFRR Energy Revenue')
                rename_dict['afrr_energy_revenue'] = 'aFRR Energy Revenue'

            monthly_summary = monthly_df.groupby('month')[summary_columns].sum().reset_index()
            monthly_summary.rename(columns=rename_dict, inplace=True)

            fig_monthly = px.bar(monthly_summary, x='month', y=y_columns, title='Monthly Revenue & Savings Stack', height=500)
            fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Total Value (€)', legend_title='Revenue Stream')
            st.plotly_chart(fig_monthly, use_container_width=True)

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
                        
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_th_discharge'], name='Discharging', line=dict(color='red', dash='dot')), row=1, col=1, secondary_y=True)
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
                        f"- Thermal Contribution from Electricity: {elec_percentage:.1f}%\n"
                    )

                    if enable_afrr_capacity or enable_afrr_energy:
                        params_text += f"- Total aFRR Revenue: €{total_afrr_revenue:.2f}\n"
                        if enable_afrr_capacity:
                            params_text += f"- aFRR Capacity Revenue: €{total_afrr_cap_revenue:.2f}\n"
                        if enable_afrr_energy:
                            params_text += f"- aFRR Energy Revenue: €{total_afrr_energy_revenue:.2f}\n"
                    
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