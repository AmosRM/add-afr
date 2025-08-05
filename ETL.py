import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime
from typing import Union, Dict, Optional

def fetch_enappsys_data(
    api_type: str = "chart",  # "chart" or "bulk"
    credentials: Dict[str, str] = None,
    chart_code: str = None,
    bulk_type: str = None,
    entities: str = "ALL",
    start_date: str = None,
    end_date: str = None,
    resolution: str = "qh",
    timezone: str = "CET",
    currency: str = "EUR",
    **kwargs
) -> pd.DataFrame:
    """
    Fetch data from EnAppSys API (Chart API or Bulk API)
    
    Parameters:
    -----------
    api_type : str
        Either "chart" for Chart API or "bulk" for Bulk API
    credentials : dict
        Dictionary with 'user' and 'pass' keys
    chart_code : str
        Chart code for Chart API (e.g., "gb/elec/pricing/daprices")
    bulk_type : str
        Data type for Bulk API (e.g., "NL_SOLAR_FORECAST")
    entities : str
        Entities to fetch (default "ALL")
    start_date : str
        Start date in format "dd/mm/yyyy hh:mm" or "yyyymmddhhmm"
    end_date : str
        End date in format "dd/mm/yyyy hh:mm" or "yyyymmddhhmm"
    resolution : str
        Data resolution ("qh" for quarter hourly, "hourly", "daily", etc. - default is "qh")
    timezone : str
        Timezone (CET, WET, EET)
    currency : str
        Currency code
    **kwargs : dict
        Additional parameters for the API
    
    Returns:
    --------
    pd.DataFrame
        Raw data from API as DataFrame
    """
    
    if not credentials or 'user' not in credentials or 'pass' not in credentials:
        raise ValueError("Credentials must be provided with 'user' and 'pass' keys")
    
    # Convert date format if needed
    def convert_date_format(date_str):
        if not date_str:
            return None
        # If already in yyyymmddhhmm format, return as is
        if len(date_str) == 12 and date_str.isdigit():
            return date_str
        # Convert from dd/mm/yyyy hh:mm format
        try:
            dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
            return dt.strftime('%Y%m%d%H%M')
        except ValueError:
            try:
                dt = datetime.strptime(date_str, '%d/%m/%Y')
                return dt.strftime('%Y%m%d%H%M')
            except ValueError:
                raise ValueError(f"Invalid date format: {date_str}. Use 'dd/mm/yyyy hh:mm' or 'yyyymmddhhmm'")
    
    start_formatted = convert_date_format(start_date)
    end_formatted = convert_date_format(end_date)
    
    if api_type.lower() == "chart":
        if not chart_code:
            raise ValueError("chart_code is required for Chart API")
        
        url = "https://app.enappsys.com/datadownload"
        params = {
            "code": chart_code,
            "currency": currency,
            "start": start_formatted,
            "end": end_formatted,
            "res": resolution,
            "tag": "csv",
            "timezone": timezone,
            "user": credentials['user'],
            "pass": credentials['pass']
        }
        
    elif api_type.lower() == "bulk":
        if not bulk_type:
            raise ValueError("bulk_type is required for Bulk API")
        
        url = "https://app.enappsys.com/csvapi"
        params = {
            "type": bulk_type,
            "entities": entities,
            "start": start_formatted,
            "end": end_formatted,
            "res": resolution,
            "tag": "csv",
            "timezone": timezone,
            "user": credentials['user'],
            "pass": credentials['pass']
        }
    else:
        raise ValueError("api_type must be either 'chart' or 'bulk'")
    
    # Add any additional parameters
    params.update(kwargs)
    
    print(f"Fetching data from {api_type.upper()} API...")
    print(f"URL: {url}")
    print(f"Parameters: {dict((k, v if k != 'pass' else '*****') for k, v in params.items())}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        if response.status_code != 200:
            raise requests.exceptions.RequestException(f'API error. Status code: {response.status_code}. Response: {response.text}')
        
        # Check if response contains error messages
        response_text = response.text
        if "Username and password are not valid" in response_text:
            raise ValueError("Invalid credentials")
        elif "You have selected to download too much data" in response_text:
            raise ValueError("Too much data requested. Please reduce the date range.")
        elif "does not exist" in response_text or "invalid" in response_text.lower():
            raise ValueError(f"API returned error: {response_text[:200]}...")
        
        # Parse CSV response
        df = pd.read_csv(io.StringIO(response_text))
        
        print(f"Successfully fetched {len(df)} rows from API")
        return df
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch data from API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing API response: {str(e)}")


def etl_long_to_wide(
    input_source: Union[str, pd.DataFrame] = None, 
    output_file: Optional[str] = None, 
    datetime_column_name: str = 'Date (CET)',
    value_column_name: str = 'Day Ahead Price',
    input_date_format: str = '%d/%m/%Y %H:%M',
    # API parameters
    use_api: bool = False,
    api_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Enhanced ETL function that supports both file input and API data collection.
    
    Parameters:
    -----------
    input_source : str or pd.DataFrame
        File path to CSV or DataFrame (ignored if use_api=True)
    output_file : str, optional
        Output file path
    datetime_column_name : str
        Name of the datetime column
    value_column_name : str
        Name of the value column
    input_date_format : str
        Format of input datetime strings
    use_api : bool
        Whether to fetch data from API instead of file
    api_config : dict
        API configuration dictionary with the following structure:
        {
            'api_type': 'chart' or 'bulk',
            'credentials': {'user': 'username', 'pass': 'password'},
            'chart_code': 'chart_code' (for chart API),
            'bulk_type': 'data_type' (for bulk API),
            'entities': 'entities_list' (for bulk API),
            'start_date': 'dd/mm/yyyy hh:mm',
            'end_date': 'dd/mm/yyyy hh:mm',
            'resolution': 'qh',
            'timezone': 'CET',
            'currency': 'EUR'
        }
    
    Returns:
    --------
    pd.DataFrame
        Transformed wide-format DataFrame
    """
    
    def _transform_dataframe(df):
        """Helper function to perform the core transformation logic."""
        # Handle potential units row
        if len(df) > 0 and df.iloc[0].astype(str).str.contains('EUR|MW|%', na=False).any():
            print("Detected units row, removing it...")
            df = df.drop(0).reset_index(drop=True)

        # Validate columns
        if datetime_column_name not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"Datetime column '{datetime_column_name}' not found. Available columns: {available_cols}")
        if value_column_name not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"Value column '{value_column_name}' not found. Available columns: {available_cols}")
            
        # Process data
        print(f"Processing datetime from column: '{datetime_column_name}'")
        df[datetime_column_name] = df[datetime_column_name].astype(str).str.replace(r'[\[\]]', '', regex=True)
        df = df[df[datetime_column_name].str.strip() != ''].copy()
        df['datetime'] = pd.to_datetime(df[datetime_column_name], format=input_date_format, errors='coerce')
        
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        if len(df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid date formats.")

        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')

        df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
        if df[value_column_name].isnull().any():
            print(f"Warning: Coerced non-numeric values to NaN in '{value_column_name}'.")

        print("Transforming to wide format using pivot_table...")
        wide_df = df.pivot_table(
            index='date', 
            columns='time', 
            values=value_column_name,
            aggfunc='mean'
        )
        
        wide_df = wide_df.reset_index()
        wide_df['date'] = pd.to_datetime(wide_df['date']).dt.strftime('%Y-%m-%d')
        
        date_col_data = wide_df['date']
        time_cols_df = wide_df.drop('date', axis=1)
        
        print(f"Handling missing values...")
        time_cols_df = time_cols_df.interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
        
        wide_df = pd.concat([date_col_data, time_cols_df], axis=1)
        
        time_cols = [col for col in wide_df.columns if col != 'date']
        time_cols_sorted = sorted(time_cols, key=lambda x: datetime.strptime(x, '%H:%M:%S').time())
        wide_df = wide_df[['date'] + time_cols_sorted]
        
        return wide_df

    # Main Logic
    if use_api:
        if not api_config:
            raise ValueError("api_config is required when use_api=True")
        
        print("Fetching data from API...")
        try:
            df = fetch_enappsys_data(**api_config)
            print(f"API data shape: {df.shape}")
            print(f"API data columns: {list(df.columns)}")
            
            # Auto-detect column names for API data
            if datetime_column_name not in df.columns:
                # Common datetime column names in EnAppSys data
                possible_datetime_cols = ['Date (CET)', 'Date (WET)', 'Date (EET)', 'DateTime', 'Date']
                for col in possible_datetime_cols:
                    if col in df.columns:
                        datetime_column_name = col
                        print(f"Auto-detected datetime column: {datetime_column_name}")
                        break
                else:
                    # If still not found, use first column
                    datetime_column_name = df.columns[0]
                    print(f"Using first column as datetime: {datetime_column_name}")
            
            if value_column_name not in df.columns:
                # Try to find a numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_column_name = numeric_cols[0]
                    print(f"Auto-detected value column: {value_column_name}")
                else:
                    # Use second column if first is datetime
                    value_column_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    print(f"Using column as value: {value_column_name}")
            
        except Exception as e:
            raise Exception(f"Failed to fetch data from API: {str(e)}")
    else:
        # Original file-based logic
        try:
            # Attempt 1: Assume a clean CSV with no metadata row (skiprows=0)
            print("Attempting to read CSV with skiprows=0...")
            if hasattr(input_source, 'seek'):
                input_source.seek(0)
            df = pd.read_csv(input_source)
            if datetime_column_name not in df.columns:
                raise ValueError("Header not found on the first line.")
            print("Successfully read with skiprows=0. Transforming data...")

        except (ValueError, KeyError) as e:
            print(f"Reading with skiprows=0 failed ({e}). Retrying with skiprows=1...")
            try:
                if hasattr(input_source, 'seek'):
                    input_source.seek(0)
                df = pd.read_csv(input_source, skiprows=1)
                print("Successfully read with skiprows=1. Transforming data...")
            except Exception as e_inner:
                print(f"Second attempt (skiprows=1) also failed: {e_inner}")
                raise ValueError(f"Could not process the CSV file with either 0 or 1 skipped rows. Please check the file format. Error: {e_inner}")

    # Transform the data
    final_df = _transform_dataframe(df)
    print("Transformation complete!")
    
    if output_file:
        print(f"Saving to {output_file}...")
        final_df.to_csv(output_file, index=False)
    
    return final_df


def main():
    """Example usage of the enhanced ETL function"""
    
    # Example 1: Using file input (original functionality)
    print("=== Example 1: File Input ===")
    try:
        input_file = 'idprices-epexshort.csv'
        output_file = 'output_wide_format_file.csv'
        result_df = etl_long_to_wide(input_source=input_file, output_file=output_file)
        if result_df is not None:
            print("File-based ETL Process Successful")
            print(result_df.head())
    except Exception as e:
        print(f"File-based ETL process failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using Chart API
    print("=== Example 2: Chart API ===")
    try:
        api_config = {
            'api_type': 'chart',
            'credentials': {
                'user': 'your_username',  # Replace with actual credentials
                'pass': 'your_password'   # Replace with actual credentials
            },
            'chart_code': 'gb/elec/pricing/daprices',  # UK day-ahead prices
            'start_date': '01/01/2023 00:00',
            'end_date': '31/01/2023 23:59',
            'resolution': 'qh',
            'timezone': 'WET',
            'currency': 'GBP'
        }
        
        result_df = etl_long_to_wide(
            use_api=True,
            api_config=api_config,
            output_file='output_wide_format_api_chart.csv'
        )
        print("Chart API ETL Process Successful")
        print(result_df.head())
        
    except Exception as e:
        print(f"Chart API ETL process failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Using Bulk API
    print("=== Example 3: Bulk API ===")
    try:
        api_config = {
            'api_type': 'bulk',
            'credentials': {
                'user': 'your_username',  # Replace with actual credentials
                'pass': 'your_password'   # Replace with actual credentials
            },
            'bulk_type': 'NL_SOLAR_FORECAST',
            'entities': 'NL_SOLAR,NL_SOLAR_P10,NL_SOLAR_P90',
            'start_date': '01/01/2023 00:00',
            'end_date': '31/01/2023 23:59',
            'resolution': 'qh',
            'timezone': 'CET'
        }
        
        result_df = etl_long_to_wide(
            use_api=True,
            api_config=api_config,
            output_file='output_wide_format_api_bulk.csv',
            datetime_column_name='Date (CET)',  # Adjust based on API response
            value_column_name='NL_SOLAR'       # Adjust based on API response
        )
        print("Bulk API ETL Process Successful")
        print(result_df.head())
        
    except Exception as e:
        print(f"Bulk API ETL process failed: {e}")


if __name__ == "__main__":
    main()