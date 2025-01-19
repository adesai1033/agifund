import requests
import json
from datetime import datetime, timedelta
import time
import os

def get_intraday_price_history(symbol, api_key, interval='5min', days=7):
    """Helper function to get intraday price history"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}&entitlement=delayed'
    r = requests.get(url)
    data = r.json()
    
    if f"Time Series ({interval})" not in data:
        raise Exception(f"Error fetching intraday price data: {data}")
        
    prices = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    time_series = data[f"Time Series ({interval})"]
    for timestamp, values in time_series.items():
        try:
            if len(timestamp) == 16:
                timestamp = timestamp + ":00"
                
            candle_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            if start_date <= candle_time <= end_date:
                prices.append({
                    "timestamp": timestamp,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"])
                })
        except ValueError:
            continue
            
    return sorted(prices, key=lambda x: x["timestamp"])

def get_technical_indicator(symbol, api_key, indicator, output_file, interval='5min', days=7, **params):
    """Generic function to fetch technical indicators from Alpha Vantage"""
    base_url = 'https://www.alphavantage.co/query'
    
    # Calculate the time window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query_params = {
        'function': f'{indicator}',
        'symbol': symbol,
        'interval': interval,
        'apikey': api_key,
        'outputsize': 'full',
        **params
    }
    
    try:
        response = requests.get(base_url, params=query_params)
        data = response.json()
        
        # Different indicators have different response formats
        indicator_data_key = ""
        if indicator == "RSI":
            indicator_data_key = "Technical Analysis: RSI"
        elif indicator == "MACD":
            indicator_data_key = "Technical Analysis: MACD"
        elif indicator == "BBANDS":
            indicator_data_key = "Technical Analysis: BBANDS"
        elif indicator == "STOCH":
            indicator_data_key = "Technical Analysis: STOCH"
        else:
            indicator_data_key = f"Technical Analysis: {indicator}"
            
        if indicator_data_key not in data:
            raise Exception(f"Error fetching {indicator} data: {data.get('Note', 'Unknown error')}")
            
        formatted_data = {
            "symbol": symbol,
            "indicator": indicator,
            "parameters": params,
            "interval": interval,
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "data": []
        }
        
        # Get all timestamps from the indicator data within our time window
        for timestamp, values in data[indicator_data_key].items():
            try:
                if len(timestamp) == 16:
                    timestamp = timestamp + ":00"
                    
                data_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                
                # Only include data points within our time window
                if start_date <= data_time <= end_date:
                    entry = {"timestamp": timestamp}
                    entry.update({k: float(v) for k, v in values.items()})
                    formatted_data["data"].append(entry)
            except ValueError:
                continue
            
        formatted_data["data"].sort(key=lambda x: x["timestamp"])
        
        if not formatted_data["data"]:
            raise Exception(f"No valid data points found for {indicator} within the specified time range")
            
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"{indicator} data saved to {output_file} with {len(formatted_data['data'])} data points")
        
    except Exception as e:
        print(f"Error calculating {indicator}: {str(e)}")

def get_economic_events(api_key, output_dir):
    """Fetch economic events and indicators"""
    events = {
        'GDP': 'REAL_GDP',
        'FEDERAL_FUNDS_RATE': 'FEDERAL_FUNDS_RATE',
        'TREASURY_YIELD': 'TREASURY_YIELD',
        'CPI': 'CPI',
        'INFLATION': 'INFLATION',
        'UNEMPLOYMENT': 'UNEMPLOYMENT',
        'NONFARM_PAYROLL': 'NONFARM_PAYROLL',
        'RETAIL_SALES': 'RETAIL_SALES',
        'DURABLES': 'DURABLES'
    }
    
    for event_name, function in events.items():
        time.sleep(1)  # Rate limiting
        url = f'https://www.alphavantage.co/query?function={function}&apikey={api_key}'
        
        try:
            response = requests.get(url)
            data = response.json()
            
            output_file = os.path.join(output_dir, f"{event_name.lower()}.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Saved {event_name} data to {output_file}")
            
        except Exception as e:
            print(f"Error fetching {event_name}: {str(e)}")

def get_earnings_calendar(api_key, output_dir):
    """Fetch earnings calendar for major companies"""
    url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        output_file = os.path.join(output_dir, "earnings_calendar.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved earnings calendar to {output_file}")
        
    except Exception as e:
        print(f"Error fetching earnings calendar: {str(e)}")

def get_ipo_calendar(api_key, output_dir):
    """Fetch IPO calendar"""
    url = f'https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        output_file = os.path.join(output_dir, "ipo_calendar.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved IPO calendar to {output_file}")
        
    except Exception as e:
        print(f"Error fetching IPO calendar: {str(e)}")

def get_intraday_data(symbol, api_key, output_file, interval='5min', days=7):
    """Fetches intraday data and calculates various technical indicators at the same sampling frequency"""
    # Create data directories
    data_dir = f"{symbol}_DATA"
    events_dir = "Economic_EVENTS"
    for directory in [data_dir, events_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Fetch economic events and calendars
    print("\nFetching economic events and market calendars...")
    get_economic_events(api_key, events_dir)
    get_earnings_calendar(api_key, events_dir)
    get_ipo_calendar(api_key, events_dir)
        
    try:
        # Get price history
        prices = get_intraday_price_history(symbol, api_key, interval, days)
        
        formatted_data = {
            "symbol": symbol,
            "period_start": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "period_end": datetime.now().strftime("%Y-%m-%d"),
            "interval": interval,
            "candles": prices
        }
        
        output_path = os.path.join(data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"Price data successfully saved to {output_path}")
        
        # Calculate various technical indicators
        print("\nCalculating technical indicators...")
        
        # Define indicator configurations
        indicators = {
            # Moving Averages
            'MA_TYPES': [
                ('EMA', {'time_period': 20, 'series_type': 'close'}),
                ('WMA', {'time_period': 20, 'series_type': 'close'}),
                ('DEMA', {'time_period': 20, 'series_type': 'close'}),
                ('TEMA', {'time_period': 20, 'series_type': 'close'}),
                ('TRIMA', {'time_period': 20, 'series_type': 'close'}),
                ('KAMA', {'time_period': 20, 'series_type': 'close'}),
                ('MAMA', {'series_type': 'close'}),
                ('T3', {'time_period': 20, 'series_type': 'close'})
            ],
            # Momentum Indicators
            'MOMENTUM': [
                ('MACD', {'series_type': 'close', 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
                ('RSI', {'time_period': 14, 'series_type': 'close'}),
                ('STOCH', {}),
                ('ADX', {'time_period': 14}),
                ('CCI', {'time_period': 20}),
                ('AROON', {'time_period': 14}),
                ('MFI', {'time_period': 14})
            ],
            # Volatility Indicators
            'VOLATILITY': [
                ('BBANDS', {'time_period': 20, 'series_type': 'close', 'nbdevup': 2, 'nbdevdn': 2}),
                ('ATR', {'time_period': 14})
            ],
            # Volume Indicators
            'VOLUME': [
                ('OBV', {})
            ]
        }

        # Calculate all indicators
        for category in ['MA_TYPES', 'MOMENTUM', 'VOLATILITY', 'VOLUME']:
            for indicator_config in indicators[category]:
                time.sleep(1)
                indicator_name, params = indicator_config
                get_technical_indicator(
                    symbol,
                    api_key,
                    indicator_name,
                    os.path.join(data_dir, f"{indicator_name.lower()}.json"),
                    interval=interval,
                    days=days,
                    **params
                )
                
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage:
api_key = "B8MJERYXGJNMHAON"
get_intraday_data("NVDA", api_key, "NVDA_DATA.json", interval='5min', days=7)
