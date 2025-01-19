import requests
import json
from datetime import datetime, timedelta
import time
import os

def get_daily_price_history(symbol, api_key, lookback_days=None):
    """Helper function to get daily price history"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}&entitlement=delayed'
    r = requests.get(url)
    data = r.json()
    
    if "Time Series (Daily)" not in data:
        raise Exception(f"Error fetching daily price data: {data}")
        
    prices = []
    end_date = datetime.now()
    
    daily_data = data["Time Series (Daily)"]
    for date_str in daily_data:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if lookback_days:
            start_date = end_date - timedelta(days=lookback_days)
            if not (start_date <= date <= end_date):
                continue
                
        prices.append({
            "date": date_str,
            "close": float(daily_data[date_str]["4. close"])
        })
            
    return sorted(prices, key=lambda x: x["date"])

def calculate_daily_sma(symbol, api_key, output_file, period):
    """Calculates SMA using daily sampling frequency."""
    try:
        prices = get_daily_price_history(symbol, api_key)
        
        if len(prices) < period:
            raise Exception(f"Insufficient price history. Need {period} days, got {len(prices)}")
        
        formatted_data = {
            "symbol": symbol,
            "period": period,
            "interval": "daily",
            "indicators": []
        }
        
        for i in range(period-1, len(prices)):
            window = prices[i-period+1:i+1]
            sma = sum(p["close"] for p in window) / period
            
            indicator_data = {
                "timestamp": prices[i]["date"],
                "close_price": prices[i]["close"],
                f"sma_{period}": round(sma, 2)
            }
            formatted_data["indicators"].append(indicator_data)
            
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"Daily SMA-{period} data saved to {output_file}")
        
    except Exception as e:
        print(f"Error calculating SMA-{period}: {str(e)}")

def get_technical_indicator(symbol, api_key, indicator, output_file, **params):
    """Generic function to fetch technical indicators from Alpha Vantage"""
    base_url = 'https://www.alphavantage.co/query'
    
    query_params = {
        'function': f'{indicator}',
        'symbol': symbol,
        'apikey': api_key,
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
        else:
            indicator_data_key = f"Technical Analysis: {indicator}"
            
        if indicator_data_key not in data:
            raise Exception(f"Error fetching {indicator} data: {data.get('Note', 'Unknown error')}")
            
        formatted_data = {
            "symbol": symbol,
            "indicator": indicator,
            "parameters": params,
            "data": []
        }
        
        for date, values in data[indicator_data_key].items():
            entry = {"timestamp": date}
            entry.update({k: float(v) for k, v in values.items()})
            formatted_data["data"].append(entry)
            
        formatted_data["data"].sort(key=lambda x: x["timestamp"])
        
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"{indicator} data saved to {output_file}")
        
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

def get_intraday_data(symbol, api_key, output_file, days=7):
    """Fetches 5-minute intraday data and calculates various technical indicators"""
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
        
    base_url = 'https://www.alphavantage.co/query'
    
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '5min',
        'apikey': api_key,
        'outputsize': 'full',
        'entitlement': 'delayed'
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "Time Series (5min)" not in data:
            raise Exception(f"Error fetching data: {data.get('Note', 'Unknown error')}")
            
        time_series = data["Time Series (5min)"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        formatted_data = {
            "symbol": symbol,
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "interval": "5min",
            "candles": []
        }
        
        for timestamp, values in time_series.items():
            try:
                if len(timestamp) == 16:
                    timestamp = timestamp + ":00"
                    
                candle_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                
                if start_date <= candle_time <= end_date:
                    candle_data = {
                        "timestamp": timestamp,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    }
                    formatted_data["candles"].append(candle_data)
            except ValueError as e:
                print(f"Skipping invalid timestamp: {timestamp}")
                continue
        
        formatted_data["candles"].sort(key=lambda x: x["timestamp"])
        
        output_path = os.path.join(data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"Price data successfully saved to {output_path}")
        
        # Calculate various technical indicators
        print("\nCalculating technical indicators...")
        
        # Define indicator configurations
        indicators = {
            # Simple Moving Averages
            'SMA': [
                {'period': 200, 'params': {'interval': 'daily', 'time_period': 200, 'series_type': 'close'}},
                {'period': 50, 'params': {'interval': 'daily', 'time_period': 50, 'series_type': 'close'}},
                {'period': 20, 'params': {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}}
            ],
            # Moving Averages
            'MA_TYPES': [
                ('EMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('WMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('DEMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('TEMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('TRIMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('KAMA', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('MAMA', {'interval': 'daily', 'series_type': 'close'}),
                ('T3', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'})
            ],
            # Momentum Indicators
            'MOMENTUM': [
                ('MACD', {'interval': 'daily', 'series_type': 'close', 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
                ('MACDEXT', {'interval': 'daily', 'series_type': 'close'}),
                ('STOCH', {'interval': 'daily'}),
                ('STOCHF', {'interval': 'daily'}),
                ('RSI', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
                ('STOCHRSI', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
                ('WILLR', {'interval': 'daily', 'time_period': 14}),
                ('ADX', {'interval': 'daily', 'time_period': 14}),
                ('ADXR', {'interval': 'daily', 'time_period': 14}),
                ('APO', {'interval': 'daily', 'series_type': 'close'}),
                ('PPO', {'interval': 'daily', 'series_type': 'close'}),
                ('MOM', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
                ('BOP', {'interval': 'daily'}),
                ('CCI', {'interval': 'daily', 'time_period': 20}),
                ('CMO', {'interval': 'daily', 'time_period': 14, 'series_type': 'close'}),
                ('ROC', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
                ('ROCR', {'interval': 'daily', 'time_period': 10, 'series_type': 'close'}),
                ('AROON', {'interval': 'daily', 'time_period': 14}),
                ('AROONOSC', {'interval': 'daily', 'time_period': 14}),
                ('MFI', {'interval': 'daily', 'time_period': 14}),
                ('TRIX', {'interval': 'daily', 'time_period': 30, 'series_type': 'close'}),
                ('ULTOSC', {'interval': 'daily'}),
                ('DX', {'interval': 'daily', 'time_period': 14}),
                ('MINUS_DI', {'interval': 'daily', 'time_period': 14}),
                ('PLUS_DI', {'interval': 'daily', 'time_period': 14}),
                ('MINUS_DM', {'interval': 'daily', 'time_period': 14}),
                ('PLUS_DM', {'interval': 'daily', 'time_period': 14})
            ],
            # Volatility Indicators
            'VOLATILITY': [
                ('BBANDS', {'interval': 'daily', 'time_period': 20, 'series_type': 'close', 'nbdevup': 2, 'nbdevdn': 2}),
                ('MIDPOINT', {'interval': 'daily', 'time_period': 20, 'series_type': 'close'}),
                ('MIDPRICE', {'interval': 'daily', 'time_period': 20}),
                ('SAR', {'interval': 'daily'}),
                ('TRANGE', {'interval': 'daily'}),
                ('ATR', {'interval': 'daily', 'time_period': 14}),
                ('NATR', {'interval': 'daily', 'time_period': 14})
            ],
            # Volume Indicators
            'VOLUME': [
                ('OBV', {'interval': 'daily'})
            ],
            # Cycle Indicators
            'CYCLE': [
                ('HT_TRENDLINE', {'interval': 'daily', 'series_type': 'close'}),
                ('HT_SINE', {'interval': 'daily', 'series_type': 'close'}),
                ('HT_TRENDMODE', {'interval': 'daily', 'series_type': 'close'}),
                ('HT_DCPERIOD', {'interval': 'daily', 'series_type': 'close'}),
                ('HT_DCPHASE', {'interval': 'daily', 'series_type': 'close'}),
                ('HT_PHASOR', {'interval': 'daily', 'series_type': 'close'})
            ]
        }

        # Calculate SMAs
        for sma in indicators['SMA']:
            time.sleep(1)
            calculate_daily_sma(symbol, api_key, os.path.join(data_dir, f"SMA{sma['period']}_daily.json"), sma['period'])

        # Calculate all other indicators
        for category in ['MA_TYPES', 'MOMENTUM', 'VOLATILITY', 'VOLUME', 'CYCLE']:
            for indicator_config in indicators[category]:
                time.sleep(1)
                indicator_name, params = indicator_config
                get_technical_indicator(
                    symbol, 
                    api_key, 
                    indicator_name, 
                    os.path.join(data_dir, f"{indicator_name}_daily.json"), 
                    **params
                )
                
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage:
api_key = "B8MJERYXGJNMHAON"
get_intraday_data("TSLA", api_key, "TSLA_DATA.json", days=7)
