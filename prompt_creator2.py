from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from pathlib import Path
import argparse
import os



def parse_date(date_str: str) -> datetime:
    """Convert date string in format 'YYYY_MM' to datetime object."""
    return datetime.strptime(date_str, '%Y_%m')

def calculate_months_between(start_date: str, end_date: str) -> int:
    """Calculate the number of months between two dates in 'YYYY_MM' format."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    return (end.year - start.year) * 12 + (end.month - start.month) + 1

def validate_date_range(start_date: str, end_date: str, max_months: int) -> bool:
    """Validate that the date range is within the maximum allowed months."""
    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        if end < start:
            print(f"Error: End date {end_date} is before start date {start_date}")
            return False
            
        months_requested = calculate_months_between(start_date, end_date)
        if months_requested > max_months:
            print(f"Error: Requested {months_requested} months of data exceeds maximum allowed {max_months} months")
            return False
            
        return True
        
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return False

def process_txtfile(txtfile):
    with open(txtfile, 'r') as file:
        content = file.read()
    return content

def calculate_max_months(model_context_window: int, indicator: str, prompt: str) -> int:
    """
    Calculate the maximum number of months of data that can fit in the model's context window.
    
    Args:
        model_context_window: The context window size of the model in tokens
        indicator: The technical indicator name to analyze
    
    Returns:
        int: Maximum number of months that can fit in the context window
    """
    # Get the base prompt token count (approximate 4 chars per token)
    base_prompt_tokens = len(prompt) // 4
    
    # Get sample data from one month to calculate average size
    sample_path = Path("NVDA_HISTORICAL/2025_01")
    
    # Read sample stock data
    with open(sample_path / "NVDA_DATA.json", 'r') as f:
        stock_data = json.load(f)
    
    # Read sample indicator data
    with open(sample_path / f"{indicator}.json", 'r') as f:
        indicator_data = json.load(f)
    
    # Calculate tokens for one month of data (approximate 4 chars per token)
    stock_data_tokens = len(json.dumps(stock_data)) // 4
    indicator_data_tokens = len(json.dumps(indicator_data)) // 4
    tokens_per_month = stock_data_tokens + indicator_data_tokens
    
    # Calculate available tokens after base prompt
    available_tokens = model_context_window - base_prompt_tokens
    
    # Calculate max months (leave 20% buffer for safety)
    max_months = int((available_tokens * 0.8) // tokens_per_month)
    
    return max(1, min(max_months, 12))  # Ensure between 1 and 12 months

def get_context_window(model_name: str) -> int:
    """
    Get the context window size for the specified model.
    
    Args:
        model_name: Name of the LLM model
    
    Returns:
        int: Context window size in tokens
    """
    # Add more models as needed
    context_windows = {
        "claude-sonnet-3.5": 128000,
        "gemini-2.0-flash-thinking-exp-01-21": 1000000,
        "gemini-exp-1206": 1000000
    }
    
    return context_windows.get(model_name, 8192)  # Default to 8192 if model not found

def collect_data_for_period(ticker: str, indicator: str, start_date: str, end_date: str) -> Tuple[Dict, Dict]:
    """
    Collect stock and indicator data for the specified period.
    
    Args:
        ticker: Stock ticker symbol
        indicator: Technical indicator name
        start_date: Start date in YYYY_MM format
        end_date: End date in YYYY_MM format
    
    Returns:
        Tuple of (combined_stock_data, combined_indicator_data)
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    combined_stock_data = {}
    combined_indicator_data = {}
    
    current = start
    while current <= end:
        month_str = current.strftime('%Y_%m')
        data_path = Path(f"{ticker}_HISTORICAL/{month_str}")
        
        # Read stock data
        try:
            with open(data_path / f"{ticker}_DATA.json", 'r') as f:
                month_stock_data = json.load(f)
                combined_stock_data.update(month_stock_data)
        except FileNotFoundError:
            print(f"Warning: No stock data found for {month_str}")
            
        # Read indicator data
        try:
            with open(data_path / f"{indicator}.json", 'r') as f:
                month_indicator_data = json.load(f)
                combined_indicator_data.update(month_indicator_data)
        except FileNotFoundError:
            print(f"Warning: No indicator data found for {month_str}")
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return combined_stock_data, combined_indicator_data

def create_filled_prompt(ticker: str, indicator: str, start_date: str, end_date: str, prompt_path: str) -> str:
    """
    Create a prompt with the data filled in.
    
    Args:
        ticker: Stock ticker symbol
        indicator: Technical indicator name
        start_date: Start date in YYYY_MM format
        end_date: End date in YYYY_MM format 
    
    Returns:
        str: Filled prompt template
    """
    # Collect data for the period
    stock_data, indicator_data = collect_data_for_period(ticker, indicator, start_date, end_date)
    prompt = process_txtfile(prompt_path)
    # Create the filled prompt
    filled_prompt = prompt.format(
        STOCK_DATA=json.dumps(stock_data, indent=2),
        INDICATOR_DATA=json.dumps(indicator_data, indent=2),
        TECH_INDICATOR=indicator,
        TICKER=ticker,
        PERIOD_START=start_date,
        PERIOD_END=end_date,
        INTERVAL="daily"  # You might want to make this configurable
    )
    
    return filled_prompt

def main():
    tech_indicators = [
    "ht_trendmode", "ht_sine", "ht_phasor", "ht_dcphase", "ht_dcperiod",
    "adosc", "obv", "midprice", "midpoint", "bbands", "trange", "natr",
    "atr", "willr", "plus_dm", "minus_dm", "plus_di", "minus_di", "dx",
    "ultosc", "trix", "mfi", "aroonosc", "aroon", "rocr", "roc", "cmo",
    "cci", "bop", "mom", "ppo", "apo", "adxr", "adx", "stochrsi", "stochf",
    "stoch", "rsi", "macd", "vwap", "t3", "mama", "kama", "trima", "tema",
    "dema", "wma", "ema", "sma"
    ]
    
    #claude sonnet 3.5 for nonreasoning.txt
    #reasoning.txt for o1 mini
    #gemini-exp-1206for google_nonreasoning.txt
    #gemini-2.0-flash-thinking-exp-01-21 for google_reasoning.txt
    models = ["claude-sonnet-3.5", "gemini-exp-1206", "gemini-2.0-flash-thinking-exp-01-21"]

    parser = argparse.ArgumentParser(description="Input model, ticker, technical indicator, and path to prompt")
    parser.add_argument("--model", type=str, required=True, help="The model to use. Options are: " + ", ".join(models))
    parser.add_argument("--ticker", type=str, required=True, help="The stock ticker to analyze")
    parser.add_argument("--indicator", type=str, required=True, help="The technical indicator to evaluate. Options are: " + ", ".join(tech_indicators))
    args = parser.parse_args()

    if args.model == "claude-sonnet-3.5":
        prompt_path = "prompts/nonreasoning.txt"
    elif args.model == "o1-mini":
        prompt_path = "prompts/reasoning.txt"
    elif args.model == "gemini-exp-1206":
        prompt_path = "prompts/google_nonreasoning.txt"
    elif args.model == "gemini-2.0-flash-thinking-exp-01-21":
        prompt_path = "prompts/google_reasoning.txt"
    else:
        print("Invalid model")
        exit(1)


    # Read the prompt template first
    prompt = process_txtfile(prompt_path)

    # Calculate max months based on model's context window
    context_window = get_context_window(args.model)
    max_months = calculate_max_months(context_window, args.indicator, prompt)
    
    print(f"\nMaximum months of data that can be analyzed: {max_months}")
    
    # Get date range from user after showing max months
    while True:
        try:
            start_date = input("\nEnter start date (YYYY_MM format): ")
            end_date = input("Enter end date (YYYY_MM format): ")
            
            if validate_date_range(start_date, end_date, max_months):
                print("\nDate range is valid!")
                # Store the validated dates for further processing
                args.start_date = start_date
                args.end_date = end_date
                break
            else:
                print("\nPlease try again with a valid date range.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit(1)
            
    # After successful date validation, create the filled prompt
    filled_prompt = create_filled_prompt(
        ticker=args.ticker,
        indicator=args.indicator,
        start_date=args.start_date,
        end_date=args.end_date,
        prompt_path=args.prompt_path
    )
    
    # Print the length of the prompt (helpful for debugging)
    print(f"\nPrompt length (chars): {len(filled_prompt)}")
    print(f"Approximate tokens: {len(filled_prompt) // 4}")
    
    # Extract just the prompt template name from the path
    prompt_template_name = Path(args.prompt_path).stem  # This gets filename without extension
    
    output_dir = Path("prompts_filled")
    output_dir.mkdir(exist_ok=True)
    
    # Create a cleaner output filename
    output_file = output_dir / f"{args.ticker}_{args.indicator}_{args.start_date}_{args.end_date}_{prompt_template_name}.txt"
    with open(output_file, 'w') as f:
        f.write(filled_prompt)
    
    print(f"\nGenerated prompt saved to: {output_file}")

if __name__ == "__main__":
    main()