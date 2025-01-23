from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse
import os
from datetime import datetime, timedelta

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = ""
USER_PROMPT_TEMPLATE = ""
MERGED_PROMPT_REASONING_TEMPLATE = """
<Goal>GOAL<Goal>
Your goal is to conduct a novel and thorough analysis of the relationship between a specified technical indicator and the price movements of a given stock, strictly using only the data provided. You are tasked with uncovering insightful and potentially unconventional patterns or predictive capabilities that might not be immediately obvious through traditional financial analysis methods. Your analysis should be data-driven, rigorous, and aimed at exploring both conventional and groundbreaking interpretations of the indicator's behavior in relation to stock prices. The ultimate objective is to assess the indicator's potential for predicting stock price shifts based solely on the provided data, pushing beyond standard analytical approaches to discover hidden insights.
<Return Format>RETURN FORMAT<Return Format>
Your response must be structured in three distinct sections, each enclosed within specific tags:
<detailed_examination> - This section will contain your in-depth and methodical analysis. Within this section, you are required to follow a series of predefined steps (a through i, as detailed in the Context Dump). This is where you will explore the data, identify patterns, analyze relationships, and present your arguments and counterarguments, all supported by data references.
<summary> - Following the detailed examination, provide a concise summary of your most significant findings and insights. This section should highlight the key takeaways from your analysis regarding the technical indicator's behavior and potential predictive validity.
<disclaimer> - Conclude your response with a clear disclaimer stating that your analysis is for informational purposes only and should not be considered financial advice or the sole basis for making investment decisions.
<Warnings>WARNINGS<Warnings>
Data Limitation: You are strictly limited to using only the data provided within <stock_data> and <indicator_data>. Do not incorporate any external knowledge, pre-existing financial theories, or data from outside sources. All conclusions and observations must be derived solely from the datasets given to you.
No Financial Advice: Your analysis is for informational and analytical purposes only. It must not be construed as financial advice, investment recommendations, or guidance for trading decisions. You must explicitly state this in the disclaimer section.
Objective and Data-Driven: Maintain a purely objective and data-driven approach. Avoid speculation that is not directly supported by the provided data. Focus on identifying patterns, correlations, and potential predictive signals strictly within the confines of the given datasets.
Comprehensive Analysis: While seeking novel insights, ensure your analysis is comprehensive and addresses both conventional and unconventional interpretations. Explore potential counterarguments and validate your findings rigorously against the data.
<contextdump>CONTEXT DUMP<contextdump>
Data Provided:
<stock_data>{STOCK_DATA}</stock_data>: This contains candle data for the specified stock, including Open, High, Low, Close, and Volume for each time interval.
<indicator_data>{INDICATOR_DATA}</indicator_data>: This includes the calculated values of the specified technical indicator for each corresponding time interval in the stock data.
<tech_indicator>{TECH_INDICATOR}</tech_indicator>: Specifies the name of the technical indicator being analyzed.
<ticker>{TICKER}</ticker>: Indicates the stock ticker symbol.
<period_start>{PERIOD_START}</period_start>: Defines the starting date for the analysis period.
<period_end>{PERIOD_END}</period_end>: Defines the ending date for the analysis period.
<interval>{INTERVAL}</interval>: Specifies the time interval of the data (e.g., daily, hourly).
Analysis Steps within <detailed_examination>:
Examine Relationship: Analyze the relationship between the <tech_indicator> values and the stock price movements as represented in <stock_data>. Consider various aspects of price movement (e.g., price direction, volatility, magnitude of changes).
Identify Patterns: Search for any recurring patterns or signals where the technical indicator might precede, coincide with, or follow specific stock price movements. Explore both traditional patterns and any subtle, unconventional patterns. Look for potential predictive components.
Short-Term Price Shifts: Investigate how the <tech_indicator> might be used to identify potential short-term price shifts. Analyze if specific indicator values or changes correlate with subsequent short-term price increases or decreases.
Typical Thresholds: If applicable and evident from the data, identify and describe any typical threshold levels for the <tech_indicator> (e.g., levels that might be traditionally considered "overbought" or "oversold"). Assess their relevance based on the provided data.
Indicator Combinations: Suggest potential ways a trader might combine this <tech_indicator> with other hypothetical signals or conditions (derived solely from the data itself, not external indicators) to create a more robust trading strategy. Explore how combining observable patterns could improve signal reliability.
Predictive Validity: Summarize the predictive validity of the <tech_indicator> based on your analysis of the data. Provide specific data points and observations to either support or refute its effectiveness as a predictor of price movements for the given stock.
Detailed Examination Sub-points (within <detailed_examination>):
a. Key Data Points and Patterns: List specific data points that stand out and describe any initial patterns you observe between the <tech_indicator> and price movements. Be precise and reference specific data instances.
b. Correlations Analysis: Analyze and describe any correlations you identify between the <tech_indicator> values and stock price movements. Quantify correlations qualitatively (e.g., "strong positive correlation when indicator is above X").
c. Unconventional Relationships: Explore and identify any unconventional or less obvious relationships that might exist. Think beyond standard interpretations of the indicator and consider non-linear relationships or unusual signal patterns.
d. Counterarguments: For each observed pattern or potential predictive signal, consider and present potential counterarguments or data points that contradict your initial observations. This ensures a balanced and critical analysis.
e. Indicator Combinations (Data-Driven): Explore hypothetical combinations of observations or patterns within the data itself (not external indicators). How could combining different data-driven signals improve the analysis?
f. Data-Only Verification: Explicitly verify that all your conclusions are drawn solely from the provided <stock_data> and <indicator_data>. State that no external knowledge is used for each conclusion.
g. Data Visualization (Mental): Mentally visualize the trends of both the <tech_indicator> and stock prices. Describe what you "see" in these visualizations – e.g., "When the indicator rises sharply, price tends to follow with a slight delay," or "High indicator values coincide with increased price volatility."
h. Lagging/Leading Effects: Analyze and describe any potential lagging or leading effects of the <tech_indicator> in relation to price movements. Does the indicator typically precede price changes, or does it react to price changes? Quantify the observed lag or lead if possible.
i. Supporting/Contradicting Data Points: For each major observation or conclusion, explicitly list specific data points from the provided datasets that either strongly support or contradict your point. This grounds your analysis in concrete evidence.
"""
MERGED_PROMPT_NON_REASONING_TEMPLATE = ""

def parse_date(date_str: str) -> datetime:
    """Convert date string in format 'YYYY_MM' to datetime object."""
    return datetime.strptime(date_str, '%Y_%m')

def calculate_months_between(start_date: str, end_date: str) -> int:
    """Calculate the number of months between two dates in 'YYYY_MM' format."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    return (end.year - start.year) * 12 + (end.month - start.month) + 1

def validate_date_range(start_date: str, end_date: str, max_months: int) -> bool:
    """
    Validate that the date range is within the maximum allowed months and end date is after start date.
    """
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

def calculate_max_months(model_context_window: int, indicator: str) -> int:
    """
    Calculate the maximum number of months of data that can fit in the model's context window.
    
    Args:
        model_context_window: The context window size of the model in tokens
        indicator: The technical indicator name to analyze
    
    Returns:
        int: Maximum number of months that can fit in the context window
    """
    # Get the base prompt token count (approximate 4 chars per token)
    base_prompt_tokens = len(MERGED_PROMPT_REASONING_TEMPLATE) // 4
    
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
        "gpt-4o": 128000,
        "gemini-2.0-flash-thinking-exp-01-21": 1000000,
        "deepseek-reasoner": 128000
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

def create_filled_prompt(ticker: str, indicator: str, start_date: str, end_date: str) -> str:
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
    
    # Create the filled prompt
    filled_prompt = MERGED_PROMPT_REASONING_TEMPLATE.format(
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
    models = ["gpt-4o", "deepseek-reasoner", "gemini-2.0-flash-thinking-exp-01-21"]

    parser = argparse.ArgumentParser(description="Input ticker, indicator, and date range")
    parser.add_argument("--model", type=str, required=True, help="The model to use. Options are: " + ", ".join(models))
    parser.add_argument("--ticker", type=str, required=True, help="The stock ticker to analyze")
    parser.add_argument("--indicator", type=str, required=True, help="The technical indicator to evaluate. Options are: " + ", ".join(tech_indicators))
    args = parser.parse_args()

    # Calculate max months based on model's context window
    context_window = get_context_window(args.model)
    max_months = calculate_max_months(context_window, args.indicator)
    
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
        end_date=args.end_date
    )
    
    # Print the length of the prompt (helpful for debugging)
    print(f"\nPrompt length (chars): {len(filled_prompt)}")
    print(f"Approximate tokens: {len(filled_prompt) // 4}")
    
    # TODO: Send the filled prompt to your chosen LLM
    # You might want to save the prompt to a file for inspection
    output_dir = Path("generated_prompts")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{args.ticker}_{args.indicator}_{args.start_date}_{args.end_date}.txt"
    with open(output_file, 'w') as f:
        f.write(filled_prompt)
    
    print(f"\nGenerated prompt saved to: {output_file}")

if __name__ == "__main__":
    main()