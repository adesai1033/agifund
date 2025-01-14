# Stock Analysis with Market Event Correlation

## Project Overview
This project analyzes stock performance while accounting for significant market-wide events. It uses a two-stage approach:
1. Identifies and catalogs major market-moving events
2. Analyzes technical indicators while considering these events as potential outliers

## Directory Structure
```
.
├── Economic_EVENTS/              # Economic indicator data
│   ├── federal_funds_rate.json
│   ├── treasury_yield.json
│   ├── inflation.json
│   ├── unemployment.json
│   ├── retail_sales.json
│   ├── nonfarm_payroll.json
│   ├── gdp.json
│   └── durables.json
│
├── AMD_DATA/                     # Stock-specific data
│   ├── stock_prices.json        # Raw stock price data
│   ├── SMA_daily.json          # Simple Moving Average data
│   ├── EMA_daily.json          # Exponential Moving Average data
│   ├── MACD_daily.json         # MACD indicator data
│   ├── ADX_daily.json          # Average Directional Index data
│   └── WMA_daily.json          # Weighted Moving Average data
│
├── date_organizer.py            # Processes economic events and identifies market-moving dates
├── market_date_processor.py     # Creates analysis prompts considering market events
├── analysis_prompt.json         # Generated structured analysis prompt
├── analysis_prompt.txt          # Human-readable analysis prompt
├── market_moving_dates.txt      # List of significant market dates
├── economic_analysis.txt        # Detailed economic event analysis
├── economic_events.json         # Structured economic event data
└── .env                        # API keys and configuration
```

## Key Files and Their Functions

### Core Processing Files
- `date_organizer.py`: 
  - Analyzes economic event data
  - Identifies significant market-moving dates
  - Uses o1-mini model for initial analysis
  - Outputs: economic_analysis.txt, economic_events.json, market_moving_dates.txt

- `market_date_processor.py`:
  - Creates sophisticated analysis prompts
  - Integrates market events with technical analysis
  - Handles both JSON and TXT data formats
  - Outputs: analysis_prompt.json, analysis_prompt.txt

### Data Files
- `Economic_EVENTS/*.json`:
  - Contains historical economic indicator data
  - Each file represents a different economic metric
  - Used to identify market-moving events

- `AMD_DATA/*.json`:
  - Stock-specific technical indicators
  - Price history and various technical analysis data
  - Used for detailed stock analysis

### Output Files
- `market_moving_dates.txt`: List of dates with significant market impact
- `economic_analysis.txt`: Detailed analysis of economic events
- `economic_events.json`: Structured data of economic events
- `analysis_prompt.json`: Machine-readable analysis template
- `analysis_prompt.txt`: Human-readable analysis template

## Configuration
- `.env` file should contain:
  ```
  OPENAI_API_KEY=your_openai_key_here
  Alpha Vantage API key goes into "stockandtechnicals.py" at the bottom
  ```

## Usage Flow
1. Run `date_organizer.py` first to process economic events
2. Run `market_date_processor.py` to create analysis prompts
3. Use generated prompts for stock-specific analysis

## Dependencies
- OpenAI's o1-mini model for data analysis
- Alpha Vantage API for market data
- Python packages:
  - langchain
  - pandas
  - python-dotenv
  - langgraph

## Data Formats
- Stock data: JSON format
- Technical indicators: JSON or TXT format
- Economic events: JSON format
- Generated analysis: Both JSON and TXT formats

## Important Notes
- Always run date_organizer.py before market_date_processor.py
- Keep API keys secure in .env file
- Ensure all data files follow the expected format
- Check economic_analysis.log for processing warnings and errors

## Future Development
- Add parallel processing for multiple stocks
- Implement real-time data updates
- Enhance market event correlation analysis
- Add more technical indicators

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact
[Your contact information here]
