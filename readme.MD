# AGI Fund Technical Analysis Pipeline

A sophisticated pipeline for analyzing stock market data and economic indicators using technical analysis and machine learning.

## Overview

This project implements a three-stage pipeline for comprehensive market analysis:

1. **Data Collection & Technical Analysis** (`stock_technicals_current.py`)
   - Fetches historical stock data and calculates technical indicators
   - Supports multiple timeframes and granularities
   - Includes a wide range of technical indicators (Moving Averages, Momentum, Volatility, Volume)
   - Creates organized directory structure for data storage

2. **Economic Event Analysis** (`econ_formatting.py`)
   - Analyzes economic indicators and events
   - Uses AI to identify significant market-moving events
   - Calculates event severity and potential market impact
   - Generates comprehensive economic analysis reports

3. **Technical Analysis Prompt Generation** (`prompt_creator.py`)
   - Creates sophisticated analysis prompts combining technical and fundamental data
   - Integrates market events with technical indicators
   - Generates structured analysis templates for consistent evaluation
   - Supports multiple technical indicators and timeframes

## Directory Structure

```
.
├── Economic_EVENTS/           # Economic indicator data
│   ├── cpi.json
│   ├── gdp.json
│   └── ...
├── [TICKER]_HISTORICAL/      # Stock-specific data directories
│   ├── YYYY_MM/             # Monthly data organization
│   │   ├── TICKER_DATA.json # Price data
│   │   ├── sma.json        # Technical indicators
│   │   └── ...
├── stock_technicals_current.py
├── econ_formatting.py
├── prompt_creator.py
└── requirements.txt
```

## Features

### Technical Analysis
- Support for multiple technical indicators:
  - Moving Averages (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA)
  - Momentum Indicators (MACD, RSI, Stochastic, ADX)
  - Volatility Indicators (Bollinger Bands, ATR)
  - Volume Indicators (OBV, ADOSC)
- Customizable timeframes and intervals
- Automated data organization by time periods

### Economic Analysis
- Processing of major economic indicators:
  - GDP
  - Inflation
  - Employment data
  - Interest rates
  - Retail sales
- AI-powered event significance analysis
- Market impact assessment
- Comprehensive reporting system

### Prompt Generation
- Integration of technical and fundamental analysis
- Customizable analysis templates
- Support for multiple analysis approaches
- Structured output for consistent evaluation

## Usage

1. Data Collection:
```python
python stock_technicals_current.py
```

2. Economic Analysis:
```python
python econ_formatting.py
```

3. Generate Analysis Prompts:
```python
python prompt_creator.py
```

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`
- Alpha Vantage API key for data collection

## Output Files

The pipeline generates several output files:
- `economic_analysis.txt`: Detailed economic event analysis
- `economic_events.json`: Structured economic data
- `major_market_events.txt`: Significant market events analysis
- `market_moving_dates.txt`: Key dates affecting market movements
- `analysis_prompt.json`: Generated analysis prompts
- `analysis_prompt.txt`: Human-readable analysis templates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
