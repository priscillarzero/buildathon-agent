# ODCV Data Analyst

AI-powered root cause analysis for HVAC/BMS data.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## What You Need
- Python 3.9+
- Anthropic API key (get one at https://console.anthropic.com)
- BMS CSV data files

## Features
- Chat-based analysis of BMS data
- Automatic anomaly detection
- Interactive time-series visualization
- Root cause analysis for HVAC issues

## How It Works
1. Upload your BMS CSV files in the sidebar
2. Click "Load Data" to process them
3. Ask questions in natural language
4. Get AI-powered analysis with specific insights

## Example Questions
- "Why did OA CFM drop on November 15th?"
- "Are there any anomalies in the temperature data?"
- "Is the economizer working correctly?"
- "Why are we not tracking expected SOO values?"
