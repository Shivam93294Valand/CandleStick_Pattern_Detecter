# Stock Market Data Repository

## Overview
This directory contains historical stock market data for technical analysis and pattern recognition. The data is organized by company and date, with minute-by-minute price information that can be analyzed using various technical indicators and candlestick patterns.

## Data Structure

### Directory Organization
- `5Scripts/` - Contains data for 5 major stocks
  - `BAJAJ-AUTO/` - Bajaj Auto stock data
  - `BHARTIARTL/` - Bharti Airtel stock data
  - `ICICIBANK/` - ICICI Bank stock data
  - `RELIANCE/` - Reliance Industries stock data
  - `TCS/` - Tata Consultancy Services stock data

### File Naming Convention
Files are named according to the date format: `DD-MM-YYYY.csv`

### Data Format
Each CSV file contains the following columns:
- `date` - Trading date (DD-MM-YYYY)
- `time` - Time of the day (HH:MM:SS)
- `open` - Opening price for the time period
- `high` - Highest price during the time period
- `low` - Lowest price during the time period
- `close` - Closing price for the time period
- `volume` - Trading volume
- `oi` - Open Interest
- `exchangecode` - Code for the exchange
- `symbolcode` - Code for the symbol
- `expiry` - Expiry date for derivatives

## Usage
This data is used by the Hack_Maya application for:
1. Technical analysis of stock price movements
2. Candlestick pattern recognition including:
   - Hammer
   - Dragonfly Doji
   - Rising Window
   - Evening Star
   - Three White Soldiers
3. Time-based analysis with various timeframes:
   - 1 Minute
   - 5 Minutes
   - 10 Minutes
   - 15 Minutes
   - 30 Minutes
   - 1 Hour

## Data Processing
The application loads and processes this data using pandas for data manipulation and resampling to different timeframes. The processed data is then analyzed for various technical patterns using either TA-Lib or custom implementations.

## Note
The data in this repository appears to be future-dated (2025), suggesting it may be simulated or test data rather than actual historical market data.