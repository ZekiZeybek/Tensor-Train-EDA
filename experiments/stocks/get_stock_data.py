import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

def download_stock_data(api_key, tickers, 
                        start_date=None, 
                        end_date=None, output_file="all_stock_data.csv"):
    """
    Downloads daily stock data for a list of ticker symbols using Alpha Vantage API,
    stores them in a single CSV file, and filters by date range.

    Parameters:
        api_key (str): Your Alpha Vantage API key.
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to None (no filtering).
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to None (no filtering).
        output_file (str): Path to the CSV file to store all stock data. Defaults to 'all_stock_data.csv'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the stock data for all tickers.
    """
    # Initialize Alpha Vantage TimeSeries client
    ts = TimeSeries(key=api_key, output_format="pandas")

    # Check if the output file exists
    if os.path.exists(output_file):
        print(f"Loading existing data from {output_file}")
        all_data = pd.read_csv(output_file, index_col="date", parse_dates=True)
    else:
        all_data = pd.DataFrame()

    for ticker in tickers:
        if ticker in all_data["ticker"].unique() if not all_data.empty else []:
            print(f"Data for {ticker} already exists in {output_file}")
            continue

        print(f"Downloading data for {ticker}...")
        try:
            # Fetch daily stock data
            data, _ = ts.get_daily(symbol=ticker, outputsize="full")
            data.index.name = "date"
            data["ticker"] = ticker  # Add a column for the ticker symbol

            # Filter by date range if specified
            if start_date or end_date:
                data = data.loc[start_date:end_date]

            # Append to the main DataFrame
            all_data = pd.concat([all_data, data])
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
            continue

    # Save the combined data to the output file
    all_data.to_csv(output_file)
    return all_data

if __name__ == "__main__":
    with open("api.txt", "r") as file:
        API_KEY = file.read().strip()
    # API_KEY = "your_alpha_vantage_api_key"
    # List of 30 famous ticker symbols (e.g., from the Dow Jones Industrial Average)
    FAMOUS_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "JNJ", "V",
        "WMT", "PG", "JPM", "UNH", "HD", "MA", "DIS", "PYPL", "NFLX", "KO",
        "PEP", "XOM", "CVX", "MRK", "ABBV", "T", "PFE", "CSCO", "INTC", "ADBE"
    ]

    # List of 70 less popular ticker symbols
    LESS_POPULAR_TICKERS = [
        "F", "GE", "GM", "UAL", "DAL", "AAL", "NCLH", "CCL", "RCL", "UAL",
        "BBBY", "AMC", "GME", "PLTR", "SNAP", "TWLO", "SQ", "ROKU", "UBER", "LYFT",
        "DKNG", "PINS", "ZM", "DOCU", "CRWD", "OKTA", "ZS", "NET", "DDOG", "MDB",
        "SHOP", "ETSY", "SPOT", "BIDU", "BABA", "JD", "NIO", "XPEV", "LI", "BYND",
        "PTON", "WORK", "FSLY", "NKLA", "QS", "RIVN", "LCID", "SOFI", "AFRM", "UPST",
        "DNA", "ARKK", "ARKG", "ARKF", "ARKW", "ARKQ", "ARKX", "SPCE", "ASTR", "BLNK",
        "CHPT", "PLUG", "FCEL", "BE", "RUN", "SEDG", "ENPH", "TSLAQ", "NKTR", "CRSP"
    ]

    TICKERS = FAMOUS_TICKERS + LESS_POPULAR_TICKERS
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"

    stock_data = download_stock_data(API_KEY, TICKERS, start_date=START_DATE, end_date=END_DATE)
