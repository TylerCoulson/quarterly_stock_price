import pandas as pd
from pathlib import Path
from datetime import datetime
import iexfinance.stocks as stock_api
import time


pd.options.display.float_format = '{:,.3f}'.format

def filter_stock_metadata():
    """fitler metadata from historical_stocks.csv"""

    # read in historical_stocks
    stock_metadata = pd.read_csv(Path("data/original/stock/historical_stocks.csv"), usecols=["name", "ticker", "sector"])

    # Drop non-common stock
    filtered_stock_metadata = stock_metadata[~(stock_metadata['ticker'] == 'Symbol')]
    filtered_stock_metadata = filtered_stock_metadata.drop_duplicates(["name"])

    # Convert HTML number to it's symbol
    filtered_stock_metadata['name'] = filtered_stock_metadata['name'].str.replace("&#39;", "'")

    return filtered_stock_metadata

def filter_stock_prices_by_date():
    """Filters the historical_stock_prices dataset to prices after Q3 2008"""

    # import the historical stock prices datset, taken from Kaggle
    historical_stock_prices = pd.read_csv(
        Path('data/original/stock/historical_stock_prices.csv'),
        parse_dates=['date'],)

    # make columns into approriate format
    historical_stock_prices.columns = historical_stock_prices.columns.str.lower().str.replace(" ", "_")

    # filter columns to those compatiable with iexfinance api
    historical_stock_prices = historical_stock_prices[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

    # filter to dates after Q3 2008
    mask_q3_2008 = historical_stock_prices['date'] >= '9-30-2008'
    prices_after_2009 = historical_stock_prices[mask_q3_2008]

    return prices_after_2009

def update_stock_prices():
    """Uses the iexfinance api to get historical stock price date up to 5 years ago"""

    if Path('data/interim/updated_stock_prices.csv'):
        stock_prices = pd.read_csv(
            'data/interim/updated_stock_prices.csv',
            parse_dates=['date'])
    else:
        stock_prices = filter_stock_prices_by_date()

    tickers = stock_prices['ticker'].unique()

    start = stock_prices['date'].max()
    end = datetime.today().date()

    completed_calls = pd.DataFrame()
    for ticker in tickers:
        df = stock_api.get_historical_data(ticker, start, end, output_format='pandas')
        df['ticker'] = ticker
        df = df.reset_index()
        completed_calls = pd.concat([completed_calls, df], ignore_index=True)
        time.sleep(.1)

    completed_calls[['open', 'high', 'low', 'close']] = completed_calls[['open', 'high', 'low', 'close']].round(2)

    complete_stock_prices = pd.concat([completed_calls, stock_prices], ignore_index=True,)

    complete_stock_prices = complete_stock_prices[~(complete_stock_prices.duplicated())]

    complete_stock_prices = complete_stock_prices.sort_values(['ticker', 'date'])

    return complete_stock_prices

def filter_qtr_end_stock_prices():
    """Return quarter end stock prices"""

    # Read in stock_metadata
    stock_metadata = pd.read_csv(
        Path('data/interim/stock_metadata.csv')
    )

    # Read in historical_stock_prices data
    stock_price = pd.read_csv(
        Path('data/interim/updated_stock_prices.csv'),
        parse_dates=['date']
    )
    stock_price = stock_price.rename(columns={'close':'stock_price',})

    # Filter rows to the last trading day for each quarter
    months = (3,6,9,12)

    qtr_end_price = stock_price[(
        (stock_price['date'].dt.month.isin(months)) &
        (stock_price['date'].isin(
            stock_price['date'].groupby(by=[
            stock_price['date'].dt.year,
            stock_price['date'].dt.month]).max()
        )))]

    # Add end date of quarter
    qtr_end_price['period'] = (
        qtr_end_price['date'].dt.to_period('Q').dt.end_time)

    # Creates a next_period column
    qtr_end_price['next_period'] = (
        (qtr_end_price['period']
        + pd.DateOffset(days=90)
        ).dt.to_period('Q').dt.end_time)

    # Adds the stock price corresponding to the next_period column
    qtr_end_price = (
        qtr_end_price.merge(
            right = qtr_end_price[['ticker', 'stock_price', 'period']],
            how='left',
            left_on=['ticker','next_period'],
            right_on=['ticker','period'],
            suffixes=('','_x')
        ))

    # renaming and reordering the columns for better readability
    qtr_end_price = qtr_end_price.rename(
        columns={'stock_price_x' : 'next_period_price'})

    qtr_end_price = qtr_end_price[['ticker', 'period', 'stock_price', 'next_period', 'next_period_price',]]

    # Adds name and sector columns from the stock_metadata df
    qtr_end_with_metadata = stock_metadata.merge(qtr_end_price, how='inner', on='ticker')

    # Rounding data for export
    qtr_end_with_metadata = qtr_end_with_metadata.round(3)

    return qtr_end_with_metadata
