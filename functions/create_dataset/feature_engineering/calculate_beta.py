import pandas as pd
from pathlib import Path
import numpy as np

def get_beta(dataframe):
    percent_change = dataframe.pct_change(1)

    covariance = percent_change.cov()
    beta = covariance['market']/percent_change['market'].var()
    return beta

def calculate_beta_for_dataframe():
    stock_prices = pd.read_csv(
        Path('data/interim/updated_stock_prices.csv'),
        usecols=['date', 'ticker',  'close'],
        parse_dates=['date'],
        )

    seperated_tickers = stock_prices.pivot_table(
        index='date',
        columns='ticker',
        values='close'
        )

    market = pd.read_csv(
        Path('data/original/stock/^GSPC.csv'),
        usecols=['Date', 'Close'],
        parse_dates=['Date'],
        )

    market = market.rename(columns={'Close': 'market', 'Date': 'date'})

    beta_test = seperated_tickers.merge(market, on='date')
    beta_test = beta_test.set_index('date')

    beta_test = beta_test['2008Q4':]

    # Seperate rows by quarter

    quarters = beta_test.index.to_period('Q').unique()

    test_quarters_df = pd.DataFrame()

    for quarter in quarters:
        quarter_end = quarter.strftime('%FQ%q')
        test_get_beta = get_beta(beta_test[quarter_end])
        test_quarters_df[quarter.asfreq('d' ,how='end')] = test_get_beta

    rename_index = test_quarters_df.rename_axis(mapper='ticker').rename_axis(mapper='period', axis=1)
    test_transform = rename_index.unstack()
    unstacked_df = test_transform.reset_index().rename(columns={0: 'beta'})

    # reseting the period column to datetime
    unstacked_df['period'] = pd.to_datetime(
        arg = unstacked_df['period'].astype('str'),
        format='%Y-%m-%')

    #
    unstacked_df.loc[:,'next_period'] = (
        (unstacked_df.loc[:,'period']
        + pd.DateOffset(days=90)
        ).dt.to_period('Q').dt.end_time)

    return unstacked_df
