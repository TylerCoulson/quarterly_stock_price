import pandas as pd
from functions.create_dataset.cleaning.filter_sec_files import *
from functions.create_dataset.cleaning.filter_stocks import *
from functions.create_dataset.feature_engineering.calculate_beta import *

def create_sec_and_stock_dataset():

    # creates dataset of merged sec filings
    create_sec_df().to_csv('data/interim/merged_sec_files.csv', index=False)

    # creates a csv of stock_metadata
    filter_stock_metadata().to_csv('data/interim/stock_metadata.csv', index=False)

    # updates historic prices dataset
    update_stock_prices().to_csv('data/interim/updated_stock_prices.csv', index=False)

    # filter to quarter end stock prices
    qtr_end_price = filter_qtr_end_stock_prices()

    # re read in merged sec files
    merged_sec_files = pd.read_csv('data/interim/merged_sec_files.csv', parse_dates=['period'])

    stock_sec_data = merged_sec_files.merge(qtr_end_price, on=['name', 'period'])

    stock_sec_data.to_csv('data/interim/stock_sec_data.csv', index=False)

    # Create previous quarter daily Beta for Stocks
    calculate_beta_for_dataframe().to_csv('data/interim/stock_betas.csv', index=False)

    # Merge betas and stock_sec_data
    stock_sec_data = pd.read_csv('data/interim/stock_sec_data.csv', parse_dates=['period']) 

    stock_betas = pd.read_csv('data/interim/stock_betas.csv', parse_dates=['period'])

    beta_merged = stock_sec_data.merge(stock_betas, on=['period', 'ticker'], copy=False)

    # clean column names

    beta_merged = beta_merged[[
        'adsh',
        'cik',
        'name',
        'ticker',
        'sector',
        'period',
        'stock_price',
        'next_period_x',
        'next_period_price',
        'beta',
        'Assets',
        'EarningsPerShareBasic',
        'Liabilities',
        'NetIncomeLoss',
        'StockholdersEquity',
        'WeightedAverageNumberOfSharesOutstandingBasic',
        ]]

    rename_columns = {
        'next_period_x': 'next_period',
        'Assets': 'assets',
        'EarningsPerShareBasic': 'earnings_per_share_basic',
        'Liabilities': 'liabilities',
        'NetIncomeLoss': 'net_income_loss',
        'StockholdersEquity': 'stockholders_equity',
        'WeightedAverageNumberOfSharesOutstandingBasic': 'weighted_shares_outstanding_basic',
    }

    beta_merged = beta_merged.rename(columns=rename_columns)

    beta_merged.to_csv('data/processed_data/sec_and_stock_dataset.csv', index=False)
