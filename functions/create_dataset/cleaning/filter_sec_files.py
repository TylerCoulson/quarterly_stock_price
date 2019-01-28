import zipfile
from pathlib import Path
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format

def filter_sub(file_path):
    """filter and transform submission files from sec"""

    # Importing submissions data
    dtypes = {
    'adsh':'str',
    'name':'str',
    'cik':'str',
    'countryinc':'str',
    'form':'str',
    'filed':'str',
    'period':'str',
    'fy':'str',}

    sub = pd.read_table(
        file_path,
        usecols=dtypes.keys(),
        dtype=dtypes,
        parse_dates=['period', 'filed']
        )

    # sub filters
    form = ((sub['form'] == '10-Q') | (sub['form'] == '10-K'))
    period = (sub['period'].isin(sub['period'].mode()))
    country = (sub['countryinc'] == 'US')

    # Filtering sub dataframe
    filtered_sub = sub[country & form & period]

    return filtered_sub.reset_index(drop=True)

def filter_num(filepath, tags=['EarningsPerShareBasic', 'WeightedAverageNumberOfSharesOutstandingBasic', 'StockholdersEquity', 'NetIncomeLoss', 'Assets', 'Liabilities']):

    """filter and transform num files from sec"""
    dtypes = {
    'adsh' : 'str',
    'qtrs' : 'str',
    'tag' : 'str',
    'uom' : 'str',
    'value' : 'float',
    'dimh' : 'str',
    'iprx' : 'str',
    'ddate' : 'str'}

    filepath = 'data/original/test_files/num.tsv'

    num = pd.read_table(
        filepath,
        usecols=dtypes.keys(),
        dtype=dtypes,
        parse_dates=['ddate'])

    # num filters
    value = num['value'].notna()
    qtrs = (num['qtrs'] <= "1")
    uom  = ((num['uom'] == "USD") | (num['uom'] == "shares"))
    dimh =  num['dimh'] == '0x00000000'
    iprx = num['iprx'] == '0'
    ddate = (num['ddate'].isin(num['ddate'].mode()))

    filtered_num = num[value & qtrs & uom & dimh & iprx & ddate]

    # Financial Ratios Items
    for tag in tags:
        filtered_num = filtered_num[filtered_num['tag'] == tag]

    financial_ratio_factors = (
        filtered_num[[
            "adsh",
            "tag",
            "uom",
            "value",
            'ddate',
            'qtrs']].drop_duplicates())

    financial_ratio_factors = financial_ratio_factors.drop(
        financial_ratio_factors[(
            (financial_ratio_factors[['adsh', 'tag']].duplicated(keep=False)) & (financial_ratio_factors['qtrs'] == '0'))].index)

    pivot_ratio_factors = financial_ratio_factors.pivot(
        index='adsh',
        columns='tag',
        values='value'
        )

    pivot_ratio_factors = pivot_ratio_factors.reset_index('adsh')

    return pivot_ratio_factors

def extract_zipfile(filepath, filename):

    with zipfile.ZipFile(filepath) as file:

        unziped_file = file.open(filename)

    return unziped_file

def unzip_and_combine(filestem, tags=None):

    filestems = {
        'num': filter_num,
        'sub': filter_sub
    }

    df = pd.DataFrame()
    zip_files = Path('data/original/sec_zip_files')


    for zip_file in zip_files.iterdir():
        if zipfile.is_zipfile(zip_file):
            extracted_file = extract_zipfile(zip_file, f'{filestem}.tsv')
            if filestem == 'num' & tags != None:
                extracted_file = filestems[filestem](extracted_file, tags=tags)
            else:
                extracted_file = filestems[filestem](extracted_file)
            print(f'{zip_file.stem}/{filestem}.tsv')

        else:
            pass

        df = df.append(extracted_file, ignore_index=True, sort=True)

    return df


def create_sec_df(tags=None):
    sub_df = unzip_and_combine('sub')
    num_df = unzip_and_combine('num', tags)

    df = sub_df.merge(num_df, on=['adsh'], how='inner')

    return df
