# Predicting Quarterly Stock Return Type with SEC Data
## Color Meanings
<font color='red'>Update files</font>
</br>
<font color='orange'>Change Wording</font>

## Introduction
The objective of this project is to predict the quarterly return type of a stock. The data contains financial ratios from the SEC financial and notes datasets, previously scrapped stock data, and daily stock information from the iexfinance.stocks api. The dates range from Q1 2009 to Q3 2018

## Creating the Dataset
The data for this project is not in a database or a single file. It's spread across many files. So our first major step for this project is to gather all the data into a single dataset.

### Stock Data


1.  Historical Stock Data from <https://www.kaggle.com/ehallmar/daily-historical-stock-prices-1970-2018>
2.  Using historical_stocks.csv we can get more the most up-to-date stock prices using the iexfinance.stocks api
3.  Finally we can get an approximation of the overall market prices by downloaded S&P 500 prices(^GSPC.csv) from yahoo finance

### SEC Data
Each quarter the SEC publishes the 10-K and 10-Q filings for every publicly traded company in a xbrl file. These filings include the financial statements for the companies.

1.  Download all quarterly zip files from <https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html>

## Transforming the Data

### Stock Data

Since the SEC data only goes back to 2009 we don't need any of the stock data before that point. We'll also need to filter out all preferred stocks from our stock ticker database to leave us only with common stocks.

1.  Use filter_stock_prices_by_date from processing_scripts\cleaning\filter_stocks.py to filter out stocks
2.  Use filter_stock_metadata from processing_scripts\cleaning\filter_stocks.py to filter stocks to only common stock tickers
3.  Next we have to get the newest stock data by using update_stock_prices <font color='red'>**Need to update because it needs to be able to pull from a none updated file as well as the updated one**</font>

4.  Finally we only need the end prices for each quarter. To get that we'll use filter_qtr_end_stock_prices from processing_scripts\cleaning\filter_stocks.py.


### SEC DATA
The quarterly files are in zip files. So first we need to unzip them before we process them. For this project we only need to the sub and num files. These files contain the data most relevant to our <font color='orange'>**objective**</font>.

#### sub.tsv
This file contains every filings given to submitted to the SEC.
  ##### Columns We Will Use
-   **adsh** - unique key for each filings
-   **name** - name of the company
-   **cik** - Central Index Key 10 digit number assigned to each registrant who submits filings
-   **countryinc** - The country of incorporation for the registrant
-   **form** - the submission type of the filing
-   **filed** - the date of the filing
-   **period** - Balance Sheet Date
-   **fy** - The fiscal year of the submission

#### num.tsv
This file contains the numeric data for every submission file. Each row is one data point in a financial statement.
##### Columns We Will Use
-   **adsh** - unique key for each filings
-   **qrts** - the number of quarters the value takes from
-   **tag** - the unique identifier for the value
-   **uom** - unit of measure for the value
-   **value** - the numeric amount associated with the tag
-   **dimh** - key for dimensional information, mainly used to breakdown number into regional values
-   **iprx** - A positive integer to distinguish different reported facts that  otherwise would have the same primary key
-   **ddate** - The end date for the data value, rounded to the
nearest month  end.

### Feature Extraction
Now is the time to decided the ultimate financial ratio features for we want to use in our dataset. To help us decided which financial ratios we want to use we're going to look several financial ratios and there availability in the files.

price-to-earnings -  price per share / earning per share
price-to-book - price per share / stockholders equity
return-on-assets - net income /total assets
return-on-equity - net income stockholder equity
profit-margin - net income / sales
current-ratio - current assets / current liabilities
quick-ratio - (current assets - inventory) / current liabilities
debt-to-equity - total liabilities / total stockholders_equity
asset-turnover - sales / total assets
operating-cash-flow-ratio - cash flows from operations / current liabilities
price-cash-flow-ratio - share price / Operating cash flow per share 





## Functions walk-through



### filter_sec_files.py
#### filter_sub
filter_sub reduces the submission filings to only those that are a 10-K/Q from the US and where the period is the most used in the file

#### filter_num
This function returns the basis for the majority of the financial ratio features for our dataset.

filter_num returns the

EarningsPerShareBasic -
WeightedAverageNumberOfSharesOutstandingBasic
StockholdersEquity
NetIncomeLoss
Assets
Liabilities
