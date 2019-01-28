# Prediction Quarterly Stock Return Type with SEC Data

## Introduction
The objective of this project is to predict the quarterly return type of a stock. The data contains financial ratios from the SEC financial and notes datasets, previously scrapped stock data, and daily stock information from the iexfinance.stocks api. The dates range from Q1 2009 to Q3 2018

## Approach
I decided early on for this project that it didn't matter much whether I got the exact price for the return but it is more important that they fall into the correct categories. So I broke the log rate of returns into 5 categories.

1.  **High negative return** - log rate of return is <= -0.10
2.  **Negative return** - log rate of return between -0.10 and -0.0075
3.  **No return** - log rate of return between -0.0075 and 0.0075
4.  **Positive return** - log rate of return between 0.0075 and 0.10
5.  **High positive return** - log rate of return is >= 0.10

The reasoning for choosing 5 categories instead of simply positive return and negative return is that we want to know the degree of the return. The rates themselves are mostly arbitrary. I chose ±0.10 because many consider that to be an average return for a yearly return. So getting that return for a single quarter is a good measure for declaring a high return. The no return range of ±0.0075 is a quarter of the historic annual rate of inflation.

The features of the dataset are mainly different financial ratios or the sector of the economy that the company falls into. The only other column of interest is the beta which is a measure of the volatility/risk of a stock during the stock previous quarter.
