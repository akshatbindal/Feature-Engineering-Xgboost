import yfinance as yf
import ta
import pandas as pd

def fetch_data(start_date, end_date, stock_index="^NSEI"):
    data = yf.download(stock_index, start=start_date, end=end_date)
    data['Date'] = data.index
    return data

def calculate_technical_indicators(data):
    data['SMA'] = ta.trend.SMAIndicator(close=data['Close'], window=14).sma_indicator()
    data['EMA'] = ta.trend.EMAIndicator(close=data['Close'], window=14).ema_indicator()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(close=data['Close']).macd_diff()
    data['ADX'] = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14).adx()
    data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14).cci()
    data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()
    data['Stochastic'] = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14).stoch_signal()
    data['Bollinger_High'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_hband()
    data['Bollinger_Low'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_lband()
    data['Momentum'] = ta.momentum.ROCIndicator(close=data['Close'], window=10).roc()
    data['TRIX'] = ta.trend.TRIXIndicator(close=data['Close'], window=15).trix()
    data['MFI'] = ta.volume.MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14).money_flow_index()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Rolling_Std_7'] = data['Close'].rolling(window=7).std()
    return data
