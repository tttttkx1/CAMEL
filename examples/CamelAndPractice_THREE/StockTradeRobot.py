import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf

class StockTradingBot:
    def __init__(self, ticker='AAPL', period='1y'):
        self.ticker = ticker
        self.period = period
        self.model = RandomForestClassifier(n_estimators=100)
        self.data = None
        
    def fetch_data(self):
        """
        从雅虎财经获取指定股票的历史数据
        """
        self.data = yf.download(self.ticker, period=self.period)
        self.data.dropna(inplace=True)
        return self.data
    
    def prepare_features(self):
        """
        准备特征数据，包括移动平均线、RSI等技术指标
        """
        # 计算5日和20日移动平均线
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        
        # 计算相对强弱指数(RSI)
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # 创建目标变量（未来5天的收益率）
        self.data['Return'] = self.data['Close'].pct_change(5).shift(-5)
        self.data['Target'] = (self.data['Return'] > 0).astype(int)
        
        # 删除缺失值
        self.data.dropna(inplace=True)
        
        return self.data[
            ['MA5', 'MA20', 'RSI']
        ]
    
    def split_data(self):
        """
        划分训练集和测试集
        """
        X = self.prepare_features()
        y = self.data['Target']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        """
        训练机器学习模型
        """
        X_train, X_test, y_train, y_test = self.split_data()
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
    
    def predict_signal(self, data_point):
        """
        根据当前市场数据预测交易信号
        """
        signal = self.model.predict([data_point])
        return 'BUY' if signal[0] == 1 else 'SELL'
    
    def execute_trade(self, signal):
        """
        执行实际交易操作（此处为模拟交易）
        """
        print(f"执行 {signal} 操作")
    
    def run(self):
        """
        运行交易机器人
        """
        self.fetch_data()
        self.prepare_features()
        accuracy = self.train_model()
        print(f"模型准确率: {accuracy:.2f}")
        
        # 实时预测最新交易信号
        latest_data = self.data.iloc[-1][['MA5', 'MA20', 'RSI']]
        signal = self.predict_signal(latest_data)
        self.execute_trade(signal)
        
        return signal, accuracy