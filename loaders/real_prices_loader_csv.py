import os
from typing import List
import pandas as pd

from loaders.real_prices_loader_base import RealPricesLoaderBase

class RealPricesLoaderCSV(RealPricesLoaderBase):
    
    __cache = {}
    
    def __init__(self, prices_csv_file: str):
        self.__df = pd.read_csv(prices_csv_file)
        self.__df['date'] = pd.to_datetime(self.__df['date'], utc=True, format='%Y-%m-%d %H:%M:%S')
        # self.__df = self.__df[['date', 'symbol', 'open', 'close', 'volume', 'buy_volume', 'sell_volume']]
        self.symbols: List[str] = self.__df['symbol'].unique().tolist()
        self.symbols.sort()
    
    def df(self) -> pd.DataFrame:
        return self.__df
        
    def get_symbols(self) -> List[str]:
        return self.symbols
    
    def load(self, symbol: str) -> pd.DataFrame:
        if symbol in RealPricesLoaderCSV.__cache:
            return RealPricesLoaderCSV.__cache[symbol]
    
        df = self.__df[self.__df['symbol'] == symbol]
        
        if symbol not in RealPricesLoaderCSV.__cache:
            RealPricesLoaderCSV.__cache[symbol] = {}
        RealPricesLoaderCSV.__cache[symbol] = df
        
        return df
