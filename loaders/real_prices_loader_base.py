from typing import List
import pandas as pd

class RealPricesLoaderBase:
    def get_symbols(self) -> List[str]:
        raise Exception('Not implemented')

    def load(self, symbol: str) -> pd.DataFrame:
        raise Exception('Not implemented')
    
    def df(self) -> pd.DataFrame:
        raise Exception('Not implemented')
