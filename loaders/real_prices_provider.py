from dataclasses import dataclass
import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
from loaders.real_prices_loader_base import RealPricesLoaderBase

@dataclass
class PricesHistItem:
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray


class RealPricesProvider:
    
    def __init__(self, real_prices_loader: RealPricesLoaderBase, filter_symbols: Optional[List[str]] = None):
        self.real_prices_loader = real_prices_loader
        self.filter_symbols = filter_symbols

    def get_prices_df(self, symbol: str):
        return self.real_prices_loader.load(symbol)
        
    def get_prices_np(self, symbol: str, date: datetime.datetime, hist_cnt: int):
        # Example usage:
        df = self.get_prices_df(symbol)  # or your DataFrame with many symbols
        # If df_raw already has MultiIndex, reset index to get symbol and date as columns:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # df = self.interpolate_zeros_by_time(df, date_col='date', symbol_col='symbol', value_col='close', freq='D', fill_edges=False)
        # df = self.interpolate_zeros_by_time(df, date_col='date', symbol_col='symbol', value_col='open', freq='D', fill_edges=False)
        # df = self.interpolate_zeros_by_time(df, date_col='date', symbol_col='symbol', value_col='volume', freq='D', fill_edges=False)

        # If you want to restore MultiIndex (symbol, date):
        df = df.set_index(['symbol', 'date']).sort_index()

        if df['close'].eq(0).any():
            raise RuntimeError(f'{symbol} zero value')

        idx = pd.IndexSlice
        n = df.loc[idx[[symbol], :date], :]['close'].tail(hist_cnt).to_numpy().squeeze()

        return n

    def get_volumes_np(self, symbol: str, date: datetime.datetime, hist_cnt: int):
        df  = self.get_prices_df(symbol)
        df = df.set_index(['symbol', 'date'])
        idx = pd.IndexSlice
        n = df.loc[idx[[symbol], :date], :]['volume'].tail(hist_cnt).to_numpy().squeeze()
        return n

    def get_prices_hist_item(self, symbol: str, date: datetime.datetime, hist_cnt: int):
        df  = self.get_prices_df(symbol)
        df = df.set_index(['symbol', 'date'])
        df2 = df.loc[pd.IndexSlice[[symbol], :date], :]
        return PricesHistItem(
            open=df2['open'].tail(hist_cnt).to_numpy().squeeze(),
            close=df2['close'].tail(hist_cnt).to_numpy().squeeze(),
            volume=df2['volume'].tail(hist_cnt).to_numpy().squeeze(),
            buy_volume=df2['buy_volume'].tail(hist_cnt).to_numpy().squeeze(),
            sell_volume=df2['sell_volume'].tail(hist_cnt).to_numpy().squeeze()
        )
    
    def get_symbols(self) -> List[str]:
        if self.filter_symbols is not None:
            return self.filter_symbols
        return self.real_prices_loader.get_symbols()

    def interpolate_zeros_by_time(self, df, date_col='date', symbol_col='symbol', value_col='close', freq='D', fill_edges=False):

        df = df.copy()

        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

        # Ensure the value column is numeric (this handles pd.NA!)
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

        results = []
        for symbol, sub in df.groupby(symbol_col, sort=True):

            sub = sub.sort_values(date_col)

            # Replace zeros with NaN AFTER converting to numeric
            sub[value_col] = sub[value_col].replace(0, np.nan)

            # Build full date range
            start, end = sub[date_col].min(), sub[date_col].max()
            full_index = pd.date_range(start, end, freq=freq, tz=start.tz)

            # Create a numeric Series indexed by date
            s = sub.set_index(date_col)[value_col]

            # Reindex → interpolate → back to original dates
            s_full = s.reindex(full_index)

            s_interp = s_full.interpolate(method='timeseries', limit_area='inside')

            if fill_edges:
                s_interp = s_interp.fillna(method='bfill').fillna(method='ffill')

            # return only original timestamps
            s_final = s_interp.reindex(s.index)

            # Reattach to sub
            sub[value_col] = s_final.values
            results.append(sub)

        df_out = pd.concat(results).sort_values([symbol_col, date_col])

        return df_out
