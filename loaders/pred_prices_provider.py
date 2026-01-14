import datetime

import pandas as pd
from predictions.data.model_def import ModelDef
from loaders.pred_prices_loader_base import PredPricesLoaderBase


class PredPricesProvider:
    
    def __init__(
        self,
        predicted_prices_loader: PredPricesLoaderBase
    ):
        self.predicted_prices_loader = predicted_prices_loader

    def get_prices_np(self, model_def: ModelDef, symbol: str, date: datetime.datetime, hist_cnt: int):
        df  = self.predicted_prices_loader.load(model_def)
        idx = pd.IndexSlice
        n = df.loc[idx[[symbol], :date], :]['close'].tail(hist_cnt).to_numpy().squeeze()
        return n
    
    def get_models_defs(self):
        return self.predicted_prices_loader.all_models_defs
