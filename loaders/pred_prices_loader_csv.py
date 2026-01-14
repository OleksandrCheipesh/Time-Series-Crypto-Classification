import os
from typing import List, Optional
import pandas as pd

from predictions.data.model_def import ModelDef
from loaders.pred_prices_loader_base import PredPricesLoaderBase

class PredPricesLoaderCSV(PredPricesLoaderBase):
    
    def __init__(
        self,
        prices_base_dir: str,
        filter_models: Optional[List[str]] = None
    ):
        super().__init__()
        self.prices_base_dir = prices_base_dir
        self.__cache = {}
    
        csv_files_path = f'{self.prices_base_dir}'
        csv_files = [f for f in os.listdir(csv_files_path) if os.path.isfile(os.path.join(csv_files_path, f))]
        
        for f in csv_files:
            arr = f.removesuffix('.csv').split('__')
            model_def = ModelDef(
                name=arr[0],
                ctx=int(arr[1].split('_')[1]),
            )
            
            if filter_models is not None and model_def.name not in filter_models:
                continue
            
            model_id = model_def.get_model_id()
            
            if model_id not in self.all_models_defs:
                self.all_models_defs[model_id] = model_def
            
            pred_df = pd.read_csv(os.path.join(csv_files_path, f))
            pred_df['date'] = pd.to_datetime(pred_df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)
            pred_df = pred_df.set_index(['symbol', 'date'])
            
            self.__cache[model_id] = pred_df
            
    def load(self, model_def: ModelDef) -> pd.DataFrame:
        return self.__cache[model_def.get_model_id()]


class PredPricesLoaderCSV2(PredPricesLoaderBase):

    def __init__(
            self,
            prices_file: str,
            filter_models: Optional[List[str]] = None
    ):
        super().__init__()
        self.__cache = {}

        main_df = pd.read_csv(prices_file)

        for (model, ctx_len), pred_df in main_df.groupby(['model', 'ctx']):
            model_def = ModelDef(
                name=model,
                ctx=ctx_len,
            )

            if filter_models is not None and model_def.name not in filter_models:
                continue

            model_id = model_def.get_model_id()

            if model_id not in self.all_models_defs:
                self.all_models_defs[model_id] = model_def

            pred_df['date'] = pd.to_datetime(pred_df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)
            pred_df = pred_df.set_index(['symbol', 'date'])
            pred_df.sort_index(ascending=True, inplace=True)
            pred_df.rename(columns={'predicted': 'close'}, inplace=True)
            pred_df.drop(columns=['model', 'ctx'], inplace=True)

            self.__cache[model_id] = pred_df

    def load(self, model_def: ModelDef) -> pd.DataFrame:
        return self.__cache[model_def.get_model_id()]
