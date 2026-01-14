import pandas as pd

from predictions.data.model_def import ModelDef

class PredPricesLoaderBase:
    
    def __init__(self):
        self.all_models_defs = {}

    def load(self, model_def: ModelDef) -> pd.DataFrame:
        raise Exception('Not implemented')
    
