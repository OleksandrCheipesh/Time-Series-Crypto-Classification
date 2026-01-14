from dataclasses import dataclass


@dataclass
class ModelDef:
    name: str
    ctx: int
    
    def get_model_id(self):
        return f'{self.name}__{self.ctx}'
