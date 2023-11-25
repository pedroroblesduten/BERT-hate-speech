from dataclasses import dataclass, field
from typing import Any, Union
import torch

@dataclass
class LoadDataConfig:
    csv_path: str = './data/classificacao.csv'
    batch_size: int = 2
    label_columns: list = field(
        default_factory=lambda: [
            'sexism',
            'body',
            'racism',
            'homophobia',
            'neutral'
            ]
    )

# ---- TRAINER CONFIG ----
@dataclass
class TrainerConfig:
    load_data_config: LoadDataConfig
    save_model_path: str = "./model_checkpoints/"
    save_losses_path: str = "./artifacts/losses"
    save_weights_interval: int = 10
    device: str = "cuda"
    epochs: int = 400
    huggingface_model_name = 'pysentimiento/bertabaporu-pt-hate-speech'
    model_name = 'bert_hate_speech_discord'

def get_trainer_config():
    trainer_config = TrainerConfig(load_data_config=LoadDataConfig())
    return trainer_config
