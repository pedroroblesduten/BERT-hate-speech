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
    load_data_config: LoadDataConfig = field(
        default_factory=lambda: LoadDataConfig()
    )
    save_model_path: str = "./model_checkpoints/"
    save_losses_path: str = "./artifacts/losses"
    save_weights_interval: int = 10
    device: str = "cuda"
    epochs: int = 60
    huggingface_model_name: str = 'pysentimiento/bertabaporu-pt-hate-speech'
    model_name: str = 'bert_hate_speech_discord'
    num_labels: int = 5


# ---- VALIDATION CONFIG ----
@dataclass
class EvaluateConfig:
    load_data_config: LoadDataConfig = field(
        default_factory=lambda: LoadDataConfig()
    )
    save_model_path: str = "./model_checkpoints/"
    save_results_path: str = "./artifacts/results"
    device: str = "cuda"
    huggingface_model_name: str = 'pysentimiento/bertabaporu-pt-hate-speech'
    model_name: str = 'bert_hate_speech_discord'
    num_labels: int = 5
    classes: list = field(
        default_factory=lambda: [
            'sexism',
            'body',
            'racism',
            'homophobia',
            'neutral'
            ]
    )

def get_trainer_config():
    trainer_config = TrainerConfig(load_data_config=LoadDataConfig())
    return trainer_config

def get_evaluate_config():
    config = EvaluateConfig(load_data_config=LoadDataConfig())
    return config
