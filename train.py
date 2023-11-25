from trainer import Trainer
from config import get_trainer_config

if __name__ == "__main__":
    config = get_trainer_config()

    Trainer(**config.__dict__).train()
