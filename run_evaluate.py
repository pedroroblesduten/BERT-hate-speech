from evaluate import BERTClassificationEvaluation
from config import get_evaluate_config

if __name__ == "__main__":
    config = get_evaluate_config()

    BERTClassificationEvaluation(**config.__dict__).run(best=False)
