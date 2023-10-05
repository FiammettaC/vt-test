# model.py
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer

class Model:
  """A model class to lead the model and tokenizer"""

    def __init__(self) -> None:
        pass
  
    def load_model():
        #model = AutoModelForSequenceClassification.from_pretrained("./models/roberta-base/")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model

    def load_tokenizer():
        #tokenizer = AutoTokenize.from_pretrained("./models/roberta-base/")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer