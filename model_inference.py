import os
import time
import gc
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from deeppavlov import build_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


CASES = [
    'привет',
    'я люблю тебя!!',
    'малолетние дебилы',
    'всё очень плохо'
]

DP_CONFIG_LIST = [
    "rusentiment_cnn",
    "rusentiment_bigru_superconv",
    "rusentiment_elmo_twitter_cnn",
    "rusentiment_bert",
    "rusentiment_convers_bert",
    "sentiment_twitter",
    "sentiment_twitter_preproc"
]


def check_xlm_large():
    # model loading takes 1 minute, prediction on 1 sample - 2 sec
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = "sismetanin/xlm_roberta_large-ru-sentiment-rutweetcorp"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = [
        "Стас Михайлов отсудил у телевидения 150 тыс. рублей #культура http://t.co/UnVUQFiZqj",
        "@marinaysol а, а то подумала, что у тебя там пробежечка.)"
    ]
    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt").to(device)
    model = model.to(device)
    output = model(**tokens)
    probas = output.logits.softmax(dim=-1).tolist()
    print(probas)


def model_download():
    # download dostoevsky
    os.system("python -m dostoevsky download fasttext-social-network-model")
    # download deeppavlov
    for config in DP_CONFIG_LIST:
        print("config: ", config)
        os.system("python -m deeppavlov install {}".format(config))
        model = build_model(config, download=True)  # download=True - in case of necessity to download some data
    # xlm roberta large
    checkpoint = "sismetanin/xlm_roberta_large-ru-sentiment-rutweetcorp"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def check_dostoevsky():
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(CASES, k=2)

    for message, sentiment in zip(CASES, results):
        print(message, '->', sentiment)


def check_deeppavlov(config_name):
    model = build_model(config_name, download=False)
    print(model(CASES))
    del model
    gc.collect()


if __name__ == "__main__":
    # download models if using first time
    # model_download()
    print("CHECK DOSTOEVSKIY")
    check_dostoevsky()
    print("CHECK DEEPPAVLOV")
    for config in DP_CONFIG_LIST:
        print("config: ", config)
        check_deeppavlov(config)
    print("CHECK XLM ROBERTA LARGE")
    check_xlm_large()
