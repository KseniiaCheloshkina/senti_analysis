import os
import gc
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from deeppavlov import build_model


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


def model_download():
    # download dostoevsky
    os.system("python -m dostoevsky download fasttext-social-network-model")
    # download deeppavlov
    for config in DP_CONFIG_LIST:
        print("config: ", config)
        os.system("python -m deeppavlov install {}".format(config))
        model = build_model(config, download=True)  # download=True - in case of necessity to download some data


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
    # TODO: add BERT model from transformers
    # model_download()
    print("CHECK DOSTOEVSKIY")
    check_dostoevsky()
    print("CHECK DEEPPAVLOV")
    for config in DP_CONFIG_LIST:
        print("config: ", config)
        check_deeppavlov(config)
