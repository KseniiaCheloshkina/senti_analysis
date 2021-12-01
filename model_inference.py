import os
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel


CASES = [
    'привет',
    'я люблю тебя!!',
    'малолетние дебилы',
    'всё очень плохо'
]


def first_use():
    # download model
    os.system("python -m dostoevsky download fasttext-social-network-model")


def check_dostoevsky():

    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(CASES, k=2)

    for message, sentiment in zip(CASES, results):
        print(message, '->', sentiment)


if __name__ == "__main__":
    check_dostoevsky()
