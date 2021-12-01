import json
import pandas as pd
import xml.etree.ElementTree as ET


def read_rureviews():
    print("RUREVIEWS")
    data = pd.read_csv("data/women-clothing-accessories.3-class.test.csv", sep="\t")
    print(len(data))
    print(data["sentiment"].value_counts(normalize=True, dropna=False))
    print(data.sample(5)['review'].values)


def read_sentirueval():
    print("SENTIRUEVAL")
    tree = ET.parse("data/SentiRuEval_car_markup_train.xml")
    root = tree.getroot()

    for elem in root:
        for subelem in elem:
            print(subelem.text)


def read_kaggle_news():
    print("NEWS")
    with open("data/train.json", "rb") as f:
        train_data = json.load(f)
    df = pd.DataFrame(train_data)
    print(len(train_data))
    print(df["sentiment"].value_counts(normalize=True))
    print(df.sample(1)['text'].values[0])

    with open("data/test.json", "rb") as f:
        test_data = json.load(f)
    df_test = pd.DataFrame(test_data)
    print(len(test_data))
    print(df_test.sample(1)['text'].values[0])


if __name__ == "__main__":
    read_kaggle_news()
    read_rureviews()
    read_sentirueval()
    # TODO: add data from sentirueval - twitter
