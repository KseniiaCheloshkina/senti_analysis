import json
import pandas as pd
import xml.etree.ElementTree as ET


def read_rureviews():
    print("RUREVIEWS")
    data = pd.read_csv("data/women-clothing-accessories.3-class.csv", sep="\t")
    print(len(data))
    print(data["sentiment"].value_counts(normalize=True, dropna=False))
    print(data.sample(5)['review'].values)


def read_one_sentirueval_twitter(filename, entity_fields):
    tree = ET.parse(filename)
    root = tree.getroot()

    n_mismatches = 0
    all_vals = []
    for elem in root:
        if elem.tag == "database":
            for row in elem:
                targets = []
                for cell in row:
                    if cell.attrib['name'] == 'text':
                        text = cell.text
                    if cell.attrib['name'] in entity_fields:
                        if cell.text != "NULL":
                            targets.append(cell.text)
                if len(set(targets)) > 1:
                    n_mismatches = n_mismatches + 1
                    continue
                all_vals.append({
                    'text': text,
                    'sentiment': targets[0]
                })
    print("confused cases: ", n_mismatches)
    return pd.DataFrame(all_vals)


def read_sentirueval_twitter():
    """
    Может быть 2 метки сентимента (относительно разных банков)
    Если метки отличаются (для одного текста разные метки для разных банков), такие кейсы исключаем
    :return:
    """
    print("SENTIRUEVAL bank train")
    banks_names = ['rshb', 'uralsib', 'raiffeisen', 'bankmoskvy', 'alfabank', 'gazprom', 'vtb', 'sberbank']
    train_filename = "data/SentiRuEval_sent_bank_train_2016.xml"
    df = read_one_sentirueval_twitter(train_filename, banks_names)
    print(df.shape)
    print(df['sentiment'].value_counts(dropna=False))
    print(df['sentiment'].value_counts(dropna=False, normalize=True))
    print("SENTIRUEVAL bank test etalon")
    test_filename = "data/SentiRuEval_sent_banks_test_etalon.xml"
    df = read_one_sentirueval_twitter(test_filename, banks_names)
    print(df.shape)
    print(df['sentiment'].value_counts(dropna=False))
    print(df['sentiment'].value_counts(dropna=False, normalize=True))

    print("SENTIRUEVAL telecom train")
    telecom_names = ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink']
    train_filename = "data/SentiRuEval_sent_tkk_train_2016.xml"
    df = read_one_sentirueval_twitter(train_filename, telecom_names)
    print(df.shape)
    print(df['sentiment'].value_counts(dropna=False))
    print(df['sentiment'].value_counts(dropna=False, normalize=True))
    print("SENTIRUEVAL telecom test etalon")
    test_filename = "data/SentiRuEval_sent_tkk_test_etalon.xml"
    df = read_one_sentirueval_twitter(test_filename, telecom_names)
    print(df.shape)
    print(df['sentiment'].value_counts(dropna=False))
    print(df['sentiment'].value_counts(dropna=False, normalize=True))


def read_sentirueval():
    print("SENTIRUEVAL")
    tree = ET.parse("data/SentiRuEval_car_markup_train.xml")
    root = tree.getroot()

    for elem in root:
        for subelem in elem:
            print(subelem)


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
    read_sentirueval_twitter()
    # TODO: add data from sentirueval - aspect based
    # TODO: add data from RuSentiment
