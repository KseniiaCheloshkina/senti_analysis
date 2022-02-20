import json
import tqdm
import pandas as pd

from read_data import read_one_sentirueval_twitter


class Dataset(object):
    def __init__(self, dataset_name: str = None):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.dataset_name = dataset_name
        self.dataset_paths = {
            'rureviews': {
                'all': '"data/women-clothing-accessories.3-class.csv"'
            },
            "kaggle_news": {
                "train": '"data/train.json"',
                "test": '"data/test.json"'
            },
            "sentirueval_banks": {
                "train": '"data/SentiRuEval_sent_bank_train_2016.xml"',
                "test": '"data/SentiRuEval_sent_banks_test_etalon.xml"'
            },
            "sentirueval_tkk": {
                "train": '"data/SentiRuEval_sent_tkk_train_2016.xml"',
                "test": '"data/SentiRuEval_sent_tkk_test_etalon.xml"'
            },
            "rusentiment": {
                "train": '"data/rusentiment_preselected_posts.csv"',
                "test": '"data/rusentiment_test.csv"'
            },
            "rutweetcorp": {
                "all": '"data/rutweetcorp_full.csv"'
            }
        }
        if dataset_name:
            data_path = self.dataset_paths[self.dataset_name]
            if 'train' in data_path:
                self.train_data = self.load_data(dataset_name=self.dataset_name, path=data_path['train'])
            if 'all' in data_path:
                self.data = self.load_data(dataset_name=self.dataset_name, path=data_path['all'])
            if 'test' in data_path:
                self.test_data = self.load_data(dataset_name=self.dataset_name, path=data_path['test'])

    def load_data(self, dataset_name: str, path: str):
        func_name = "self.load_{}(path={})".format(dataset_name, path)
        data = eval(func_name)
        return data

    @staticmethod
    def load_sentirueval_banks(path):
        banks_names = ['rshb', 'uralsib', 'raiffeisen', 'bankmoskvy', 'alfabank', 'gazprom', 'vtb', 'sberbank']
        data = read_one_sentirueval_twitter(path, banks_names)
        data.rename(columns={'sentiment': 'target'}, inplace=True)
        return data

    @staticmethod
    def load_sentirueval_tkk(path):
        telecom_names = ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink']
        data = read_one_sentirueval_twitter(path, telecom_names)
        data.rename(columns={'sentiment': 'target'}, inplace=True)
        return data

    @staticmethod
    def load_rureviews(path):
        data = pd.read_csv(path, sep="\t")
        data.rename(columns={'review': 'text', 'sentiment': 'target'}, inplace=True)
        return data

    @staticmethod
    def load_rusentiment(path):
        data = pd.read_csv(path)
        data.rename(columns={'label': 'target'}, inplace=True)
        return data

    @staticmethod
    def load_rutweetcorp(path):
        data = pd.read_csv(path)
        return data

    @staticmethod
    def load_kaggle_news(path):
        with open(path, "rb") as f:
            data = json.load(f)
        data = pd.DataFrame(data)
        data.rename(columns={'sentiment': 'target'}, inplace=True)
        data.drop(['id'], axis=1, inplace=True)
        return data


def save_all_data_to_file():
    data = []
    datasets = Dataset()
    datasets = list(datasets.dataset_paths.keys())
    for dataset_name in tqdm.tqdm(datasets):
        dataset = Dataset(dataset_name=dataset_name)
        for data_type in ["data", "train_data", "test_data"]:
            current_df = dataset.__getattribute__(data_type)
            if current_df is not None:
                current_df['dataset_name'] = dataset_name
                current_df['dataset_type'] = data_type
                data.append(current_df)
    df = pd.concat(data, axis=0)
    print(df.sample(2).T)
    print(df.shape)
    df.to_csv("data/all_datasets.csv")


def save_to_csv():
    data = Dataset("sentirueval_banks")
    print(data)
    data_files = {
        "train": "data/sentirueval_banks_train.csv",
        "test": "data/sentirueval_banks_test.csv"
    }
    data.train_data.to_csv(data_files["train"])
    data.test_data.to_csv(data_files["test"])


if __name__ == "__main__":
    save_all_data_to_file()
    save_to_csv()
