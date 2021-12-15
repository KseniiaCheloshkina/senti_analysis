import json
import tqdm
import gc
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from deeppavlov import build_model


from read_data import read_one_sentirueval_twitter


class Dataset(object):
    def __init__(self, dataset_name: str):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.dataset_name = dataset_name
        DATASET_PATHS = {
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
            }
        }
        data_path = DATASET_PATHS[self.dataset_name]
        if 'train' in data_path:
            self.train_data = self.load_data(dataset_name=self.dataset_name, path=data_path['train'])
        if 'all' in data_path:
            self.data = self.load_data(dataset_name=self.dataset_name, path=data_path['all'])
        if 'test' in data_path:
            self.test_data = self.load_data(dataset_name=self.dataset_name, path=data_path['test'])

    def load_data(self, dataset_name: str, path: str):
        # TODO: add rusentiment
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
    def load_kaggle_news(path):
        with open(path, "rb") as f:
            data = json.load(f)
        data = pd.DataFrame(data)
        data.rename(columns={'sentiment': 'target'}, inplace=True)
        data.drop(['id'], axis=1, inplace=True)
        return data


class Model(object):
    DP_CONFIG_LIST = [
        "rusentiment_cnn",
        "rusentiment_bigru_superconv",
        "rusentiment_elmo_twitter_cnn",
        "rusentiment_bert",
        "rusentiment_convers_bert",
        "sentiment_twitter",
        "sentiment_twitter_preproc"
    ]
    TARGET_MAPPING = {
        'negative': 'negative',
        'Negative': 'negative',
        'positive': 'positive',
        'Positive': 'positive',
        'skip': 'neutral',
        'speech': 'neutral',
        'neutral': 'neutral',
        'neautral': 'neutral',
        '0': 'positive',
        '1': 'neutral',
        '-1': 'negative'
    }

    def __init__(self, model_name: str, batch_size: int = 100):
        self.model_name = model_name
        self.batch_size = batch_size
        if model_name == "dostoevsky":
            tokenizer = RegexTokenizer()
            self.model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        if model_name in self.DP_CONFIG_LIST:
            self.model = build_model(model_name, download=False)

    def predict_dostoevsky(self, text: List[str]):
        prediction = pd.DataFrame(self.model.predict(text))
        label_pred = np.argmax(prediction.values, axis=1)
        major_pred_class = list(map(lambda x: prediction.columns[x], label_pred))
        return major_pred_class

    def predict_dp(self, text: List[str]):
        prediction = self.model(text)
        return prediction

    def predict_on_batch(self, texts: List[str]):
        if self.model_name == "dostoevsky":
            prediction = self.predict_dostoevsky(texts)
        if self.model_name in self.DP_CONFIG_LIST:
            prediction = self.predict_dp(texts)
        return prediction

    def predict(self, dataset: pd.DataFrame):
        n_texts = dataset.shape[0]
        texts = dataset['text'].values.tolist()
        predictions = []
        if n_texts > self.batch_size:
            for i in range(n_texts // self.batch_size):
                gc.collect()
                texts_cur = texts[i * self.batch_size:(self.batch_size * (i + 1))]
                pred = self.predict_on_batch(texts_cur)
                predictions.extend(pred)
            if self.batch_size * (n_texts // self.batch_size) < n_texts:
                gc.collect()
                predictions.extend(self.predict_on_batch(texts[self.batch_size * (n_texts // self.batch_size):]))
        else:
            predictions = self.predict_on_batch(texts)
        # apply mapping of target names from different models to single notation
        final_prediction = list(map(lambda x: self.TARGET_MAPPING[x], predictions))
        return final_prediction

    def evaluate(self, dataset: pd.DataFrame):
        if 'target' not in dataset.columns:
            return pd.DataFrame()
        target = dataset['target'].values.tolist()
        # apply mapping of target names from different models to single notation
        final_target = list(map(lambda x: self.TARGET_MAPPING[x], target))
        final_prediction = self.predict(dataset)
        df_metrics = {
            'f_micro_2': f1_score(y_true=final_target, y_pred=final_prediction,
                                labels=['positive', 'negative'],
                                average='micro'),
            'f_macro_2': f1_score(y_true=final_target, y_pred=final_prediction,
                                labels=['positive', 'negative'],
                                average='macro'),
            'f_micro_3': f1_score(y_true=final_target, y_pred=final_prediction,
                                  average='micro'),
            'f_macro_3': f1_score(y_true=final_target, y_pred=final_prediction,
                                  average='macro'),
            'f_pos': f1_score(y_true=final_target, y_pred=final_prediction, labels=['positive'], average='micro'),
            'f_neg': f1_score(y_true=final_target, y_pred=final_prediction, labels=['negative'], average='micro'),
            'f_neutral': f1_score(y_true=final_target, y_pred=final_prediction, labels=['neutral'], average='micro')
        }
        df_metrics.update(dataset['target'].map(self.TARGET_MAPPING).value_counts().to_dict())
        return pd.DataFrame([df_metrics])


def evaluate_all_models():
    model_names = [
        "dostoevsky",
        "sentiment_twitter_preproc",
        "rusentiment_convers_bert"
    ]
    datasets = [
        # "rureviews",
        # "kaggle_news",
        "sentirueval_banks",
        "sentirueval_tkk"
    ]
    batch_sizes = {
        "dostoevsky": 1000,
        "sentiment_twitter_preproc": 200,
        "rusentiment_convers_bert": 100
    }
    all_data = []
    for model_name in model_names:
        df_metrics_all = evaluate_model(model_name=model_name, datasets=datasets, batch_size=batch_sizes[model_name])
        df_metrics_all['model_name'] = model_name
        all_data.append(df_metrics_all)
    all_data = pd.concat(all_data)
    print(all_data)
    all_data.to_excel("data/evaluate_results.xlsx")


def evaluate_model(model_name: str, datasets: List[str], batch_size: int = 100):
    all_metrics = []
    model = Model(model_name=model_name, batch_size=batch_size)
    for dataset_name in tqdm.tqdm(datasets):
        dataset = Dataset(dataset_name=dataset_name)
        for data_type in ["data", "train_data", "test_data"]:
            current_df = dataset.__getattribute__(data_type)
            if current_df is not None:
                df_metrics = model.evaluate(current_df.sample(1000))
                df_metrics['dataset'] = dataset_name
                all_metrics.append(df_metrics)
    df_metrics_all = pd.concat(all_metrics)
    return df_metrics_all


if __name__ == "__main__":
    evaluate_all_models()
