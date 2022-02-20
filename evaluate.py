import tqdm
import gc
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from deeppavlov import build_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from dataset import Dataset


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
    # TODO: check if 0,1,-1 is used in other datasets
    # in sentirueval mapping is {"0": "neutral", "1": "positive", "-1": "negative"}
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
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if model_name == "dostoevsky":
            tokenizer = RegexTokenizer()
            self.model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        if model_name in self.DP_CONFIG_LIST:
            self.model = build_model(model_name, download=False)
        if model_name == "xlm_roberta_large":
            checkpoint = "sismetanin/xlm_roberta_large-ru-sentiment-rutweetcorp"
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            self.model = model.to(self.device)

    def predict_dostoevsky(self, text: List[str]):
        prediction = pd.DataFrame(self.model.predict(text))
        label_pred = np.argmax(prediction.values, axis=1)
        major_pred_class = list(map(lambda x: prediction.columns[x], label_pred))
        print(major_pred_class)
        return major_pred_class

    def predict_dp(self, text: List[str]):
        prediction = self.model(text)
        return prediction

    def predict_xlm_roberta_large(self, text: List[str]):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        output = self.model(**tokens)
        prediction = output.logits.softmax(dim=-1).tolist()
        prediction = np.array(prediction)
        label_pred = np.argmax(prediction, axis=1)
        target_names = ['positive', 'neutral', 'negative']
        major_pred_class = list(map(lambda x: target_names[x], label_pred))
        return major_pred_class

    def predict_on_batch(self, texts: List[str]):
        if self.model_name == "dostoevsky":
            prediction = self.predict_dostoevsky(texts)
        if self.model_name in self.DP_CONFIG_LIST:
            prediction = self.predict_dp(texts)
        if self.model_name == "xlm_roberta_large":
            prediction = self.predict_xlm_roberta_large(texts)
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
        df_metrics = get_metrics(final_target, final_prediction)
        df_metrics.update(dataset['target'].map(self.TARGET_MAPPING).value_counts().to_dict())
        return pd.DataFrame([df_metrics])


def get_metrics(target, prediction):
    return {
        'f_micro_2': f1_score(y_true=target, y_pred=prediction,
                              labels=['positive', 'negative'],
                              average='micro'),
        'f_macro_2': f1_score(y_true=target, y_pred=prediction,
                              labels=['positive', 'negative'],
                              average='macro'),
        'f_micro_3': f1_score(y_true=target, y_pred=prediction,
                              average='micro'),
        'f_macro_3': f1_score(y_true=target, y_pred=prediction,
                              average='macro'),
        'f_pos': f1_score(y_true=target, y_pred=prediction, labels=['positive'], average='micro'),
        'f_neg': f1_score(y_true=target, y_pred=prediction, labels=['negative'], average='micro'),
        'f_neutral': f1_score(y_true=target, y_pred=prediction, labels=['neutral'], average='micro')
    }


def evaluate_all_models():
    model_names = [
        # "dostoevsky",
        # "sentiment_twitter_preproc",
        # "rusentiment_convers_bert",
        "xlm_roberta_large"
    ]
    datasets = [
        # "rureviews",
        # "kaggle_news",
        # "sentirueval_banks",
        # "sentirueval_tkk",
        # "rusentiment",
        "rutweetcorp"
    ]
    batch_sizes = {
        "dostoevsky": 1000,
        "sentiment_twitter_preproc": 200,
        "rusentiment_convers_bert": 100,
        "xlm_roberta_large": 5
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
                df_metrics['dataset'] = "{}_{}".format(dataset_name, data_type)
                all_metrics.append(df_metrics)
    df_metrics_all = pd.concat(all_metrics)
    del model
    gc.collect()
    return df_metrics_all


if __name__ == "__main__":
    # TODO: remove nrows=1000
    evaluate_all_models()
