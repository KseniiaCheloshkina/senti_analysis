# В целом хорошо показывает разговорный БЕРТ (rusentiment_convers_bert, 20 tokens)
# он обучен на корпусе rusentiment (на нем и показывает лучшее качество из всех)
# Дообучить эту модель на датасетах, на которых эта модель показывает качество хуже (sentirueval_banks, sentirueval_tkk)
# 1. Обучаем на трейн этого датасета, оцениваем качество на тесте этого датасета
# 2. Обучаем на трейне нескольких датасетов, оцениваем качество на тесте этого датасета

# multilingual BERT fine-tuned on english dialog datasets and MLM on Russian
# English datasets (each utterance is labeled):
# labeled by emotions: EmoryNLP, MELD, EmotionLines, EmoContext, DailyDialog
# labeled additionally with sentiment: MELD

# Use t5 as in mrm8488/t5-base-finetuned-span-sentiment-extraction

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import Dataset
from evaluate import get_metrics


CASES = [
    'привет',
    'я люблю тебя!!',
    'малолетние дебилы',
    'красивая платье',
    'неудобная куртка',
    'всё очень плохо'
]


class CustomBERTModel(object):

    def predict(self, texts):
        pass

    def evaluate(self,
                 dataset_name="sentirueval_banks",
                 target_mapping={"0": "neutral", "1": "positive", "-1": "negative"}
                 ):
        all_metrics = []
        dataset = Dataset(dataset_name=dataset_name)
        for data_type in ["data", "train_data", "test_data"]:
            current_df = dataset.__getattribute__(data_type)
            if current_df is not None:
                data = current_df['text'].values.tolist()
                target = current_df['target'].values.tolist()
                target = list(map(lambda x: target_mapping[x], target))
                # TODO: add batch processing
                data = data[:200]
                target = target[:200]
                predictions = self.predict(data)
                df_metrics = get_metrics(target=target, prediction=predictions)
                df_metrics['dataset'] = "{}_{}".format(dataset_name, data_type)
                all_metrics.append(df_metrics)
        df_metrics_all = pd.DataFrame(all_metrics)
        return df_metrics_all


class BaseConversationalModel(CustomBERTModel):
    """
    Base conversational RuBERT model trained on RuSentiment dataset
    # 20 tokens sequence length
    # class labels: negative, neutral, positive, skip, speech
    """
    def __init__(self):
        self.model_checkpoint = "sismetanin/rubert_conversational-ru-sentiment-rusentiment"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)
        self.model = model.to(self.device)
        # TODO: add method predict_on_batch

    def predict(self, texts):
        tokens = self.tokenizer(texts, padding="max_length", max_length=self.model.config.max_length,
                           truncation=True, return_tensors="pt").to(self.device)
        output = self.model(**tokens)
        probas = output.logits.softmax(dim=-1).tolist()
        prediction = np.array(probas)
        label_pred = np.argmax(prediction, axis=1)
        target_names = ['negative', 'neutral', 'positive', 'skip', 'speech']
        major_pred_class = list(map(lambda x: target_names[x], label_pred))
        TARGET_MAPPING = {
            'negative': 'negative',
            'positive': 'positive',
            'neutral': 'neutral',
            'skip': 'neutral',
            'speech': 'neutral'
        }
        return list(map(lambda x: TARGET_MAPPING[x], major_pred_class))


if __name__ == "__main__":
    base_model = BaseConversationalModel()
    predicted_labels = base_model.predict(CASES)
    print(predicted_labels)
    df_results = base_model.evaluate()
    print(df_results)
    df_results.to_csv("data/test_base_bert.csv")
    # PLAN:
    # 1. add batch prediction to evaluate sentirueval_banks on BaseConversationalModel
    # 2. create new class with fine-tuned model
    # 3. add method fine_tune(model_checkpoint)
    # 4. fine-tune on train, evaluate on train and test
    # 5. print reesulting table - before and after fine-tuning
