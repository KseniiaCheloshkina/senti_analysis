# В целом хорошо показывает разговорный БЕРТ (rusentiment_convers_bert)
# он обучен на корпусе rusentiment (на нем и показывает лучшее качество из всех)
# Попробуем дообучить эту модель на датасетах, на которых эта модель показывает качество хуже
# 1. Обучаем на трейн этого датасета, оцениваем качество на тесте этого датасета
# 2. Обучаем на трейне нескольких датасетов, оцениваем качество на тесте этого датасета
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CASES = [
    'привет',
    'я люблю тебя!!',
    'малолетние дебилы',
    'красивая платье',
    'неудобная куртка',
    'всё очень плохо'
]


def initialize_model():
    model_checkpoint = "sismetanin/rubert_conversational-ru-sentiment-rusentiment"
    # model_checkpoint = "sismetanin/rubert-ru-sentiment-rusentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    return tokenizer, model


def check_model(cases):
    tokenizer, model = initialize_model()
    print(model.config.max_length)
    print(model.config.id2label)


def check_tokenizer(cases):
    tokenizer, model = initialize_model()

    # SINGLE CASE
    example = cases[-1]
    # only split string on tokens - return string
    tokens = tokenizer.tokenize(example)
    print(tokens)
    # turn tokens to input_ids - no special tokens added
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    # convert inut_ids to initial string
    decoded_string = tokenizer.decode(ids)
    print(decoded_string)
    # GENERAL FUNCTION - incorporate several utilities and consequent operations
    # return dict of keys ('input_ids', 'token_type_ids', 'attention_mask')
    # input_ids with added CLS and SEP
    print(tokenizer(example))
    # additionally could be padded, truncated
    print(tokenizer(example, padding=True, truncation=True, max_length=2))
    print(model.config.max_length)
    print(tokenizer(example, padding="max_length", truncation=True, max_length=model.config.max_length))
    # to be processed by model inputs should be collected in batches of the same length as torch tensors
    print(tokenizer(example, padding="max_length", truncation=True, max_length=model.config.max_length,
                    # return_tensors='pt'
                    ))
    # VOCABULARY
    # which special tokens and their ids?
    print(tokenizer.all_special_ids)
    print(tokenizer.all_special_tokens)
    # dict token: input_id
    vocab = tokenizer.get_vocab()
    # other tokens could be added to vocabulary
    unknown_token = "эквайринг"
    print(tokenizer(unknown_token)['input_ids'])
    print(tokenizer.add_tokens([unknown_token]))
    print(tokenizer(unknown_token)['input_ids'])
    # other special tokens could be added to vocabulary
    text = "я встретила в тексте [NEW_TOKEN]"
    print(tokenizer(text)['input_ids'])
    print(len(tokenizer))
    tokenizer.add_tokens(['[NEW_TOKEN]'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))  # embedding matrix should be resized
    print(len(tokenizer))
    print(tokenizer(text)['input_ids'])


if __name__ == "__main__":
    check_model(CASES)
    check_tokenizer(CASES)
