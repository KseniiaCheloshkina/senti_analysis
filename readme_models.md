## Models:
- Package dostoevsky
  - Source: https://github.com/bureaucratic-labs/dostoevsky
  - Description: for Russian language. Non-trainable, no other versions
  - Data: VKontakte comments (https://github.com/text-machine-lab/rusentiment).
  - Target: negative, positive, neutral, speech, skip
- DeepPavlov:
  - Source: http://docs.deeppavlov.ai/en/master/features/models/classifiers.html#id15
  - Description: could be trained. Twitter sentiment in Russian dataset (Twitter mokoron dataset and RuSentiment). Two versions of models: with and without preprocessing, model trained on preprocessed data is based on semantics while model trained on unprocessed data is based on punctuation and syntax.
  - Target: negative, positive