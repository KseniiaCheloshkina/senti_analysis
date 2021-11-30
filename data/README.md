# Данные:

- SentiRuEval2015:
  - Source: https://drive.google.com/drive/folders/0B7y8Oyhu03y_fjNIeEo3UFZObTVDQXBrSkNxOVlPaVAxNTJPR1Rpd2U1WEktUVNkcjd3Wms
  - Files:
    - car_markup_train.xml, car_markup_test.xml
    - rest_markup_train.xml, rest_markup_test.xml
  - Description: Reviews on cars and restaurants, sentiment is provided by category, not in whole. <category name="Whole" sentiment="positive"/> - aggregation by all categories, may be "both"
  - Target: absence, positive, negative. 
- RuReviews: 
  - Source: https://raw.githubusercontent.com/sismetanin/rureviews/
  - Files: women-clothing.csv (90 000, 33% for each class - negative, neautral, positive)
  - Description: top-ranked goods form the major e-commerce site in Russian. 5-point scale. "Women’s Clothes and Accessories" category. 
  - Target: 5-point scale to 3-point scale by combining reviews with “1” and “2” into one “negative” class and reviews with “3” and “5” scores into another one “positive” class. 
- Kaggle Russian News Dataset:
  - Source: https://www.kaggle.com/c/sentiment-analysis-in-russian/data?select=train.json
  - Files: 
    - train.json (8 263, 33% positive / 17% negative)
    - test.json (2 056)
  - Description: russian news. Много про Казахстан
  - Target: positive, negative or neutral

Было бы здорово достать корпус Mokoron Russian Twitter Corpus (http://study.mokoron.com/)

Статьи:
- https://deepai.org/publication/improving-results-on-russian-sentiment-datasets
- https://github.com/sismetanin/sentiment-analysis-in-russian
- https://habr.com/ru/post/472988/