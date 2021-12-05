# Sentiment analysis for Russian Language

Данный репозиторий содержит модели сентиментного анализа для русского языка c API для тестирования функционала.

Модель: возвращает сентимент (positive, negative, neutral).

Обращение по API:
- input: txt-файл, каждая строка представляет собой отдельный текст для анализа
- output: txt-файл с ответом, где каждая строка содержит предсказанную метку сентимента для соответствующего текста в исходном файле
Состав API: Flask app

План исследования:

1. Сбор корпуса для оценки качества моделей и дообучения
2. Применяем базовые (готовые) решения на собранном корпусе: dostoevskiy, deeppavlov, transformers bert
3. Выбираем лучшую модель и делаем API
4. Делаем fine-tuning transformers-модели на своих датасетах и проверяем, повысилось ли качество. Тестируем влияние разных гиперпараметров.

## API
### 1. Single text
WEB: http://127.0.0.1:5000/
### 2. Inline JSON
```bash
curl --location --request POST 'localhost:5000/predict_multiple' --header 'Content-Type: application/json' --data-raw '[{"text": "Привет"}, {"text": "Все очень плохо"}]'
```

### 3. File
```bash
curl -X POST -F file=@"data/test_input.json" http://localhost:5000/predict_file
```