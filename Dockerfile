FROM python:3.6-slim-buster
WORKDIR /app
COPY requirements_predict.txt requirements.txt
RUN pip install -r requirements_train.txt
RUN python -m dostoevsky download fasttext-social-network-model
COPY . .
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]