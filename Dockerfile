FROM python:3.6-slim-buster
WORKDIR /app
COPY requirements_predict.txt requirements.txt
RUN pip install -r requirements_predict.txt
COPY . .
ENV FLASK_APP=api-server
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]