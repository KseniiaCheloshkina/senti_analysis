import pandas as pd
import os
import traceback
from werkzeug.utils import secure_filename
import json
from flask import Flask, request, flash, redirect, jsonify, render_template

from evaluate import Model


app = Flask(__name__)
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 0.5 Mb
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024

CONFIG = {
    "model_name": "dostoevsky",
    "batch_size": 1000
}

global model
model = Model(model_name=CONFIG["model_name"], batch_size=CONFIG["batch_size"])


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        texts = pd.DataFrame([{'text': message}])
        my_prediction = model.predict(texts)
    return render_template('result.html', prediction=my_prediction[0])


@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    try:
        json_ = request.json
        my_prediction = model.predict(pd.DataFrame(json_))
        return jsonify({'prediction': str(my_prediction)})
    except:
        return jsonify({'trace': traceback.format_exc()})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        flash('No files')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_flnm = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_flnm)
    with open(save_flnm, 'r') as f:
        data = json.load(f)
    my_prediction = model.predict(pd.DataFrame(data))
    return jsonify({'prediction': str(my_prediction)})


if __name__ == '__main__':
    app.run(debug=True)
