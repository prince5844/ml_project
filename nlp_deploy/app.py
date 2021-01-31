import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = load_model('model.h5')
tokenizer=pickle.load(open('tokenizer','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_msg = [str(x) for x in request.form.values()]
    #print(predict_msg)
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =50)
    res=model.predict(padded)
    output=''
    if res>0.5:
      output = 'Hey!! Be Aware It is a spam message'
    else:
      output=  'Bingo!!! It is not a spam message'

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)