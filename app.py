from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('insurance_model.pkl','rb'))

@app.route('/')
def home():
    return "Insurance Cost Prediction API is Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data = np.array(data).reshape(1,-1)
    prediction = model.predict(data)
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)