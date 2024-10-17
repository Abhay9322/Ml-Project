from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Area = float(request.form.get('Area'))
    prediction = model.predict(np.array([[Area]]))[0]
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
