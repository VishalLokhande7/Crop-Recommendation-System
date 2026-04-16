from flask import Flask, render_template, request
import pickle
import numpy as np
import os


MODEL_PATH = os.path.join('models', 'best_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as model_file:
        crop_recommendation_model = pickle.load(model_file)
except Exception as err:
    print(f"Error loading model: {err}")
    crop_recommendation_model = None


app = Flask(__name__)

picFolder = os.path.join('static','images')
app.config['UPLOAD_FOLDER'] = picFolder

@ app.route('/')
def home():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)

# here the form is present.
@ app.route('/success',methods=['POST','GET'])
def crop_recommend():
    title = 'Crop Recommendation'
    if request.method == 'POST':
        if crop_recommendation_model is None:
            return render_template('crop.html', title=title, error='Model not available. Please train the model first.')

        required_fields = ['nitrogen', 'phosphorous', 'potasium', 'ph', 'temprature', 'moisture', 'rainfall']
        for field in required_fields:
            if not request.form.get(field):
                return render_template('crop.html', title=title, error='Please fill in all fields.')

        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['potasium'])
            ph = float(request.form['ph'])
            T = float(request.form['temprature'])
            H = float(request.form['moisture'])
            rainfall = float(request.form['rainfall'])
        except ValueError:
            return render_template('crop.html', title=title, error='Please enter valid numeric values.')

        T = ((1007 - T) / (1007 - 107)) * 100
        data = np.array([[N, P, K, T, H, ph, rainfall]])

        try:
            prediction = crop_recommendation_model.predict(data)[0]
        except Exception as err:
            return render_template('crop.html', title=title, error=f'Prediction error: {err}')

        return render_template('crop-result.html', prediction=prediction)

    return render_template('crop.html', title=title)



if __name__ == '__main__':
    app.run(debug=True)
