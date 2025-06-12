from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
print("sys.path:", sys.path)


from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask app
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Collect form data
        data = CustomData(
            fixed_acidity=float(request.form.get('fixed_acidity')),
            volatile_acidity=float(request.form.get('volatile_acidity')),
            citric_acid=float(request.form.get('citric_acid')),
            residual_sugar=float(request.form.get('residual_sugar')),
            chlorides=float(request.form.get('chlorides')),
            free_sulfur_dioxide=float(request.form.get('free_sulfur_dioxide')),
            total_sulfur_dioxide=float(request.form.get('total_sulfur_dioxide')),
            density=float(request.form.get('density')),
            pH=float(request.form.get('pH')),
            sulphates=float(request.form.get('sulphates')),
            alcohol=float(request.form.get('alcohol')),
            color=request.form.get('color')
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        # Prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html', results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
