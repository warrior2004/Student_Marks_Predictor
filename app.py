from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.Student_Marks_Predictor.logger import logging
from src.Student_Marks_Predictor.pipelines.prediction_pipeline import CustomData, PredictPipeline

# Initialize Flask app
app = Flask(__name__)


# Route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # Fixed incorrect field mapping
                writing_score=float(request.form.get('writing_score'))  # Fixed incorrect field mapping
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Received Data for Prediction: \n{pred_df}")

            # Predict using trained model
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            logging.info(f"Prediction Result: {results[0]}")
            return render_template('home.html', results=results[0])

        except Exception as e:
            logging.error(f"Error in Prediction: {str(e)}", exc_info=True)
            return render_template('home.html', error="Error in Prediction. Please check input values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
