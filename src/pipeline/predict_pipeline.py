import sys
import time
import pandas as pd
from src.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utilities import load_object

# Only load the full pipeline
MODEL_PATH = r'C:\Users\katch\Desktop\projects\wine_quality_prediction\artifacts\final_random_forest_classifier.pkl'

print("Loading full model pipeline...")
load_start = time.time()
model = load_object(MODEL_PATH)
print(f"Model pipeline loaded in {round(time.time() - load_start, 2)} seconds.")

class PredictPipeline:
    def predict(self, features: pd.DataFrame) -> float:
        try:
            print("Starting prediction process...")
            print("Input columns:", features.columns.tolist())
            print("Input preview:\n", features.head())

            start = time.time()

            # Predict using the full pipeline
            preds = model.predict(features)

            total = time.time() - start
            print(f"Prediction completed in {round(total, 3)} seconds")

            return preds
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)


class CustomData:
    def __init__(self, 
                 fixed_acidity: float,
                 volatile_acidity: float,
                 citric_acid: float,
                 residual_sugar: float,
                 chlorides: float,
                 free_sulfur_dioxide: float,
                 total_sulfur_dioxide: float,
                 density: float,
                 pH: float,
                 sulphates: float,
                 alcohol: float,
                 color: object):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.color = color

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("Converting custom data to DataFrame")
            custom_data_dict = {
                "fixed_acidity": [self.fixed_acidity],
                "volatile_acidity": [self.volatile_acidity],
                "citric_acid": [self.citric_acid],
                "residual_sugar": [self.residual_sugar],
                "chlorides": [self.chlorides],
                "free_sulfur_dioxide": [self.free_sulfur_dioxide],
                "total_sulfur_dioxide": [self.total_sulfur_dioxide],
                "density": [self.density],
                "pH": [self.pH],
                "sulphates": [self.sulphates],
                "alcohol": [self.alcohol],
                "color": [self.color]
            }
            df = pd.DataFrame(custom_data_dict)
            print("Constructed DataFrame:\n", df)
            return df
        except Exception as e:
            raise CustomException(f"Error converting data to DataFrame: {str(e)}", sys)
