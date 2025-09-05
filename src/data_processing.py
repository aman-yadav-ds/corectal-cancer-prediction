import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2, SelectKBest

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = []

        os.makedirs(output_path, exist_ok=True)

        logger.info("Data Processor Initialized...")

    def load_data(self):
        try:
            df = pd.read_csv(self.input_path)
            logger.info("Data Loader Successfully.")
            return df
        except Exception as e:
            logger.error(f"Error While loading data {e}")
            raise CustomException("Failed to load data.", sys)

    def preprocess_data(self, df):
        try:
            df = df.drop(columns=["Patient_ID"])
            X = df.drop(columns=["Survival_Prediction"])
            y = df["Survival_Prediction"]


            ## Label Encoding
            cat_cols = X.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            logger.info("Basic Processing Done.")
            return X, y
        except Exception as e:
            logger.error(f"Error While preprocessing data {e}")
            raise CustomException("Failed to preprocessing data.", sys)
        
    def feature_selection(self, X, y):
        try:
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            # CHI SQUARE TEST
            X_cat = X_train.select_dtypes(include=["int64", "float64"])
            chi2_selector = SelectKBest(score_func=chi2, k="all")
            chi2_selector.fit(X_cat, y_train)

            chi2_scores = pd.DataFrame({
                'feature': X_cat.columns,
                'Chi2 Score': chi2_selector.scores_
            }).sort_values(by="Chi2 Score", ascending=False)

            self.selected_features = chi2_scores.head(5)['feature'].to_list()

            logger.info(f"Selected Features are : {self.selected_features}")

            X = X[self.selected_features]
            logger.info("Feature Selection Done...")

            return X, y

        except Exception as e:
            logger.error(f"Error While feature selection {e}")
            raise CustomException("Failed to select best features.", sys)
        
    def split_and_scale_data(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            logger.info("Splitting and Scaling Done...")

            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error While splitting and scaling data {e}")
            raise CustomException("Failed to split and scale data.", sys)
    
    def save_data_and_scaler(self, X_train, X_test, y_train, y_test):
        try:
            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            joblib.dump(self.scaler, os.path.join(self.output_path, "scaler.pkl"))

            logger.info(f"All artifacts saved.")
        
        except Exception as e:
            logger.error(f"Error While saving artifacts {e}")
            raise CustomException("Failed to save artifacts.", sys)

    def run(self):
        df = self.load_data()
        X, y = self.preprocess_data(df)
        X, y = self.feature_selection(X, y)
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)

        self.save_data_and_scaler(X_train, X_test, y_train, y_test)

        logger.info("Preprocessor Ran Successfully.")

if __name__ == "__main__":
    input_path = "artifacts/raw/data.csv"
    output_path = "artifacts/processed"

    processor = DataProcessing(input_path, output_path)

    processor.run()