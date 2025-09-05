import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException

# ML tracking
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_path, model_path):
        self.processed_path = processed_path
        self.model_path = model_path
        
        os.makedirs(self.model_path, exist_ok=True)

        logger.info("Model training Initialized...")

    def load_data(self):
        try:
            X_train = joblib.load(os.path.join(self.processed_path, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.processed_path, "X_test.pkl"))
            y_train = joblib.load(os.path.join(self.processed_path, "y_train.pkl"))
            y_test = joblib.load(os.path.join(self.processed_path, "y_test.pkl"))

            logger.info("Data Loaded for model training.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error While loading data {e}")
            raise CustomException("Failed to loading data.", sys)
        
    def train_model(self, X_train, y_train):
        try:
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(X_train, y_train)

            joblib.dump(self.model, os.path.join(self.model_path, "model.pkl"))

            logger.info(f"Model Trained and saved successfully")

        except Exception as e:
            logger.error(f"Error While Training model. {e}")
            raise CustomException("Failed to Train model.", sys)
        
    def evaluate_model(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1] if len(y_test.unique()) == 2 else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label="Yes")
            recall = recall_score(y_test, y_pred, pos_label="Yes")
            f1 = f1_score(y_test, y_pred, pos_label="Yes")

            mlflow.log_metric("Accuracy Score", accuracy)
            mlflow.log_metric("Precision Score", precision)
            mlflow.log_metric("Recall Score", recall)
            mlflow.log_metric("F1 Score", f1)

            logger.info(f"Accuracy : {accuracy}\nPrecision : {precision}\nRecall : {recall}\nF1 : {f1}")

            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("ROC-AUC", roc_auc)
            logger.info(f"Roc-Auc Score : {roc_auc}")

            logger.info("Model Evaluated on Test data.")

        except Exception as e:
            logger.error(f"Error While Evaluating model. {e}")
            raise CustomException("Failed to Evaluate model.", sys)
        
    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    with mlflow.start_run():
        processed_path = "artifacts/processed"
        model_path = "artifacts/model"
        trainer = ModelTraining(processed_path, model_path)
        trainer.run()