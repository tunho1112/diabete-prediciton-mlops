import argparse
import logging
import os
import pickle

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# import logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



class ModelTrainer:
    def __init__(
        self, 
        mlflow_uri: str = "http://0.0.0.0:5000", 
        experiment_name: str = "diabetes_model"
    ) -> None:
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name

    @staticmethod
    def get_model(model_name):
        dict_model = {
            "xgb": xgb.XGBClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(),
            "ada_boost": AdaBoostClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(),
            "logistic_regression": LogisticRegression(),
        }
        return dict_model[model_name]

    def train_model(self, model_name):
        # init mlflow experiment
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name + "_" + model_name)

        # load data
        df_train = pd.read_csv("data/train.csv")
        df_test = pd.read_csv("data/test.csv")

        X_train = df_train.drop(columns=["Outcome"])
        y_train = df_train["Outcome"]
        X_test = df_test.drop(columns=["Outcome"])
        y_test = df_test["Outcome"]

        # scaler
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # save scaler
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        # create model
        model = ModelTrainer.get_model(model_name)
        # training model
        model.fit(X_train, y_train)
        logging.info(f"model {model_name} is trained")
        y_pred = model.predict(X_test)

        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }
        logging.info(f"model {model_name} metrics: {metrics}")

        # mlflow logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature,
            registered_model_name=model_name  # Register to Model Registry
        )
        mlflow.end_run()
        logging.info("finish training")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name", default="xgb")
    args = parser.parse_args()
    ModelTrainer().train_model(args.model_name)