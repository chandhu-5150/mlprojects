import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Splitting training and test input data")
      X_train, y_train, X_test, y_test = (
        train_array[:, :-1],
        train_array[:, -1],
        test_array[:, :-1],
        test_array[:, -1]
      )

      models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor()
      }
      params = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },

    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "Linear Regression": {
        "fit_intercept": [True, False]
    },

    "K-Neighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },

    "XGBRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    },

    "CatBoosting Regressor": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7]
    },

    "AdaBoost Regressor": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "loss": ["linear", "square", "exponential"]
    }
}

      model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,
                                           models=models,param=params)

      for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict Testing data
        y_test_pred = model.predict(X_test)

        # Get R2 score for the model
        test_model_score = r2_score(y_test, y_test_pred)

        model_report[model_name] = test_model_score

      # Get best model score from the dictionary
      best_model_score = max(sorted(model_report.values()))

      # Get best model name from the dictionary
      best_model_name = list(model_report.keys())[
          list(model_report.values()).index(best_model_score)
      ]

      best_model = models[best_model_name]
      
      if best_model_score < 0.6:
        raise CustomException("No best model found")

      logging.info(f"Best found model on both training and testing dataset: {best_model_name}")
      

      save_object(
          file_path=self.model_trainer_config.trained_model_file_path,
          obj=best_model
      )
      
      predicted=best_model.predict(X_test)
      r2_square = r2_score(y_test, predicted)
      print(f"\nBest Model: {best_model_name}")

      print(f"Best Model R2 Score: {r2_square}")

      return r2_square

    except Exception as e:
      raise CustomException(e, sys)
  