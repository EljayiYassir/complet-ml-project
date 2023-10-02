import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV # for hyperparameters
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        '''
        This function train models and give the score of the best model performance

        '''

        try:
            logging.info("Split the train and test array")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:,-1],
                test_array[:, :-1], 
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Gradient boosting Regression": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            models_report:dict=evaluate_models(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test, 
                                               models=models)
            best_model_score=max( models_report.values())
            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model are found on both training and testing dataset: {best_model_name} {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("Save best model completed")

            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            return best_model_name, score

        except Exception as e:
            raise CustomException(e,sys)