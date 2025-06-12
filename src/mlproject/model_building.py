from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from src.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.model_evaluation import DefaultModelVisualizer
from src.mlproject.feature_engineering import FeatureEngineering, LabelEncoding
from src.mlproject.utilities import save_object


# Base class
class ModelBuilding(ABC):
    @abstractmethod
    def build_model(self, x_train: pd.DataFrame, y_train: pd.Series,
                    x_test: pd.DataFrame, y_test: pd.Series,
                    return_metrics: bool = False, visualize: bool = False,
                    search_type: str = "grid", n_iter: int = 10) -> Any:
        pass


class LogisticRegressionModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            model = LogisticRegression()
            params = {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs'],
                'class_weight': ['balanced'],
                'max_iter': [200, 500]
            }
            search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(model, params, cv=3, n_jobs=-1)

            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for Logistic Regression: {best_params}")

            if return_metrics:
                train_score = best_model.score(x_train, y_train)
                test_score = best_model.score(x_test, y_test)
                logging.info(f"Logistic Regression - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"Logistic Regression - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_model, train_score, test_score)
            else:
                result = best_model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=best_model.classes_)
                visualizer.visualize(best_model, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training Logistic Regression model: {e}")
            raise CustomException(e, sys)


class RandomForestClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            model = RandomForestClassifier()
            params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            }
            search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(model, params, cv=3, n_jobs=-1)

            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for random_forest: {best_params}")

            if return_metrics:
                train_score = best_model.score(x_train, y_train)
                test_score = best_model.score(x_test, y_test)
                logging.info(f"Random Forest - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"Random Forest - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_model, train_score, test_score)
            else:
                result = best_model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=best_model.classes_)
                visualizer.visualize(best_model, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            raise CustomException(e, sys)



class Final_RandomForestClassifierModel:
    def build_model(self, x_train, y_train, x_test, y_test,
                    return_metrics=False, visualize=False,
                    search_type="grid", n_iter=10):
        try:
            # Define expected columns
            expected_categorical = ['color']
            expected_numeric = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                'density', 'pH', 'sulphates', 'alcohol'
            ]

            # Ensure features exist in x_train
            categorical_features = [col for col in expected_categorical if col in x_train.columns]
            numeric_features = [col for col in expected_numeric if col in x_train.columns]

            # Debug
            print("Available columns in x_train:", list(x_train.columns))
            print("Using numeric features:", numeric_features)
            print("Using categorical features:", categorical_features)

            # Preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )

            # Random Forest classifier
            rf_model = RandomForestClassifier()

            # Full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', rf_model)
            ])

            # Parameters for tuning
            params = {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [20],
                'classifier__min_samples_split': [2],
                'classifier__criterion': ['entropy']
            }

            # GridSearch or RandomizedSearch
            search = RandomizedSearchCV(pipeline, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(pipeline, params, cv=3, n_jobs=-1)

            # Fit model
            search.fit(x_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for Final RF: {best_params}")

            # Save model
            save_object(best_pipeline, 'artifacts/final_random_forest_classifier.pkl')

            # Evaluate
            if return_metrics:
                train_score = best_pipeline.score(x_train, y_train)
                test_score = best_pipeline.score(x_test, y_test)
                logging.info(f"Final RF - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"Final RF - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_pipeline, train_score, test_score)
            else:
                result = best_pipeline

            # Visualize
            if visualize:
                labels = y_train.unique() if hasattr(y_train, "unique") else best_pipeline.named_steps["classifier"].classes_
                visualizer = DefaultModelVisualizer(labels=labels)
                visualizer.visualize(best_pipeline, x_test, y_test)

            return result

        except Exception as e:
            logging.error(f"Error training Final Random Forest model: {e}")
            raise CustomException(e, sys)


 



class SupportVectorClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            model = SVC(probability=True)
            params = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(model, params, cv=3, n_jobs=-1)

            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for svc: {best_params}")

            if return_metrics:
                train_score = best_model.score(x_train, y_train)
                test_score = best_model.score(x_test, y_test)
                logging.info(f"SVC - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"SVC - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_model, train_score, test_score)
            else:
                result = best_model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=best_model.classes_)
                visualizer.visualize(best_model, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training SVC: {e}")
            raise CustomException(e, sys)


class XGBoostClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            if y_train.dtype == object or y_test.dtype == object:
                y_train_df = pd.DataFrame({'quality_label': y_train})
                y_test_df = pd.DataFrame({'quality_label': y_test})
                label_encoding_strategy = LabelEncoding(features=['quality_label'])
                fe = FeatureEngineering(strategy=label_encoding_strategy)
                y_train_encoded = fe.apply_feature_engineering(y_train_df)
                y_test_encoded = fe.apply_feature_engineering(y_test_df)
                y_train_mapped = y_train_encoded['quality_label']
                y_test_mapped = y_test_encoded['quality_label']
            else:
                y_train_mapped = y_train
                y_test_mapped = y_test

            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='hist',device='cuda')
            params = {

               'n_estimators': [100, 200, 300, 400],
               'max_depth': [3, 5, 7, 10],
               'learning_rate': [0.01, 0.05, 0.1, 0.2],
               'subsample': [0.7, 0.8, 1.0],
               'colsample_bytree': [0.7, 0.8, 1.0],
               'gamma': [0, 0.1, 0.2, 0.5]
}
            search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(model, params, cv=3, n_jobs=-1)

            search.fit(x_train, y_train_mapped)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for XGBoost: {best_params}")

            if return_metrics:
                train_score = best_model.score(x_train, y_train_mapped)
                test_score = best_model.score(x_test, y_test_mapped)
                logging.info(f"XGBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"XGBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_model, train_score, test_score)
            else:
                result = best_model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=['low', 'medium', 'high'])
                visualizer.visualize(best_model, x_test, y_test_mapped)
            return result
        except Exception as e:
            logging.error(f"Error training XGBoost Classifier: {e}")
            raise CustomException(e, sys)

class KNearestNeighborsClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            model = KNeighborsClassifier()
            params = {
                'n_neighbors': list(range(3, 21, 2)),
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
            search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42) if search_type == "random" \
                     else GridSearchCV(model, params, cv=3, n_jobs=-1)

            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            print(f"Best parameters for knn: {best_params}")           

            if return_metrics:
                train_score = best_model.score(x_train, y_train)
                test_score = best_model.score(x_test, y_test)
                logging.info(f"KNN - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"KNN - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (best_model, train_score, test_score)
            else:
                result = best_model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=best_model.classes_)
                visualizer.visualize(best_model, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training KNN model: {e}")
            raise CustomException(e, sys)


class GaussianNaiveBayesClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, **kwargs):
        try:
            model = GaussianNB()
            model.fit(x_train, y_train)

            if return_metrics:
                train_score = model.score(x_train, y_train)
                test_score = model.score(x_test, y_test)
                logging.info(f"Gaussian Naive Bayes - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"Gaussian Naive Bayes - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (model, train_score, test_score)
            else:
                result = model

            if visualize:
                visualizer = DefaultModelVisualizer(labels=model.classes_)
                visualizer.visualize(model, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training Gaussian Naive Bayes model: {e}")
            raise CustomException(e, sys)



class EnsembleVotingClassifierModel(ModelBuilding):
    def build_model(self, x_train, y_train, x_test, y_test, return_metrics=False, visualize=False, search_type="grid", n_iter=10):
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=10)
            xgb = XGBClassifier(eval_metric='mlogloss', tree_method='hist', device='cuda')
            knn = KNeighborsClassifier(n_neighbors=5)
            
            ensemble = VotingClassifier(
                estimators=[
                    ('random_forest', rf),
                    ('xgboost', xgb),
                    ('knn', knn)
                ],
                voting='soft'
            )
            ensemble.fit(x_train, y_train)
            
            if return_metrics:
                train_score = ensemble.score(x_train, y_train)
                test_score = ensemble.score(x_test, y_test)
                logging.info(f"Ensemble Voting - Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"Ensemble Voting - Train: {train_score:.4f}, Test: {test_score:.4f}")
                result = (ensemble, train_score, test_score)
            else:
                result = ensemble

            if visualize:
                visualizer = DefaultModelVisualizer(labels=np.unique(y_train))
                visualizer.visualize(ensemble, x_test, y_test)

            return result
        except Exception as e:
            logging.error(f"Error training Ensemble Voting Classifier: {e}")
            raise CustomException(e, sys)

class ModelBuilder:
    def __init__(self, model_type: str, search_type: str = "grid", n_iter: int = 10):
        self.model_type = model_type.lower()
        self.search_type = search_type
        self.n_iter = n_iter

    def build_model(self, x_train: pd.DataFrame, y_train: pd.Series,
                    x_test: Optional[pd.DataFrame] = None,
                    y_test: Optional[pd.Series] = None,
                    return_metrics: bool = False,
                    visualize: bool = False) -> Any:

        model_mapping = {
            "logistic_regression": LogisticRegressionModel(),
            "random_forest_classifier": RandomForestClassifierModel(),
            "svc": SupportVectorClassifierModel(),
            "xgboost_classifier": XGBoostClassifierModel(),
            "knn": KNearestNeighborsClassifierModel(),
            "naive_bayes": GaussianNaiveBayesClassifierModel(),
            "ensemble_voting_classifier": EnsembleVotingClassifierModel(),
            "final_random_forest_classifier": Final_RandomForestClassifierModel()
        }

        model_class = model_mapping.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model_class.build_model(
            x_train, y_train, x_test, y_test,
            return_metrics=return_metrics,
            visualize=visualize,
            search_type=self.search_type,
            n_iter=self.n_iter
        )