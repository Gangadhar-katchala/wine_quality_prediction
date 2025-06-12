import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class FeatureEngineeringStrategy(ABC):
    """Abstract class for feature engineering."""

    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformation to the DataFrame."""
        pass

class LogTransformation(FeatureEngineeringStrategy):
    """Log transformation for feature engineering."""

    def __init__(self, features: list):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to the specified features."""
        try:
            logging.info(f"Applying log transformation to features: {self.features}")
            df_transformed = df.copy()
            for feature in self.features:
                df_transformed[feature] = np.log1p(df_transformed[feature])
            logging.info(f"Log transformation applied to features: {self.features}")
            return df_transformed
        except Exception as e:
            logging.error(f"Error in log transformation: {e}")
            raise CustomException(e, sys)

class StandardScaling(FeatureEngineeringStrategy):
    """Standard scaling for feature engineering."""

    def __init__(self, features: list):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to the specified features."""
        try:
            logging.info(f"Applying standard scaling to features: {self.features}")
            df_transformed = df.copy()
            df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
            logging.info(f"Standard scaling applied to features: {self.features}")
            return df_transformed
        except Exception as e:
            logging.error(f"Error in standard scaling: {e}")
            raise CustomException(e, sys)

class MinMaxScaling(FeatureEngineeringStrategy):
    """Min-max scaling for feature engineering."""

    def __init__(self, features: list):
        self.features = features
        self.scaler = MinMaxScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling to the specified features."""
        try:
            logging.info(f"Applying min-max scaling to features: {self.features}")
            df_transformed = df.copy()
            df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
            logging.info(f"Min-max scaling applied to features: {self.features}")
            return df_transformed
        except Exception as e:
            logging.error(f"Error in min-max scaling: {e}")
            raise CustomException(e, sys)

class OneHotEncoding(FeatureEngineeringStrategy):
    """One-hot encoding for categorical features."""

    def __init__(self, features: list):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to the specified categorical features."""
        try:
            logging.info(f"Applying one-hot encoding to features: {self.features}")
            df_transformed = df.copy()
            encoded_features = self.encoder.fit_transform(df_transformed[self.features])
            encoded_feature_names = self.encoder.get_feature_names_out(self.features)
            df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_transformed.index)
            df_transformed = pd.concat([df_transformed.drop(columns=self.features), df_encoded], axis=1)
            logging.info(f"One-hot encoding applied to features: {self.features}")
            return df_transformed
        except Exception as e:
            logging.error(f"Error in one-hot encoding: {e}")
            raise CustomException(e, sys)

class LabelEncoding(FeatureEngineeringStrategy):
    """Label encoding for categorical features."""

    def __init__(self, features: list):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to the specified categorical features."""
        try:
            logging.info(f"Applying label encoding to features: {self.features}")
            df_transformed = df.copy()
            for feature in self.features:
                encoder = LabelEncoder()
                df_transformed[feature] = encoder.fit_transform(df_transformed[feature])
            logging.info(f"Label encoding applied to features: {self.features}")
            return df_transformed
        except Exception as e:
            logging.error(f"Error in label encoding: {e}")
            raise CustomException(e, sys)

class FeatureEngineering:
    """Class to apply feature engineering transformations."""

    def __init__(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """Set the strategy for feature engineering."""
        self.strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering using the specified strategy."""
        logging.info("Applying feature engineering.")
        return self.strategy.apply_transformation(df)

