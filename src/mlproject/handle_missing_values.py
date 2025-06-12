from abc import ABC, abstractmethod
import pandas as pd
from exception import CustomException
from logger import logging
import sys

class MissingValueHandlingStrategy(ABC):
    """Abstract class for handling missing values."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy to drop rows with missing values."""

    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values."""
        try:
            logging.info("Dropping rows with missing values.")
            df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
            logging.info(f"Rows with missing values dropped: {df.shape[0] - df_cleaned.shape[0]}")
            return df_cleaned
        except Exception as e:
            logging.error(f"Error dropping missing values: {e}")
            raise CustomException(e, sys)

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy to fill missing values with a specified value."""

    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"Filling missing values with {self.method}.")
            df_cleaned = df.copy()
            if self.method == "mean":
                numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
            elif self.method == "median":
                numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
            elif self.method == "mode":
                categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
                for column in categorical_columns:
                    if df_cleaned[column].isnull().any():
                        df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
            elif self.method == "constant":
                df_cleaned.fillna(self.fill_value, inplace=True)
            else:
                logging.warning(f"Unknown method: {self.method}. No filling applied.")
            logging.info(f"Missing values filled using {self.method}.")
            return df_cleaned
        except Exception as e:
            logging.error(f"Error filling missing values: {e}")
            raise CustomException(e, sys)

class MissingValueHandler:
    """Class to handle missing values in a DataFrame."""

    def __init__(self, strategy: MissingValueHandlingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """Set the strategy for handling missing values."""
        self.strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using the specified strategy."""
        logging.info("Handling missing values.")
        return self.strategy.handle(df)