import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class DataSplittingStrategy(ABC):
    """Abstract class for data splitting strategies."""
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        """Split the DataFrame into sets."""
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple train-test split strategy."""
    def __init__(self, test_size: float = 0.2, random_state: int = 42): 
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

class TrainValTestSplitStrategy(DataSplittingStrategy):
    """Train/Validation/Test split strategy."""
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            # First split: train vs temp (test+val)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.test_size + self.val_size, random_state=self.random_state
            )
            # Validation size as a proportion of temp set
            val_ratio = self.val_size / (self.test_size + self.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1 - val_ratio, random_state=self.random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logging.error("Error during TrainValTestSplitStrategy.")
            raise CustomException(e, sys)

class DataSplitter:
    """Class to split data into sets using a chosen strategy."""
    def __init__(self, strategy: DataSplittingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        self.strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str) -> tuple:
        try:
            logging.info(f"Splitting data using strategy: {self.strategy.__class__.__name__}")
            return self.strategy.split_data(df, target_column)
        except Exception as e:
            logging.error("Error during data splitting in DataSplitter.")
            raise CustomException(e, sys)
