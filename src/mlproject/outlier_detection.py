import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exception import CustomException
from logger import logging
import sys

class OutlierDetectionStrategy(ABC):
    """Abstract class for outlier detection."""

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Detect outliers in the DataFrame for a specific column."""
        pass

class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """Z-score based outlier detection."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Detect outliers using Z-score method."""
        try:
            logging.info(f"Detecting outliers using Z-score method for column: {column}")
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            outliers = df[np.abs(z_scores) > self.threshold]
            logging.info(f"Outliers detected using Z-score method: {outliers.shape[0]}")
            return outliers
        except Exception as e:
            logging.error(f"Error in Z-score outlier detection: {e}")
            raise CustomException(e, sys)

class IQROutlierDetection(OutlierDetectionStrategy):
    """IQR based outlier detection."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        try:
            logging.info(f"Detecting outliers using IQR method for column: {column}")
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (self.multiplier * IQR)
            upper_bound = Q3 + (self.multiplier * IQR)
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            logging.info(f"Outliers detected using IQR method: {outliers.shape[0]}")
            return outliers
        except Exception as e:
            logging.error(f"Error in IQR outlier detection: {e}")
            raise CustomException(e, sys)

class OutlierDetector:
    """Class to detect and handle outliers in a DataFrame."""

    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """Set the strategy for outlier detection."""
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Detect outliers in the DataFrame using the specified strategy."""
        logging.info(f"Detecting outliers in column: {column} using strategy: {self.strategy.__class__.__name__}")
        return self.strategy.detect_outliers(df, column)

    def handle_outliers(self, df: pd.DataFrame, column: str, method="remove") -> pd.DataFrame:
        """Handle outliers in the DataFrame for a specific column."""
        logging.info(f"Handling outliers in column: {column} using method: {method}")
        outliers = self.detect_outliers(df, column)
        if method == "remove":
            df_cleaned = df.drop(outliers.index)
            logging.info(f"Outliers removed: {outliers.shape[0]}")
            return df_cleaned
        elif method == "cap":
            # Cap outliers to the bounds
            if isinstance(self.strategy, IQROutlierDetection):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.strategy.multiplier * IQR)
                upper_bound = Q3 + (self.strategy.multiplier * IQR)
            elif isinstance(self.strategy, ZScoreOutlierDetection):
                mean = df[column].mean()
                std = df[column].std()
                lower_bound = mean - (self.strategy.threshold * std)
                upper_bound = mean + (self.strategy.threshold * std)
            else:
                logging.warning("Unknown strategy for capping. No capping applied.")
                return df
            df_cleaned = df.copy()
            df_cleaned[column] = np.clip(df_cleaned[column], lower_bound, upper_bound)
            logging.info("Outliers capped.")
            return df_cleaned
        else:
            logging.warning(f"Unknown method: {method}. No handling applied.")
            return df

    def visualize_outliers(self, df: pd.DataFrame, features: list) -> None:
        """Visualize outliers using box plots."""
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Box plot for {feature}")
            plt.xlabel(feature)
            plt.show()
        logging.info("Outlier visualization completed.")