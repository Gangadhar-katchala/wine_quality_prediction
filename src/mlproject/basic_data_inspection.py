from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import sys
from src.exception import CustomException
from src.mlproject.logger import logging
from IPython.display import display

class DatainspectionStrategy(ABC):
    """Abstract base class for data inspection strategies."""
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        pass

class DataTypeInspectionStrategy(DatainspectionStrategy):
    """Strategy for inspecting data types and non-null counts."""
    def inspect(self, df: pd.DataFrame) -> None:
        try:
            print("\n" + "="*40)
            print("📝 DATA OVERVIEW")
            print("="*40)
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print("\nColumn Types:")
            col_types = pd.DataFrame({'dtype': df.dtypes.astype(str)})
            display(col_types)
            print("\nNon-Null Counts:")
            non_null = pd.DataFrame({'non-null count': df.notnull().sum()})
            display(non_null)
            logging.info("Data type and non-null inspection completed successfully.")
        except Exception as e:
            logging.error("Error during DataTypeInspectionStrategy.")
            raise CustomException(e, sys)

class SummaryStatisticsInspectionStrategy(DatainspectionStrategy):
    """Strategy for inspecting summary statistics."""
    def inspect(self, df: pd.DataFrame) -> None:
        try:
            print("\n" + "="*40)
            print("📈 SUMMARY STATISTICS (Numerical)")
            print("="*40)
            display(df.describe().T.style.background_gradient(axis=0))
            print("\n" + "="*40)
            print("🏷️ SUMMARY STATISTICS (Categorical)")
            print("="*40)
            display(df.describe(include=['object']).T.style.background_gradient(axis=0))
            logging.info("Summary statistics inspection completed successfully.")
        except Exception as e:
            logging.error("Error during SummaryStatisticsInspectionStrategy.")
            raise CustomException(e, sys)

class NullValueInspectionStrategy(DatainspectionStrategy):
    """Strategy for inspecting null values."""
    def inspect(self, df: pd.DataFrame) -> None:
        try:
            print("\n" + "="*40)
            print("❓ MISSING VALUE COUNTS")
            print("="*40)
            null_counts = df.isnull().sum()
            if null_counts.sum() == 0:
                print("✅ No missing values detected!")
            else:
                display(null_counts[null_counts > 0].to_frame('missing'))
            logging.info("Null value inspection completed successfully.")
        except Exception as e:
            logging.error("Error during NullValueInspectionStrategy.")
            raise CustomException(e, sys)

class DataInspector:
    """Executes multiple data inspection strategies."""
    _default_strategies = [
        DataTypeInspectionStrategy(),
        SummaryStatisticsInspectionStrategy(),
        NullValueInspectionStrategy()
    ]

    def __init__(self, additional_strategies: Optional[List[DatainspectionStrategy]] = None):
        self.strategies = self._default_strategies.copy()
        if additional_strategies:
            self.strategies.extend(additional_strategies)

    def execute_inspection(self, df: pd.DataFrame) -> None:
        """Executes all inspection strategies on the given DataFrame."""
        try:
            for strategy in self.strategies:
                strategy.inspect(df)
            logging.info("Data inspection completed successfully.")
        except Exception as e:
            logging.error("Error during data inspection execution.")
            raise CustomException(e, sys)