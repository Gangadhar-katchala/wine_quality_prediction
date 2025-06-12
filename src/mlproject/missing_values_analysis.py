from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class MissingValuesAnalysisStrategy(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        try:
            self.identify_missing_values(df)
            self.visualize_missing_values(df)
        except Exception as e:
            logging.error(f"Error during missing values analysis: {e}")
            raise CustomException(e, sys)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        pass

class SimpleMissingValuesAnalysis(MissingValuesAnalysisStrategy):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Identifying missing values...")
            missing_count = df.isnull().sum()
            missing_percent = df.isnull().mean() * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing_count,
                'Missing %': missing_percent
            })
            print("\n" + "="*40)
            print("❓ MISSING VALUES SUMMARY")
            print("="*40)
            if missing_count.sum() == 0:
                print("✅ No missing values detected!")
            else:
                print(missing_df[missing_df['Missing Count'] > 0].to_markdown(floatfmt=".2f"))
            logging.info("Missing values identified.")
        except Exception as e:
            logging.error(f"Error identifying missing values: {e}")
            raise CustomException(e, sys)

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        try:
            if df.isnull().sum().sum() == 0:
                logging.info("No missing values to visualize.")
                return
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            logging.info("Visualizing missing values...")
            plt.show()
        except Exception as e:
            logging.error(f"Error visualizing missing values: {e}")
            raise CustomException(e, sys)