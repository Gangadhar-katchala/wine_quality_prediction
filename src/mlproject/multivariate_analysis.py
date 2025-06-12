from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class MultivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame) -> None:
        try:
            self.generate_correlation_heatmap(df)
        except Exception as e:
            logging.error(f"Error during multivariate analysis: {e}")
            raise CustomException(e, sys)

    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        try:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt='.2f',
                linewidths=0.5,
                cmap='Purples',
                mask=mask,
                square=True,
                cbar_kws={"shrink": .8}
            )
            plt.title('Correlation Heatmap')
            logging.info("Generated correlation heatmap.")
            plt.show()
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {e}")
            raise CustomException(e, sys)