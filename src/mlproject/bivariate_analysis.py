from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.mlproject.visulization import ColumnClassifier
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, x: str, y: str) -> None:
        pass   

class NumericalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, x: str, y: str) -> None:
        try:
            logging.info(f"Bivariate Analysis for Numerical Columns: {x} and {y}")
            print(f"\n{'='*40}\n🔗 Bivariate Analysis for Numerical Columns: {x} vs {y}\n{'='*40}")
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            # Scatter Plot
            sns.scatterplot(data=df, x=x, y=y, ax=axes[0])
            axes[0].set_title(f'Scatter Plot: {x} vs {y}')
            axes[0].set_xlabel(x)
            axes[0].set_ylabel(y)
            # Hexbin Plot
            axes[1].hexbin(df[x], df[y], gridsize=30, cmap='Blues')
            axes[1].set_title(f'Hexbin Plot: {x} vs {y}')
            axes[1].set_xlabel(x)
            axes[1].set_ylabel(y)
            # KDE Plot
            sns.kdeplot(x=df[x], y=df[y], fill=True, cmap='mako', ax=axes[2])
            axes[2].set_title(f'KDE Plot: {x} vs {y}')
            axes[2].set_xlabel(x)
            axes[2].set_ylabel(y)
            plt.tight_layout()
            plt.show()
            logging.info(f"Multiple bivariate plots between {x} and {y} displayed successfully.")
        except Exception as e:
            logging.error(f"Error during NumericalBivariateAnalysis for {x} and {y}.")
            raise CustomException(e, sys)

class CategoricalvsNumericalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, x: str, y: str) -> None:
        try:
            logging.info(f"Bivariate Analysis for Categorical Column {x} and Numerical Column {y}")
            print(f"\n{'='*40}\n🔗 Bivariate Analysis for Categorical {x} vs Numerical {y}\n{'='*40}")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # Box Plot
            sns.boxplot(data=df, x=x, y=y, ax=axes[0])
            axes[0].set_title(f'Box Plot of {y} by {x}')
            axes[0].set_xlabel(x)
            axes[0].set_ylabel(y)
            axes[0].tick_params(axis='x', rotation=45)
            # Violin Plot
            sns.violinplot(data=df, x=x, y=y, ax=axes[1])
            axes[1].set_title(f'Violin Plot of {y} by {x}')
            axes[1].set_xlabel(x)
            axes[1].set_ylabel(y)
            axes[1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()
            logging.info(f"Box and violin plots of {y} by {x} displayed successfully.")
        except Exception as e:
            logging.error(f"Error during CategoricalvsNumericalBivariateAnalysis for {x} and {y}.")
            raise CustomException(e, sys)

class BivariateAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.classifier = ColumnClassifier(df)
        self.numerical_columns = self.classifier.get_numerical_columns()
        self.categorical_columns = self.classifier.get_categorical_columns()
        self.strategies = {
            'num_num': NumericalBivariateAnalysis(),
            'cat_num': CategoricalvsNumericalBivariateAnalysis()
        }

    def analyze(self) -> None:
        try:
            logging.info("Starting bivariate analysis for all column pairs.")
            # Numerical vs Numerical
            for i, col1 in enumerate(self.numerical_columns):
                for col2 in self.numerical_columns[i+1:]:
                    self.strategies['num_num'].analyze(self.df, col1, col2)
            # Categorical vs Numerical
            for cat_col in self.categorical_columns:
                for num_col in self.numerical_columns:
                    self.strategies['cat_num'].analyze(self.df, cat_col, num_col)
            logging.info("Bivariate analysis completed successfully.")
        except Exception as e:
            logging.error("Error during bivariate analysis execution.")
            raise CustomException(e, sys)