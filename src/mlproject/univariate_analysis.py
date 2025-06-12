from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.mlproject.visulization import ColumnClassifier
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, column: str) -> None:
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, column: str) -> None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # Histogram/KDE
            sns.histplot(df[column], kde=True, bins=30, ax=axes[0])
            axes[0].set_title(f'Histogram and KDE for {column}')
            axes[0].set_xlabel(column)
            axes[0].set_ylabel('Frequency')
            # Boxplot
            sns.boxplot(x=df[column], ax=axes[1], orient='h')
            axes[1].set_title(f'Boxplot for {column}')
            axes[1].set_xlabel(column)
            plt.tight_layout()
            logging.info(f"Numerical Univariate Analysis for {column}:")
            plt.show()
        except Exception as e:
            logging.error(f"Error in numerical univariate analysis for {column}: {e}")
            raise CustomException(e, sys)

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, column: str) -> None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # Countplot
            sns.countplot(data=df, x=column, ax=axes[0])
            axes[0].set_title(f'Count Plot for {column}')
            axes[0].set_xlabel(column)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            # Pie chart
            df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
            axes[1].set_title(f'Pie Chart for {column}')
            axes[1].set_ylabel('')
            plt.tight_layout()
            logging.info(f"Categorical Univariate Analysis for {column}:")
            plt.show()
        except Exception as e:
            logging.error(f"Error in categorical univariate analysis for {column}: {e}")
            raise CustomException(e, sys)

class UnivariateAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.classifier = ColumnClassifier(df)
        self.strategies = {
            'numerical': NumericalUnivariateAnalysis(),
            'categorical': CategoricalUnivariateAnalysis()
        }

    def classify_column(self, column: str) -> str:
        try:
            if column in self.classifier.get_numerical_columns():
                return 'numerical'
            elif column in self.classifier.get_categorical_columns():
                return 'categorical'
            else:
                return 'other'
        except Exception as e:
            logging.error(f"Error classifying column {column}: {e}")
            raise CustomException(e, sys)

    def analyze(self) -> None:
        for column in self.df.columns:
            column_type = self.classify_column(column)
            if column_type == 'numerical':
                self.strategies['numerical'].analyze(self.df, column)
            elif column_type == 'categorical':
                self.strategies['categorical'].analyze(self.df, column)
            else:
                print(f"Skipping column '{column}' (not numerical or categorical).")