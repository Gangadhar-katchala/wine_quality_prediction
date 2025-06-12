from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomException
from src.mlproject.logger import logging
import sys

class ColumnClassifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_numerical_columns(self) -> list:
        numerical_columns = [col for col in self.df.columns 
                if pd.api.types.is_numeric_dtype(self.df[col])]
        logging.info(f"Numerical Columns: {numerical_columns}")
        print(f"Numerical Columns: {numerical_columns}")
        return numerical_columns

    def get_categorical_columns(self) -> list:
        categorical_columns = [col for col in self.df.columns 
                if not pd.api.types.is_numeric_dtype(self.df[col]) 
                and not pd.api.types.is_datetime64_any_dtype(self.df[col])]
        logging.info(f"Categorical Columns: {categorical_columns}")
        print(f"Categorical Columns: {categorical_columns}")
        return categorical_columns

class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, df: pd.DataFrame, column: str) -> None:
        pass

class NumericalVisualization(VisualizationStrategy):
    _plot_methods = {
        'histplot': 'visualize_histplot',
        'boxplot': 'visualize_boxplot',
        'violinplot': 'visualize_violinplot',
        'density': 'visualize_density_plot',
        'lineplot': 'visualize_lineplot'
    }

    def visualize(self, df: pd.DataFrame, column: str) -> None:
        self.visualize_histplot(df, column)
    
    def visualize_histplot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Numerical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Histogram and KDE for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        logging.info(f"histplot for {column}:")
        plt.show()
    
    def visualize_boxplot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Boxplot for Numerical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot for {column}')
        plt.xlabel(column)
        logging.info(f"Boxplot for {column}:")
        plt.show()
    
    def visualize_scatterplot(self, df: pd.DataFrame, column1: str, column2: str) -> None:
        print(f"Visualizing Scatterplot for Numerical Columns: {column1} vs {column2}")
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x=column1, y=column2)
        plt.title(f'Scatterplot for {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        logging.info(f"Scatterplot for {column1} vs {column2}:")
        plt.show()
    
    def visualize_pairplot(self, df: pd.DataFrame, columns: List[str]) -> None:
        print(f"Visualizing Pairplot for Numerical Columns: {columns}")
        sns.pairplot(df[columns])
        plt.suptitle(f'Pairplot for {columns}', y=1.02)
        logging.info(f"Pairplot for {columns}:")
        plt.show()
    
    def visualize_correlation_matrix(self, df: pd.DataFrame) -> None:
        print("Visualizing Correlation Matrix")
        plt.figure(figsize=(12, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        logging.info("Correlation Matrix:")
        plt.show()
    
    def visualize_violinplot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Violin Plot for Numerical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=df[column])
        plt.title(f'Violin Plot for {column}')
        plt.xlabel(column)
        logging.info(f"Violin Plot for {column}:")
        plt.show()
    
    def visualize_lineplot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Line Plot for Numerical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=df.index, y=column)
        plt.title(f'Line Plot for {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        logging.info(f"Line Plot for {column}:")
        plt.show()
    
    def visualize_density_plot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Density Plot for Numerical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df[column], fill=True)
        plt.title(f'Density Plot for {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        logging.info(f"Density Plot for {column}:")
        plt.show()

    def get_plot_method(self, plot_type: str) -> Tuple[Callable, str]:
        method_name = self._plot_methods.get(plot_type, 'visualize_histplot')
        return getattr(self, method_name), method_name

class CategoricalVisualization(VisualizationStrategy):
    _plot_methods = {
        'countplot': 'visualize_countplot',
        'pie': 'visualize_pie_chart',
        'bar': 'visualize_bar_chart'
    }

    def visualize(self, df: pd.DataFrame, column: str) -> None:
        self.visualize_countplot(df, column)
    
    def visualize_countplot(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Count Plot for Categorical Column: {column}")
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=column)
        plt.title(f'Count Plot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        logging.info(f"Count Plot for {column}:")
        plt.show()
    
    def visualize_pie_chart(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Pie Chart for Categorical Column: {column}")
        plt.figure(figsize=(12, 6))
        df[column].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart for {column}')
        plt.ylabel('')
        logging.info(f"Pie Chart for {column}:")
        plt.show()
    
    def visualize_bar_chart(self, df: pd.DataFrame, column: str) -> None:
        print(f"Visualizing Bar Chart for Categorical Column: {column}")
        plt.figure(figsize=(12, 6))
        df[column].value_counts().plot.bar()
        plt.title(f'Bar Chart for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        logging.info(f"Bar Chart for {column}:")
        plt.show()
    
    def visualize_grouped_bar_chart(self, df: pd.DataFrame, column1: str, column2: str) -> None:
        print(f"Visualizing Grouped Bar Chart for Categorical Columns: {column1} vs {column2}")
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=column1, hue=column2)
        plt.title(f'Grouped Bar Chart for {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        logging.info(f"Grouped Bar Chart for {column1} vs {column2}:")
        plt.show()

    def get_plot_method(self, plot_type: str) -> Tuple[Callable, str]:
        method_name = self._plot_methods.get(plot_type, 'visualize_countplot')
        return getattr(self, method_name), method_name

class VisualizationExecutor:
    def __init__(self, strategies: Optional[Dict[str, str]] = None):
        self.classifier = None
        self.strategies = strategies or {}
        self.numerical_strategy = NumericalVisualization()
        self.categorical_strategy = CategoricalVisualization()

    def visualize_all(self, df: pd.DataFrame) -> None:
        self.classifier = ColumnClassifier(df)
        
        numerical_columns = self.classifier.get_numerical_columns()
        categorical_columns = self.classifier.get_categorical_columns()
        
        self._visualize_columns(df, numerical_columns, 'numerical')
        self._visualize_columns(df, categorical_columns, 'categorical')

    def _visualize_columns(self, df: pd.DataFrame, columns: list, col_type: str) -> None:
        for col in columns:
            if col in self.strategies:
                self._visualize_custom(df, col, col_type)
            else:
                self._visualize_default(df, col, col_type)

    def _visualize_default(self, df: pd.DataFrame, col: str, col_type: str) -> None:
        strategy = self.numerical_strategy if col_type == 'numerical' else self.categorical_strategy
        strategy.visualize(df, col)

    def _visualize_custom(self, df: pd.DataFrame, col: str, col_type: str) -> None:
        plot_type = self.strategies[col]
        try:
            strategy = self.numerical_strategy if col_type == 'numerical' else self.categorical_strategy
            method, method_name = strategy.get_plot_method(plot_type)
            
            print(f"Custom {plot_type} for {col_type} column '{col}'")
            if method_name == 'visualize_scatterplot':
                # Handle special case for scatterplot
                secondary_col = input(f"Enter secondary column for {col} scatterplot: ")
                method(df, col, secondary_col)
            else:
                method(df, col)
        except Exception as e:
            print(f"Error creating {plot_type} for {col}: {str(e)}")
            self._visualize_default(df, col, col_type)

