import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from src.exception import CustomException
from src.mlproject.logger import logging
from typing import Any, Optional
from sklearn import metrics

# Abstract base class for visualizers
class ModelVisualizer(ABC):
    @abstractmethod
    def visualize(self, model: Any, x: pd.DataFrame, y: pd.Series) -> None:
        pass

# Helper class for creating classification plots
class ModelVisualization:
    def plot_classification_report(self, y_test: pd.Series, y_pred: pd.Series) -> None:
        try:
            cr = pd.DataFrame(metrics.classification_report(y_test, y_pred, digits=3, output_dict=True)).T
            if 'support' in cr.columns:
                cr = cr.drop(columns='support')
            plt.figure(figsize=(8, 4))
            ax = sns.heatmap(cr, cmap='Purples', annot=True, linecolor='white', linewidths=0.5)
            ax.xaxis.tick_top()
            plt.title('Classification Report Heatmap', pad=20)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting classification report: {e}")
            raise CustomException(e, sys)

    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: pd.Series, labels=None) -> None:
        try:
            cm = metrics.confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            ax = sns.heatmap(cm, annot=True, fmt='', cmap="Purples")
            ax.set_xlabel('Predicted labels', fontsize=14)
            ax.set_ylabel('True labels', fontsize=14)
            ax.set_title('Confusion Matrix', fontsize=18)
            if labels is not None:
                ax.xaxis.set_ticklabels(labels)
                ax.yaxis.set_ticklabels(labels)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            raise CustomException(e, sys)

    def classification_plot(self, y_test: pd.Series, y_pred: pd.Series, labels=None) -> None:
        try:
            cm = metrics.confusion_matrix(y_test, y_pred)
            cr = pd.DataFrame(metrics.classification_report(y_test, y_pred, digits=3, output_dict=True)).T
            if 'support' in cr.columns:
                cr = cr.drop(columns='support')

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))

            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='', cmap="Purples", ax=ax[0])
            ax[0].set_xlabel('Predicted labels', fontsize=14)
            ax[0].set_ylabel('True labels', fontsize=14)
            ax[0].set_title('Confusion Matrix', fontsize=18)
            if labels is not None:
                ax[0].xaxis.set_ticklabels(labels)
                ax[0].yaxis.set_ticklabels(labels)

            # Classification Report
            sns.heatmap(cr, cmap='Purples', annot=True, linecolor='white', linewidths=0.5, ax=ax[1])
            ax[1].xaxis.tick_top()
            ax[1].set_title('Classification Report', fontsize=18)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting combined classification plots: {e}")
            raise CustomException(e, sys)

# Concrete implementation of the visualizer
class DefaultModelVisualizer(ModelVisualizer):
    def __init__(self, labels: Optional[list] = None):
        self.visualizer = ModelVisualization()
        self.labels = labels  # Optional class labels for ticks

    def visualize(self, model: Any, x: pd.DataFrame, y: pd.Series) -> None:
        try:
            y_pred = model.predict(x)
            self.visualizer.classification_plot(y, y_pred, labels=self.labels)
        except Exception as e:
            logging.error(f"Error visualizing model: {e}")
            raise CustomException(e, sys)
