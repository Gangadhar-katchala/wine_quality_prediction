import os
import sys
import dill
from typing import Any, List
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from src.exception import CustomException
from src.mlproject.logger import logging


def save_object(obj, file_path):
    """
    Saves a Python object to the specified file path using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved successfully at {file_path}")
        return file_path
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Loads a Python object from the specified file path using dill.
    """
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def save_figures_to_pdf(figures: List[Figure], pdf_path: str) -> str:
    """
    Saves a list of matplotlib Figure objects to a single PDF file.
    """
    try:
        dir_path = os.path.dirname(pdf_path)
        os.makedirs(dir_path, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        logging.info(f"Figures saved successfully to PDF at {pdf_path}")
        return pdf_path
    except Exception as e:
        raise CustomException(e, sys)