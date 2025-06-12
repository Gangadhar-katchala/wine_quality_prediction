import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
from src.exception import CustomException
from src.mlproject.logger import logging
import sys
import shutil


class DataIngestion(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestion):
    def ingest(self, file_path: str, extract_dir: str = "extracted_files") -> pd.DataFrame:
        if not file_path.endswith('.zip'):
            raise ValueError("File is not a zip file")

        df = None 
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logging.info(f"Extracted files to directory: {extract_dir}")

            extracted_files = os.listdir(extract_dir)
            csv_files = [f for f in extracted_files if f.endswith('.csv')]
            excel_files = [f for f in extracted_files if f.endswith(('.xlsx', '.xls'))]

            if csv_files:
                if len(csv_files) > 1:
                    raise ValueError(f"Multiple CSV files found: {csv_files}")
                csv_file_path = os.path.join(extract_dir, csv_files[0])
                df = CSVDataIngestor().ingest(csv_file_path)  # Store result in df

            elif excel_files:
                if len(excel_files) > 1:
                    raise ValueError(f"Multiple Excel files found: {excel_files}")
                excel_file_path = os.path.join(extract_dir, excel_files[0])
                df = ExcelDataIngestor().ingest(excel_file_path)  # Store result in df

            else:
                raise ValueError(f"No CSV/Excel files found in: {extracted_files}")

            return df

        except zipfile.BadZipFile:
            raise ValueError("Corrupted zip file")
        except Exception as e:
            raise RuntimeError(f"Processing error: {e}")
        finally:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                logging.info(f"Cleaned up: {extract_dir}")

class CSVDataIngestor(DataIngestion):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith('.csv'):
            raise ValueError("Not a CSV file")
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded CSV: {file_path}")
            return df
        except Exception as e:
            raise RuntimeError(f"CSV error: {e}")

class ExcelDataIngestor(DataIngestion):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(('.xlsx', '.xls')):
            raise ValueError("Not an Excel file")
        try:
            df = pd.read_excel(file_path)
            logging.info(f"Loaded Excel: {file_path}")
            return df
        except Exception as e:
            raise RuntimeError(f"Excel error: {e}")

class DataIngestionFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestion:
        if file_path.endswith('.zip'):
            return ZipDataIngestor()
        elif file_path.endswith('.csv'):
            return CSVDataIngestor()
        elif file_path.endswith(('.xlsx', '.xls')):
            return ExcelDataIngestor()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

