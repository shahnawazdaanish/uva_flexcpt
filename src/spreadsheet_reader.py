import pandas as pd
from src.data_reader import DataReader

class SpreadSheetReader(DataReader):
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self, sheet_name=0, skiprows=[], index_col=None, header=None):
        """
        Reads an Excel or Csv file and returns a DataFrame.
        """
        if not (self.file_path.endswith('.xlsx') or self.file_path.endswith('.csv')):
            raise ValueError("File path must point to an Excel or CSV file.")
        if not pd.io.common.file_exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Read the Excel file using pandas
        if self.file_path.endswith('.xlsx'):
            return pd.read_excel(self.file_path, sheet_name=sheet_name, skiprows=skiprows, index_col=index_col, header=header)
        else:
            return pd.read_csv(self.file_path, skiprows=skiprows, index_col=index_col, header=header)
