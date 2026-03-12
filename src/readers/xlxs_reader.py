# derived from data_loader.py, read from csv file
import pandas as pd
from src.data_reader import DataReader
from src.data_formatter import DataFormatter

class XLXSReader(DataReader):
    def __init__(self, file_path, rename_columns=True):
        self.file_path = file_path
        self.rename_columns = rename_columns
        self.formatter = DataFormatter()
    
    def read_data(self):
        df = pd.read_csv(self.file_path)
        if self.rename_columns:
            df = self.formatter.rename_all_columns(df)
        return df