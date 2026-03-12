# derived from data_reader.py
import pandas as pd
from main.src.readers.data_reader import DataReader

class ParquetReader(DataReader):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_data(self):
        if not self.file_path.endswith('.parquet'):
            raise ValueError("File path must point to a Parquet file.")
        if not pd.io.common.file_exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        try:
            df = pd.read_parquet(self.file_path, engine='pyarrow')
        except Exception as e:
            # If pyarrow fails (e.g., ArrowKeyError), try fastparquet
            if 'pyarrow' in str(type(e).__name__) or 'ArrowKeyError' in str(type(e).__name__):
                print(f"Warning: pyarrow failed ({type(e).__name__}), falling back to fastparquet")
                df = pd.read_parquet(self.file_path, engine='fastparquet')
            else:
                raise
        return df