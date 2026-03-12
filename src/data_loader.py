import numpy as np
from src.constant_manager import ConstantManager
from src.readers.spreadsheet_reader import SpreadSheetReader
from src.data_formatter import DataFormatter
from src.readers.parquet_reader import ParquetReader
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataLoader:
    def __init__(self, file_path, sheet_index = 0):
        self.file_path = file_path
        self.primary_sheet_name = sheet_index

    def load_filetype(self, file_path):
        allowed_types = ConstantManager().ALLOWED_READER_TYPES
        if file_path.endswith('.xlsx'):
            return ConstantManager().READER_TYPE_XLSX
        elif file_path.endswith('.parquet'):
            return ConstantManager().READER_TYPE_PARQUET
        else:
            raise ValueError(f"Unsupported file type. Allowed types are: {allowed_types}")

    def load_data(self, rename_columns=False, ignore_unnamed=False, skiprows=None, index_col=None, header=None):
        # Validate file type before attempting to read
        reader_type = self.load_filetype(self.file_path)
        allowed_types = ConstantManager().ALLOWED_READER_TYPES
        if reader_type not in allowed_types:
            raise ValueError(f"Unsupported reader type: {reader_type}. Allowed types are: {allowed_types}")
        
        # Read the data using the appropriate reader based on file type
        if reader_type == ConstantManager().READER_TYPE_XLSX:
            spreadsheet_reader = SpreadSheetReader(self.file_path)
            df = spreadsheet_reader.read_data(
                sheet_name=self.primary_sheet_name, 
                skiprows=skiprows, 
                index_col=index_col, 
                header=header
            )
        elif reader_type == ConstantManager().READER_TYPE_PARQUET:
            df = ParquetReader(self.file_path).read_data()
        else:
            raise ValueError(f"Unsupported reader type: {reader_type}")

        if rename_columns:
            df = self.formatter.rename_all_columns(df, ignore_unnamed=ignore_unnamed)
        return df
    

    def save_df_to_parquet(self, df, output_path):
        try:
            df.to_parquet(output_path, engine='pyarrow', index=True)
        except Exception as e:
            # If pyarrow fails (e.g., ArrowKeyError), try fastparquet
            if 'pyarrow' in str(type(e).__name__) or 'ArrowKeyError' in str(type(e).__name__):
                print(f"Warning: pyarrow failed ({type(e).__name__}), falling back to fastparquet")
                df.to_parquet(output_path, engine='fastparquet', index=True)
            else:
                raise
        return output_path
















    def load_raw_data(self, rename_columns=False, ignore_unnamed=False, skiprows=None, index_col=None, header=None):
        spreadsheet_reader = SpreadSheetReader(self.raw_file_path)
        df = spreadsheet_reader.read_file(sheet_name=self.primary_sheet_name, skiprows=skiprows, index_col=index_col, header=header)
        if rename_columns:
            df = self.formatter.rename_all_columns(df, ignore_unnamed=ignore_unnamed)
        return df
    
    def load_main_data(self):
        spreadsheet_reader = SpreadSheetReader(self.main_file_path)
        df = spreadsheet_reader.read_file(sheet_name=self.primary_sheet_name, header=0, index_col=0)
        df = self.formatter.filter_columns(df)
        df = self.formatter.rename_columns(df)
        df = self.formatter.encode_categorical_columns(df, 'cat_1')
        return df

    def load_reference_data(self):
        spreadsheet_reader = SpreadSheetReader(self.raw_file_path)
        df = spreadsheet_reader.read_file(sheet_name=self.reference_sheet_name, skiprows=[1, 2, 3], index_col=0, header=0)
        df = self.formatter.filter_columns(df)
        df = self.formatter.rename_columns(df)
        df = self.formatter.encode_categorical_columns(df, 'cat_1')
        return df
    
    def merge_reference_data(self, df_main):
        df_ref = self.load_reference_data()
        
        df_merged = pd.concat([df_main, df_ref])
        return df_merged

    def scale_data(self, dataframe, scaler=None):
        data = dataframe.copy(deep=True)
        cols_to_ignore = ['cat_1_1', 'cat_1_2']
        cols_to_scale = [col for col in data.columns if col not in cols_to_ignore]
        
        if scaler is None:
            scaler = StandardScaler()
            data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        else:
            data[cols_to_scale] = scaler.transform(data[cols_to_scale])
        
        return data, scaler
    
    def load_grouped_data(self, dataframe, scaled=False, scaler=None):
        data = dataframe.copy(deep=True)
        
        if scaled:
            cols_to_ignore = ['cat_1_1', 'cat_1_2']
            cols_to_scale = [col for col in data.columns if col not in cols_to_ignore]
            
            if scaler is None:
                scaler = StandardScaler()
                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
            else:
                data[cols_to_scale] = scaler.transform(data[cols_to_scale])

        print(f"Mean: {data[cols_to_scale].mean().to_dict()}, Var: {data[cols_to_scale].var(ddof=1).to_dict()}")
        group1_data = data[(data['cat_1_1'] == 0) & (data['cat_1_2'] == 0)]
        group2_data = data[(data['cat_1_1'] == 1) & (data['cat_1_2'] == 0)]
        group3_data = data[(data['cat_1_1'] == 0) & (data['cat_1_2'] == 1)]

        return group1_data[cols_to_scale], group2_data[cols_to_scale], group3_data[cols_to_scale], scaler
    
    
    def load_X(self, dataframe, abstracted=False):
        """
        Load the input features from the dataframe with column names.
        """
        if abstracted:
            return self.formatter_abs.filter_input_columns(dataframe)
        return self.formatter.filter_input_columns(dataframe)
    
    def load_y(self, dataframe, abstracted=False):
        """
        Load the output features from the dataframe with column names.
        """
        if abstracted:
            return self.formatter_abs.filter_output_columns(dataframe)
        return self.formatter.filter_output_columns(dataframe)