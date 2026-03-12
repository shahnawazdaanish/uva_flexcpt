import pandas as pd

class DataCleaner:
    def __init__(self, df):
        # Validate that the input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        self.df = df

    def exclude_null_columns(self, consider_zeros_as_null=False):
        # Remove columns that are entirely empty
        if consider_zeros_as_null:
            self.df = self.df.replace(0, pd.NA)

        self.df = self.df.dropna(axis=1, how='all')
        return self.df
    
    def null_check(self):
        # Check for null values in the DataFrame
        null_counts = self.df.isnull().sum()
        return null_counts[null_counts > 0]
    
    def check_duplicates(self):
        # Check for duplicate rows in the DataFrame
        duplicate_rows = self.df[self.df.duplicated()]
        return len(duplicate_rows)
    
    def encode_joint_categorical_columns(self, col1, col2, new_col_name):
        # Combine two categorical columns into a single column and apply one-hot encoding
        self.df[new_col_name] = self.df[col1].fillna('None').astype(str) + '_' + self.df[col2].fillna('None').astype(str)
        dummies = pd.get_dummies(self.df[new_col_name], prefix=new_col_name)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop([col1, col2, new_col_name], axis=1, inplace=True)
        return self.df