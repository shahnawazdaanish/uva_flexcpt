import pandas as pd

class DataFormatter:
    def __init__(self, input_columns, output_columns, categorical_columns=[]):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.categorical_columns = categorical_columns

    def filter_columns(self, df):
        # Filter the DataFrame to include only the specified input and output columns
        df_filtered = df[self.input_columns + self.categorical_columns + self.output_columns]
        return df_filtered
    
    def filter_input_columns(self, df):
        # Filter the DataFrame to include only the specified input columns
        df_filtered = df[self.input_columns]
        return df_filtered
    
    def filter_output_columns(self, df):
        # Filter the DataFrame to include only the specified output columns
        df_filtered = df[self.output_columns]
        return df_filtered
    
    def rename_columns(self, df):
        # Rename the index to 'idx' and input/output columns to 'feat_2', 'feat_3', ..., 'feat_n'
        df.index.name = 'idx'
        
        input_feature_columns = [f'if_{i+1}' for i in range(len(self.input_columns))]
        output_feature_columns = [f'of_{i+1}' for i in range(len(self.output_columns))]
        
        if self.categorical_columns:
            # Rename categorical columns if they are specified
            cat_feature_columns = [f'cat_{i+1}' for i in range(len(self.categorical_columns))]
            df.columns = input_feature_columns + cat_feature_columns + output_feature_columns
        else:
            df.columns = input_feature_columns + output_feature_columns
        return df
    
    def rename_all_columns(self, df, ignore_unnamed=False):
        # Rename all columns which does not have unnamed in their names and merge with unnamed columns
        if ignore_unnamed:
            unknown_columns = df.loc[:, df.columns.astype(str).str.contains('^Unnamed')]
            df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
            df.columns = [f'feat_{i+1}' for i in range(len(df.columns))]
            df = pd.concat([df, unknown_columns], axis=1)
        else:
            df.columns = [f'feat_{i+1}' for i in range(len(df.columns))]
        return df
    
    def encode_categorical_columns(self, df, column, encoding_method='one-hot'):
        if encoding_method == 'one-hot':
            # if index 5729 is present, add 0.5 to the category codes or add it from 5730
            if 5729 in df.index:
                df.loc[5729:, column] += 0.5
            else:
                df.loc[5730:, column] += 0.5
            
            # Convert categorical column to category type and get codes
            # df[column + '_cat'] = df[column].astype('category').cat.codes

            # # Create dummy variables (one-hot encoding)
            # dummies = pd.get_dummies(df[column + '_cat'], prefix='cat')
            # dummies.columns = [f'{column}_{i}' for i in range(dummies.shape[1])]

            dummies = pd.get_dummies(df[column].astype('category').cat.codes, prefix='cat_1', drop_first=True)

            # drop the original categorical column
            df = df.drop(column, axis=1)
            
            # Concatenate with original DataFrame
            df = pd.concat([df, dummies], axis=1)
        return df