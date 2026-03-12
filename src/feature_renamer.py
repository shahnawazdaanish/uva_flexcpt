class FeatureRenamer:
    def __init__(self, df, input_features, output_features, categorical_features=None):
        self.df = df
        self.input_features = input_features
        self.output_features = output_features
        self.categorical_features = categorical_features

    def rename_columns(self):
        rename_mapping = {}
        for i, feature in enumerate(self.input_features):
            rename_mapping[feature] = f'if_{i+1}'
        for j, feature in enumerate(self.output_features):
            rename_mapping[feature] = f'of_{j+1}'
        if self.categorical_features is not None:
            for k, feature in enumerate(self.categorical_features):
                rename_mapping[feature] = f'cf_{k+1}'
        
        renamed_df = self.df.rename(columns=rename_mapping)
        return renamed_df
    
    def get_renamed_column_names(self):
        renamed_input_features = [f'if_{i+1}' for i in range(len(self.input_features))]
        renamed_output_features = [f'of_{j+1}' for j in range(len(self.output_features))]
        renamed_categorical_features = [f'cf_{k+1}' for k in range(len(self.categorical_features))] if self.categorical_features is not None else []
        
        return renamed_input_features, renamed_output_features, renamed_categorical_features
    
    def get_raw_column_name(self, encoded_name):
        renamed_input_features, renamed_output_features, renamed_categorical_features = self.get_renamed_column_names()
        all_renamed_features = renamed_input_features + renamed_output_features + renamed_categorical_features
        if encoded_name in all_renamed_features:
            index = all_renamed_features.index(encoded_name)
            if index < len(renamed_input_features):
                return self.input_features[index]
            elif index < len(renamed_input_features) + len(renamed_output_features):
                return self.output_features[index - len(renamed_input_features)]
            else:
                return self.categorical_features[index - len(renamed_input_features) - len(renamed_output_features)]
        return None
    
    def get_raw_names(self):
        return self.input_features, self.output_features, self.categorical_features