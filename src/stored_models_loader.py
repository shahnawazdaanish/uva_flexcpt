import torch
import gpytorch
from src.soed.dynamic_gp import DynamicGP

class StoredModelsLoader:
    def __init__(self, bundle_path: str):
        self.bundle_path = bundle_path
        self.payload = None

    def stored_object_sample(self):
        return {
            'bundle_version': 0,
            'bundle_type': '',
            'created_from': '',
            'model_groups': {
                '': {self.model_object_sample()},
            },
            'input_features': [],
            'scalers': {
                'scaler_x': [],
                'scaler_y': [],
                'scaler_x_ch4': [],
                'scaler_y_ch4': [],
                'X_scaler_s': [],
                'Y_scaler_r': [],
            },
        }
    
    def model_object_sample(self):
        return {
            'label': "",
            'fold': 0,
            'score_mean_r2': 0.0,
            'input_cols': [],
            'task_cols': [],
            'num_latents': 0,
            'model_state_dict_cpu': None,
            'likelihood_state_dict_cpu': None,
            'task_metrics': {},
            'fold_summaries': [],
        }

    def load_models(self, feature_names=None):
        print(f"Loading model from {self.bundle_path}")        
        self.payload = torch.load(self.bundle_path, map_location="cpu", weights_only=False)
        print(f"Model loaded successfully with {len(self.payload.get('model_groups', {}))} model groups "
              f"and scalers: {list(self.payload.get('scalers', {}).keys())}")
        return self

    def load_model_from_groups(self, group_key):
        loaded_model_groups = self.payload.get("model_groups", {})
        if not loaded_model_groups:
            raise ValueError("Model groups have not been loaded yet. Call load_models() first.")
        if group_key not in loaded_model_groups:
            raise KeyError(f"Group key '{group_key}' not found in loaded model groups.")
        return loaded_model_groups[group_key]

    def get_model_groups(self):
        loaded_model_groups = self.payload.get("model_groups", {})
        if not loaded_model_groups:
            raise ValueError("Model groups have not been loaded yet. Call load_models() first.")
        return loaded_model_groups

    def get_model_states(self, group_key):
        model_group = self.load_model_from_groups(group_key)
        model_state_dicts = [model_info['model_state_dict_cpu'] for model_info in model_group.values()]
        likelihood_state_dicts = [model_info['likelihood_state_dict_cpu'] for model_info in model_group.values()]
        task_columns = [model_info['task_cols'] for model_info in model_group.values()]
        return model_state_dicts, likelihood_state_dicts, task_columns

    def get_input_features(self):
        return self.payload.get("input_features", [])

    def load_scalers_state(self):
        loaded_scaler_states = self.payload.get("scalers", {})
        if not loaded_scaler_states:
            raise ValueError("Scalers have not been loaded yet. Call load_models() first.")
        return loaded_scaler_states
    