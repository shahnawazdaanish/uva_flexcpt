import torch
import gpytorch
from src.models.lcm_multitaskexactgp import LCM_MultitaskExactGP

class LoadedModelsReinstator:
    def __init__(self, loaded_model_groups):
        self.loaded_model_groups = loaded_model_groups

    def get_reinstated_models(self, X_train_scaled, Y_train_scaled, MAIN_INPUT_NAMES, ALL_OUTPUT_NAMES):
        restored_models = {}
        for key, group in self.loaded_model_groups.items():
            state_dict_model = group['model_state_dict_cpu']
            state_dict_lik = group['likelihood_state_dict_cpu']

            
            num_tasks = len(group['task_cols'])
            num_features = len(group['input_cols'])
            num_latents = group['num_latents']
            
            # 1. Create a single row of dummy data to satisfy the ExactGP constructor
            dummy_train_x = torch.zeros((1, num_features))
            dummy_train_y = torch.zeros((1, num_tasks))
            
            # 2. Initialize likelihood and model
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            
            # (Replace MyMultitaskGPModel with your actual LCM model class name)
            model = LCM_MultitaskExactGP(
                train_x=dummy_train_x, 
                train_y=dummy_train_y, 
                likelihood=likelihood, 
                num_tasks=num_tasks, 
                num_latents=num_latents,
                input_dim = num_features
            )
            
            # 3. Load states with strict=False
            # This tells PyTorch to load the lengthscales/variances and ignore 
            # the fact that train_inputs/targets are missing from the dictionary.
            model.load_state_dict(state_dict_model, strict=False)
            likelihood.load_state_dict(state_dict_lik, strict=False)

            # The planner evaluates paths in float64. The restored model weights may come from a
            # float32 bundle, so force a single dtype across the whole loaded GP for consistency.
            model = model.to(torch.float64)
            likelihood = likelihood.to(torch.float64)

            task_indices = [ALL_OUTPUT_NAMES.index(col) for col in group['task_cols']]
            input_indices = [MAIN_INPUT_NAMES.index(col) for col in group['input_cols']]
    
            # Extract the Y data just for this chunk's tasks
            Y_chunk_train = Y_train_scaled[:, task_indices]
            X_train_filtered = X_train_scaled[:, input_indices]

            X_tensor = torch.as_tensor(X_train_filtered, dtype=torch.float32)
            Y_tensor = torch.as_tensor(Y_chunk_train, dtype=torch.float32)

            print(X_tensor.shape, Y_tensor.shape)  # Debug: print shapes to verify
            
            # Inject the true training data back into the GP!
            model.set_train_data(inputs=X_tensor, targets=Y_tensor, strict=False)
            
            model.eval()
            likelihood.eval()
            
            restored_models[key] = {
                'model': model,
                'likelihood': likelihood,
                'task_cols': group['task_cols'],
                'input_cols': group['input_cols'],
                'score_mean_r2': group['score_mean_r2']
            }
            
        return restored_models













        # restored_models = {}
        # for key, group in self.loaded_model_groups.items():
        #     state_dict_model = group['model_state_dict_cpu']
        #     state_dict_lik = group['likelihood_state_dict_cpu']
            
        #     num_tasks = len(group['task_cols'])
        #     num_latents = group['num_latents']
            
        #     # PRO TRICK: GPyTorch ExactGPs require training data to initialize, 
        #     # and load_state_dict requires the dummy data to match the original training data shape.
        #     # We can extract the exact original shapes directly from the saved state dict!

        #     print(state_dict_model.keys())  # Debug: print keys to understand the structure
        #     if 'train_inputs.0' not in state_dict_model or 'train_targets' not in state_dict_model:
        #         raise KeyError("Expected keys 'train_inputs.0' and 'train_targets' not found in the model state dict.")
            
        #     original_x_shape = state_dict_model['train_inputs.0'].shape
        #     original_y_shape = state_dict_model['train_targets'].shape
            
        #     dummy_train_x = torch.zeros(original_x_shape)
        #     dummy_train_y = torch.zeros(original_y_shape)
            
        #     # Initialize likelihood and model
        #     likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            
        #     # !!! IMPORTANT: Replace MyMultitaskGPModel with your actual class name !!!
        #     model = LCM_MultitaskExactGP(
        #         train_x=dummy_train_x, 
        #         train_y=dummy_train_y, 
        #         likelihood=likelihood, 
        #         num_tasks=num_tasks, 
        #         input_dim=len(group['input_cols']),
        #         num_latents=num_latents
        #     )
            
        #     # Load states
        #     model.load_state_dict(state_dict_model, strict=True)
        #     likelihood.load_state_dict(state_dict_lik, strict=True)
            
        #     # Set to evaluation mode for sOED predictions
        #     model.eval()
        #     likelihood.eval()
            
        #     restored_models[key] = {
        #         'model': model,
        #         'likelihood': likelihood,
        #         'task_cols': group['task_cols'],
        #         'input_cols': group['input_cols'],
        #         'score_mean_r2': group['score_mean_r2']
        #     }
        # return restored_models