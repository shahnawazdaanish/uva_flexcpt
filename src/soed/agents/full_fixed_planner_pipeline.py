import torch
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from src.soed.agents.constrained_multistep_mimo_agent_fixed import ConstrainedMultiStepMIMOAgentFixed


def build_scaled_bounds(df, feature_names, scaler_x=None):
    """Create scaled bounds for optimization from feature columns.

    BoTorch expects bounds in the shape [2, d], where the first row is the lower
    bounds and the second row is the upper bounds for each feature.
    """
    unscaled_bounds = []
    for feat in feature_names:
        col = df[feat]
        unscaled_bounds.append([float(col.min()), float(col.max())])

    if scaler_x is None:
        scaler_x = StandardScaler()
        scaler_x.fit(df[feature_names].to_numpy())

    # Build a 2 x d matrix: each column corresponds to one feature.
    unscaled_bounds_array = np.array(unscaled_bounds, dtype=float).T
    unscaled_bounds_df = pd.DataFrame(unscaled_bounds_array, columns=feature_names)
    scaled_bounds_df = pd.DataFrame(scaler_x.transform(unscaled_bounds_df), columns=feature_names)
    scaled_bounds = torch.tensor(scaled_bounds_df.to_numpy(), dtype=torch.float64)

    if scaled_bounds.shape[0] != 2 or scaled_bounds.shape[1] != len(feature_names):
        raise ValueError(
            f"Bounds must have shape [2, {len(feature_names)}], got {tuple(scaled_bounds.shape)}"
        )

    return scaled_bounds, scaler_x


def prepare_inputs_outputs(df, input_features, output_features):
    X = df[input_features].to_numpy(dtype=float)
    Y = df[output_features].to_numpy(dtype=float)
    return X, Y


def run_fixed_planner_pipeline(
    df,
    input_features,
    output_features,
    current_location,
    q_steps=3,
    num_scenarios=256,
    load_limit=30.0,
    boost_slope=0.0922,
    boost_intercept=0.8378,
    boost_band=0.5,
    min_load=3.0,
    ambient_pressure=1.0,
    tc_boost_limit=3.8,
):
    """Full pipeline: preprocess -> fit GP -> run planner -> return planned path."""
    X, Y = prepare_inputs_outputs(df, input_features, output_features)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = Y

    bounds, scaler_x = build_scaled_bounds(df, input_features, scaler_x=scaler_x)

    agent = ConstrainedMultiStepMIMOAgentFixed(
        bounds=bounds,
        feature_names=input_features,
        scaler_x=scaler_x,
        mass1_name="Mass1",
        mass2_name="Mass2",
        boost_name="Boost pressure",
        load_limit=load_limit,
        boost_slope=boost_slope,
        boost_intercept=boost_intercept,
        boost_band=boost_band,
        min_load=min_load,
        ambient_pressure=ambient_pressure,
        tc_boost_limit=tc_boost_limit,
    )

    agent.fit_data(torch.tensor(X_scaled, dtype=torch.float64), torch.tensor(Y_scaled, dtype=torch.float64))

    path = agent.plan_multistep_batch(
        current_location=torch.tensor(current_location, dtype=torch.float64),
        q_steps=q_steps,
        num_scenarios=num_scenarios,
        w_dist=1.0,
        enforce_feasible_sampling=False,
        feasible_margin_weight=25.0,
    )

    return {
        "agent": agent,
        "bounds": bounds,
        "scaler_x": scaler_x,
        "path": path,
        "path_original_units": agent._to_original_units(path),
    }


if __name__ == "__main__":
    # Example end-to-end usage.
    # Replace this with your real dataframe and feature names.
    sample_df = pd.DataFrame(
        {
            "Mass1": [10.0, 12.0, 14.0, 18.0, 20.0],
            "Mass2": [2.0, 2.5, 3.0, 3.5, 4.0],
            "Boost pressure": [1.2, 1.4, 1.6, 1.8, 2.0],
            "IVO": [350.0, 360.0, 370.0, 380.0, 390.0],
            "IVC": [510.0, 515.0, 520.0, 525.0, 530.0],
            "EVO": [130.0, 135.0, 140.0, 145.0, 150.0],
            "EVC": [300.0, 310.0, 320.0, 330.0, 340.0],
            "OutputA": [0.5, 0.6, 0.7, 0.8, 0.9],
            "OutputB": [2.0, 2.4, 2.8, 3.2, 3.6],
        }
    )

    input_features = [
        "Mass1",
        "Mass2",
        "Boost pressure",
        "IVO",
        "IVC",
        "EVO",
        "EVC",
    ]
    output_features = ["OutputA", "OutputB"]

    result = run_fixed_planner_pipeline(
        df=sample_df,
        input_features=input_features,
        output_features=output_features,
        current_location=[10.0, 2.5, 1.5, 360.0, 520.0, 140.0, 320.0],
        q_steps=3,
        num_scenarios=256,
    )

    print("Planned path (scaled):")
    print(result["path"])
    print("\nPlanned path (original units):")
    print(result["path_original_units"])
