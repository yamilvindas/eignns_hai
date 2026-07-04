"""
    Generated with the help of ChatGPT 5
"""
import os
import re
import argparse
from pathlib import Path
import json
from copy import deepcopy

# -------------------------------------------------
# Global experiment parameters
# -------------------------------------------------
DATASET_NAMES = ["Hug", "SocioPatterns", "Murcia"]

FORECAST_HOR = [1, 3, 7]

MODELS = [
    "GCN",
    "GAT",
    "GraphSage",
    "TGN",
    "STM",
]


DEVICE = "cuda:0"

# -------------------------------------------------
# Model-specific base definitions
# -------------------------------------------------

MODEL_CONFIGS = {
    "GCN": {
                "model_type": "GNN",
                "model_to_use": "GCN",
                "hidden_channels": None,
                "out_channels_node_encoder": None,
                "out_channels": None,
                "num_layers_encoder": None,
                "activation": "ReLU",
                "dropout": None,
    },
    "GAT": {
                
                "model_type": "GNN",
                "model_to_use": "GAT",
                "hidden_channels": None,
                "out_channels_node_encoder": None,
                "out_channels": None,
                "heads": None,
                "dropout": None,
                "num_layers_encoder": None,
                "activation": "ReLU",
                "aggr": "mean",
    },
    "GraphSage": {
                    "model_type": "GNN",
                    "model_to_use": "GraphSAGE",
                    "hidden_channels": None,
                    "out_channels_node_encoder": None,
                    "out_channels": None,
                    "dropout": None,
                    "num_layers_encoder": None,
                    "activation": "ReLU",
    },
    "TGN": {
                "model_type": "GNN",
                "model_to_use": "TGN",
                "time_dim": None,
                "memory_dim": None,
                "hidden_channels": None,
                "dropout": None,
                "out_channels_node_encoder": None,
                "out_channels": None,
                "num_nodes": None,
    },
    "STM": {
                "model_type": "GNN",
                "model_to_use": "STM",
                "skip_connection_spatial": False,
                "time_dim": None,
                "hidden_channels": None,
                "out_channels_node_encoder": None,
                "out_channels": None,
                "dropout": None,
                "temporal_augments": True,
                "spatial_augments": True,
                "temporal_module": "attention",
                "args_temporal": {
                    "hidden_channels": None,
                    "num_layers": None,
                    "memory_capacity": 10,
                    "heads": None,
                    "out_channels": None,
                    "dropout": None
                },
                "spatial_module": "gat",
                "args_spatial": {
                    "hidden_channels": None,
                    "num_layers": None,
                    "dropout": None,
                    "heads": None,
                    "out_channels": None,
                    "act": "ReLU",
                    "aggr": "mean"
                },
    },
}

# -------------------------------------------------
# Base config (model-agnostic)
# -------------------------------------------------

BASE_CONFIG = {
                    "exp_id": None,

                    "dataset_name": None,
                    "subdataset": None,
                    "multiple_hdf5_dataset_filenames": None,
                    "normalize_ds": True,
                    "forecast_horizon": None,

                    "continual_training": False,
                    "reinitialize_model_weights": False,

                    "device": DEVICE,
                    
                    "use_optuna": False,
                    #"metric_optimize": "SE_to_I_AUC",
                    "metric_optimize": "AUCTransitions",
                    "n_trials": 30,
                    "optuna_starting_point_fn": None,

                    "predict_state_transitions": False,
                    "lambda_transitions": None,
                    "lambda_classif_seir_states": None,
                    "constrain_transitions_loss": False,
                    "lambda_constrain_transitions_loss": None,
                    "monotonicity_loss": False,
                    "lambda_monotonicity_loss": None,

                    "epidemio_informed": False,
                    "epidemio_informed_method": None,
                    "init_epi_params_predefined_vals": True,
                    "teacher_forcing_epidemio": False,
                    "ramp_epochs_teacher_forcing_sched": None,
                    "global_seir": False,

                    
                    "nb_repetitions": 1,
                    "optimizer": "Adam",
                    "lr": None,
                    "lr_model_compart_pred": None,
                    "lr_model_epi_params": None,
                    "lr_model_transitions_pred": None,
                    "weight_decay": 1e-07,
                    "batch_size_train": 1,
                    "batch_size_val": 1,
                    "batch_size_test": 1,
                    "n_epochs": 30,
                    "early_stopping": False,
                    "patience": 15,
                    "min_delta": 1e-05,
                    "loss_function": "CE",
                    "loss_function_reg": "MSE",
                    "optuna_search_all_lrs": False,

                    "epochs_step_save_preds": 10
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def build_dataset_paths(dataset_name: str, forecast_horizon: int):
    multiple_hdf5_dataset_filenames = None
    if (dataset_name.lower() == 'hug'):
        raise RuntimeError("The HUG COVID dataset is not available (private dataset).")
    elif (dataset_name.lower() == 'sociopatterns'):
        multiple_hdf5_dataset_filenames = [
                                            "./data/SocioPatterns/socio_patterns_config_1_0.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_1.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_2.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_3.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_4.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_5.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_6.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_7.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_8.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_9.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_10.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_11.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_12.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_13.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_14.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_15.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_16.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_17.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_18.hdf5",
                                            "./data/SocioPatterns/socio_patterns_config_1_19.hdf5"
                                          ]
    elif (dataset_name.lower() == 'murcia'):
        multiple_hdf5_dataset_filenames = [
                                            f"data/murcia/preprocessed/Dataset_0/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_1/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_2/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_3/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_4/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_5/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_6/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_7/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_8/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5",
                                            f"data/murcia/preprocessed/Dataset_9/MurciaGraphData_FullyConnected-False_UniformPatToPatEdges-False_ForecastHorizon-{forecast_horizon}_0_0.hdf5"
                                        ]
    
    return multiple_hdf5_dataset_filenames


def base_experiment_config(dataset_name: str, model_name: str, model_cfg: dict, forecast_horizon: int):
    cfg = deepcopy(BASE_CONFIG)
    cfg["multiple_hdf5_dataset_filenames"] = build_dataset_paths(dataset_name, forecast_horizon)
    cfg["dataset_name"] = dataset_name.lower()
    cfg["model_to_use"] = model_name
    for model_param in model_cfg:
        cfg[model_param] = model_cfg[model_param]
    cfg["exp_id"] = model_name

    return cfg

def save_json(cfg: dict, path: Path):
    with open(path, "w") as f:
        json.dump(cfg, f)

# -------------------------------------------------
# Config variants
# -------------------------------------------------

def make_optuna(cfg: dict, dataset_name: str):
    # Copy the original configuration
    new_cfg = deepcopy(cfg)

    new_cfg["exp_id"] = new_cfg["exp_id"] + f"_OPTUNA_Dataset-{dataset_name}"
    new_cfg["use_optuna"] = True
    new_cfg["optuna_search_all_lrs"] = True
    new_cfg["early_stopping"] = True
    new_cfg["nb_repetitions"] = 1
    new_cfg["hdf5_dataset_filename"] = new_cfg["multiple_hdf5_dataset_filenames"][0]
    del new_cfg["multiple_hdf5_dataset_filenames"]
    if (dataset_name.lower() == 'hug'):
        new_cfg["n_trials"] = 50

    return new_cfg

def make_single_train(cfg: dict, dataset_name: str):
    # Copy the original configuration
    new_cfg = deepcopy(cfg)
    
    new_cfg["exp_id"] = new_cfg["exp_id"] + f"_SimpleTrain_Dataset-{dataset_name}"
    new_cfg["use_optuna"] = False
    new_cfg["optuna_search_all_lrs"] = False
    new_cfg["early_stopping"] = False
    if (dataset_name.lower() == 'hug'):
        new_cfg["nb_repetitions"] = 10
    else:
        new_cfg["nb_repetitions"] = 1
    new_cfg["init_epi_params_predefined_vals"] = True

    return new_cfg

def make_epidemio_informed(cfg: dict, epidemio_informed_method: str):
    # Copy the original configuration
    new_cfg = deepcopy(cfg)

    # Transitions states losses
    new_cfg["constrain_transitions_loss"] = True
    new_cfg["monotonicity_loss"] = True

    # ODEs-based losses
    new_cfg["epidemio_informed"] = True
    new_cfg["epidemio_informed_method"] = epidemio_informed_method
    new_cfg["init_epi_params_predefined_vals"] = True
    new_cfg["teacher_forcing_epidemio"] = True
    new_cfg["global_seir"] = False

    return new_cfg

def update_models_params(model_cfg: dict, dataset_name: str):
    # Copy the original configuration
    new_model_cfg = deepcopy(model_cfg)

    # Number of input channels
    for model_name in new_model_cfg:
        # No need to define them for SocioPatterns
        if (dataset_name.lower() == 'hug'):
            new_model_cfg[model_name]["in_channels"] = {
                                                            # "Patient": 5,
                                                            "Patient": 6,
                                                            "Place": 2
                                                        }
        elif (dataset_name.lower() == 'murcia'):
            new_model_cfg[model_name]["in_channels"] = {
                                                "Patient": 453,
                                                "Place": 8
                                            }
        
        # Number of output channels
        if (dataset_name.lower() == 'hug'):
            new_model_cfg[model_name]["out_channels"] =  3
        elif (dataset_name.lower() == 'murcia'):
            new_model_cfg[model_name]["out_channels"] = 5
        else:
            new_model_cfg[model_name]["out_channels"] = 4

    # For TGN models, we need to define dim_enc_nodes_feaures and num_nodes based on the model
    if (dataset_name.lower() == 'hug'):
        new_model_cfg["TGN"]["dim_enc_nodes_feaures"] =  None
        new_model_cfg["TGN"]["num_nodes"] =  {"Patient": 1000, "Place": 1000}
    elif (dataset_name.lower() == 'murcia'):
        new_model_cfg["TGN"]["dim_enc_nodes_feaures"] =  None
        new_model_cfg["TGN"]["num_nodes"] =  {"Patient": 27448, "Place": 101}
    else:
        # dim_enc_nodes_feaures is not defined for homogeneous GNNs models
        new_model_cfg["TGN"]["num_nodes"] =  500


    return new_model_cfg

# -------------------------------------------------
# Main generation loop
# -------------------------------------------------

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--configs_folder', required=True, help="Folder where the generated config files are going to be stored", type=str)
    ap.add_argument('--predict_state_transitions', help="True if want to directly predict transitions between states too.", action="store_true")
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    configs_folder = args['configs_folder']
    predict_state_transitions = args['predict_state_transitions']


    #==========================================================================#
    # Creating main subfolder to store the generated files
    tmp_i = 0   
    subfolder_name = f'InfectionRiskPred_PredTransitions-{predict_state_transitions}'
    while (os.path.exists(f"{configs_folder}/{subfolder_name}_{tmp_i}")):
        tmp_i += 1
    BASE_OUTPUT_DIR = Path(f"{configs_folder}/{subfolder_name}_{tmp_i}")

    # Update predict_state_transitions in the base configuration if necessary
    if (predict_state_transitions):
        global BASE_CONFIG
        BASE_CONFIG["predict_state_transitions"] = True

    # Create files
    for dataset_name in DATASET_NAMES:
        if (dataset_name.lower() == 'hug'):
            dataset_folder_name = 'HugCovid'
        else:
            dataset_folder_name = dataset_name
        for forecast_hor in FORECAST_HOR:
            for model_name in MODELS:
                if (model_name.lower() == 'gcn' and dataset_name.lower() == 'sociopatterns') or (model_name.lower() != 'gcn'):
                    # Get model folder name
                    if (model_name.lower() == 'graphsage'):
                        model_folder_name = 'GraphSAGE'
                    elif (model_name.lower() == 'stm'):
                        model_folder_name = 'STM-GNN'
                    else:
                        model_folder_name = model_name.upper()
                    # Create directory general directory
                    model_dir = (
                        BASE_OUTPUT_DIR
                        / dataset_folder_name
                        / f"PredHor-{forecast_hor}"
                        / model_folder_name
                    )
                    model_dir.mkdir(parents=True, exist_ok=True)
                    # Create not epidemio-informed directory
                    not_epi_informed_model_dir = (
                        BASE_OUTPUT_DIR
                        / dataset_folder_name
                        / f"PredHor-{forecast_hor}"
                        / model_folder_name
                        / "NotEpidemioInformed"
                    )
                    not_epi_informed_model_dir.mkdir(parents=True, exist_ok=True)
                    # Create epidemio-informed directory
                    epi_informed_model_dir = (
                        BASE_OUTPUT_DIR
                        / dataset_folder_name
                        / f"PredHor-{forecast_hor}"
                        / model_folder_name
                        / "EpidemioInformed"
                    )
                    epi_informed_model_dir.mkdir(parents=True, exist_ok=True)

                    # Threshold dates for train and validation for HUG dataset
                    if (dataset_name.lower() == 'hug'):
                        hug_threshold_train_date = "2020-04-02"
                        hug_threshold_val_date = "2020-04-07"

                    # NOT epidemio informed
                    # OPTUNA
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"NotEpidemioInformed_PredHor-{forecast_hor}_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_optuna(cfg=cfg, dataset_name=dataset_name)
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, not_epi_informed_model_dir / "NotEpidemioInformed_Optuna.json")
                    # Simple Train
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"NotEpidemioInformed_PredHor-{forecast_hor}_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, not_epi_informed_model_dir / "NotEpidemioInformed.json")
                    # Continual Training
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"NotEpidemioInformed_Continual_PredHor-{forecast_hor}_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    cfg["continual_training"] = True
                    cfg["nb_repetitions"] = 1
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, not_epi_informed_model_dir / "NotEpidemioInformedContinual.json")

                    # EPIDEMIO-informed
                    # AutoDiff
                    # OPTUNA
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_PredHor-{forecast_hor}_AutoDiff_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_optuna(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="autodiff_wrt_time_odes_residuals")
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "AutoDiffWrtTimeReg_Optuna.json")
                    # Simple Train
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_PredHor-{forecast_hor}_AutoDiff_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="autodiff_wrt_time_odes_residuals")
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "AutoDiffWrtTimeReg.json")
                    # Continual Training
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_Continual_PredHor-{forecast_hor}_AutoDiff_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="autodiff_wrt_time_odes_residuals")
                    cfg["continual_training"] = True
                    cfg["nb_repetitions"] = 1
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "AutoDiffWrtTimeRegContinual.json")

                    # ODE-NSP
                    # OPTUNA
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_PredHor-{forecast_hor}_ODE-NSP_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_optuna(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="odes_next_state_residuals")
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "ODEsNextStatePredRed_Optuna.json")
                    # Simple Train
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_PredHor-{forecast_hor}_ODE-NSP_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="odes_next_state_residuals")
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "ODEsNextStatePredRed.json")
                    # Continual Training
                    model_cfg = update_models_params(model_cfg=MODEL_CONFIGS, dataset_name=dataset_name)
                    cfg = base_experiment_config(dataset_name=dataset_name, model_name=model_name, model_cfg=model_cfg[model_name], forecast_horizon=forecast_hor)
                    cfg["exp_id"] = f"EpidemioInformed_Continual_PredHor-{forecast_hor}_ODE-NSP_" + cfg["exp_id"]
                    cfg["forecast_horizon"] = forecast_hor
                    cfg = make_single_train(cfg=cfg, dataset_name=dataset_name)
                    cfg = make_epidemio_informed(cfg=cfg, epidemio_informed_method="odes_next_state_residuals")
                    cfg["continual_training"] = True
                    cfg["nb_repetitions"] = 1
                    if (dataset_name.lower() == 'hug'):
                        cfg["threshold_train_date"] = hug_threshold_train_date
                        #cfg["threshold_val_date"] = hug_threshold_val_date
                    save_json(cfg, epi_informed_model_dir / "ODEsNextStatePredRedContinual.json")


                    print(f"Generated configs in {model_dir}")


if __name__ == "__main__":
    main()
