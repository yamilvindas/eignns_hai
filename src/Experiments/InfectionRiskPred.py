#!/usr/bin/env python3
"""
    Class for an experiment training a Deep Learning model
    to fo infection risk prediction
"""
import os
import re
import pickle
import yaml
import shutil
import json
import argparse
from datetime import datetime
from tqdm import tqdm

from collections import Counter

from copy import deepcopy

import h5py

import random

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from src.Utils.tools import one_hot_encoding_np
from src.Models.modules import MulticlassClassifier,\
                               MultistepClassifier,\
                               PerIndEpidemioParamsRegressor,\
                               ODENextStatePredictor,\
                               SEIRPredictorDeeper,\
                               SEIRPredictor,\
                               init_weights


from src.Experiments.GenericExperiment import GenericExperiment
from src.DataManipulation.DynamicHeteroGraphTemporalSignal import DynamicHeteroGraphTemporalSignal
from src.DataManipulation.Murcia.data_utils import get_statistics_train_dataset_murcia
from src.DataManipulation.hug_covid.data_utils import preprocessHUGCovidDataset


# Mapping of the states
SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
INV_MAPPING = {v: k for k, v in MAPPING.items()}

# Variables for Murcia data splits
# Train
INIT_TRAIN_STEP = 0
LAST_TRAIN_STEP = 1000
# Val
INIT_VAL_STEP = LAST_TRAIN_STEP
LAST_VAL_STEP = INIT_VAL_STEP + 400
# Test
INIT_TEST_STEP = LAST_VAL_STEP
LAST_TEST_STEP = -1


def compute_transition_metrics(true_states, pred_probs, current_states):
    """
    true_states: [N, W] - Ground truth
    pred_probs: [N, W, C] - Model output probabilities
    current_states: [N] - The state index at t=0
    """
    pred_states = pred_probs.argmax(axis=-1) # [N, W]
    
    # 1. Identify "Stable" vs "Transition" indices
    # A patient is a 'Transition' case if their state at any point in W differs from t0
    is_transition = (true_states != current_states[:, np.newaxis]).any(axis=1)
    is_stable = ~is_transition

    # 2. Compute "Transition Recall" 
    # This is your most important metric.
    y_true_trans = true_states[is_transition].flatten()
    y_pred_trans = pred_states[is_transition].flatten()
    
    # 3. Compute "Stability Precision"
    # Does the model correctly predict 'no change' for those who didn't move?
    y_true_stable = true_states[is_stable].flatten()
    y_pred_stable = pred_states[is_stable].flatten()

    # Output
    try:
        transition_report = classification_report(y_true_trans, y_pred_trans, zero_division=0),
    except:
        transition_report = None
    try:
        stability_report = classification_report(y_true_stable, y_pred_stable, zero_division=0)
    except:
        stability_report = None
    
    out = {
            "transition_report": transition_report,
            "stability_report": stability_report
        }

    return out

def check_prediction_bias(pred_probs, current_states):
    # How often does the model predict a change?
    preds = pred_probs.argmax(axis=-1)
    changes_predicted = (preds != np.expand_dims(current_states, 1)).mean()
    
    print(f"Percentage of nodes predicted to change state: {changes_predicted:.2%}")
    # If this is > 50% in a hospital setting, your weights are too aggressive.

def safe_inverse_softplus(y, threshold=20):
    # Ensure y is a tensor for torch operations
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, dtype=torch.float32)
    
    # Stable inverse: for large y, softplus(y) is approximately y
    # For smaller y, we use the log(exp(y) - 1) formula
    return torch.where(y > threshold, y, torch.log(torch.expm1(y)))

class TemporalTransitionAwareLoss(nn.Module):
    def __init__(self, weights_matrix, temporal_decay=True, ignore_index=-1):
        """
            Generated with the help of Gemini
            weights_matrix: [C, C] tensor where [i, j] is weight from state i to state j
        """
        super().__init__()
        self.register_buffer('weights', weights_matrix)
        self.ignore_index = ignore_index
        self.temporal_decay = temporal_decay 

    def forward(self, logits, targets, initial_state):
        """
        logits: [N, W, C]
        targets: [N, W]
        initial_state: [N] (The state at t=0)
        """
        # Batch size, forecast window, number of states
        N, W, C = logits.shape
        
        # Flatten for calculation
        # logits_flat: [N*W, C], targets_flat: [N*W]
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        # Repeat initial state W times to compare t=0 to every step in window
        # initial_state_expanded: [N*W]
        initial_state_expanded = initial_state.repeat_interleave(W)
        
        # Calculate standard CE loss per sample (no reduction yet)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, 
                                  ignore_index=self.ignore_index, 
                                  reduction='none')
        
        # Create mask for valid labels
        mask = (targets_flat != self.ignore_index)
        
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # Index into the 2D weight matrix using [Initial_State, Target_State]
        # This gives us the specific weight for the transition that occurred
        sample_weights = self.weights[initial_state_expanded[mask], targets_flat[mask]]

        # Create a decay vector [W] e.g., [1.0, 0.9, 0.81, ...]
        if (self.temporal_decay):
            gamma = 0.9
            decay = torch.pow(gamma, torch.arange(W, device=logits.device))
            # Expand to match [N*W]
            decay_expanded = decay.repeat(N)
            # Apply both Transition Weights and Temporal Decay
            final_weights = sample_weights * decay_expanded[mask]
        else:
            final_weights = sample_weights
        
        # Apply weights and average
        weighted_loss = (ce_loss[mask] * final_weights).mean()
        
        return weighted_loss

# Abstract class
class InfectionRiskPred(GenericExperiment):
    def __init__(self, parameters_exp, use_debug=False):
        """
            Class for infection risk prediction DL experiment.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment.
            use_debug: bool
                Activate to access some specific breakpoints in the code for debug.
        """
        # Parent constructor
        super().__init__(parameters_exp, use_debug)

        # Other attributes      
        # Epidemio-informed model ?
        if ('epidemio_informed' not in self.parameters_exp):
            self.parameters_exp['epidemio_informed'] = True
        if (self.parameters_exp['epidemio_informed']):
            if ('epidemio_informed_method' not in self.parameters_exp):
                self.parameters_exp['epidemio_informed_method'] = "ODEs_Next_State_Residuals"
                #self.parameters_exp['epidemio_informed_method'] = "Autodiff_WRT_time_ODEs_Residuals"
            if ('init_epi_params_predefined_vals' not in self.parameters_exp):
                self.parameters_exp['init_epi_params_predefined_vals'] = True
                #self.parameters_exp['init_epi_params_predefined_vals'] = False
            self.init_epi_params_predefined_vals = self.parameters_exp['init_epi_params_predefined_vals']

        # Use teacher forcing for epidemio_informed loss
        if ('teacher_forcing_epidemio' not in self.parameters_exp):
            self.parameters_exp['teacher_forcing_epidemio'] = False
        if (self.parameters_exp['teacher_forcing_epidemio']):
            if ("ramp_epochs_teacher_forcing_sched" not in self.parameters_exp):
                self.parameters_exp['ramp_epochs_teacher_forcing_sched'] = 10

            # Global SEIR (instead of per individual)
            if ('global_seir' not in self.parameters_exp):
                self.parameters_exp['global_seir'] = False

        # Matrices for state transitions constraints loss
        # Sir (S=0, I=1, R=2)
        self.T_SIR = torch.tensor([
            [1, 1, 0], # From S: stays S or moves to E
            [0, 1, 1], # From I: stays I or moves to R
            [0, 0, 1], # From R: stays R (absorbing)
        ], dtype=torch.float32)
        # SEIR (S=0, E=1, I=2, R=3)
        self.T_SEIR = torch.tensor([
            [1, 1, 0, 0], # From S: stays S or moves to E
            [0, 1, 1, 0], # From E: stays E or moves to I
            [0, 0, 1, 1], # From I: stays I or moves to R
            [0, 0, 0, 1]  # From R: stays R (absorbing)
        ], dtype=torch.float32)

        # SEIRD-NS (S=0, E=1, I=2, R=3, D=4, NS=5)
        self.T_SEIRD_NS = torch.tensor([
            [1, 1, 0, 0, 0, 1], # From S: stays S, moves to E, or becomes NS
            [0, 1, 1, 0, 0, 1], # From E: stays E, moves to I, or becomes NS
            [0, 0, 1, 1, 1, 1], # From I: stays I, moves to R, D, or NS
            [0, 0, 0, 1, 0, 0], # From R: absorbing
            [0, 0, 0, 0, 1, 0], # From D: absorbing
            [0, 0, 0, 0, 0, 1]  # From NS: absorbing
        ], dtype=torch.float32)


        # Remove epidemiology state feature
        self.remove_epi_state = False

        # Number of states transitions classes
        self.n_transitions_types = None


    def preprocessSocioPatternsDataset(self, prefix_paths="", forecast_horizon=1):
        """
            Pre-process Socio Patterns datasets (SFHH and Hospital Ward)
            It creates an dictionary attribute. This dictionary containing 
            the graphs per data split. The keys are the data splits and the
            values are dicts with the following keys (the values are lists):
            nodes_ids, edges_indices, edges_weights, timestamps, features,
            targets, states_history, infection_rates, incubation_durations,
            recovery_durations.
        """
        # Creating the output variable
        self.data = {}
        for data_split_type in ['train', 'val', 'test']:
            self.data[data_split_type] = {
                                        "nodes_ids": [],
                                        "edges_indices": [],
                                        "edges_weights": [],
                                        "timestamps": [],
                                        "features": [],
                                        "targets": [],
                                        "transitions_targets": [],
                                        "states_history": [],
                                        "infection_rates": [],
                                        "incubation_durations": [],
                                        "recovery_durations": []
                                    }
        for data_split_type in self.h5_file:
            # File
            data_path = prefix_paths + self.h5_file[data_split_type].attrs['Path']
            with open(data_path, mode='rb') as pf:
                data = pickle.load(pf)
            
            # Separating variables
            nodes_ids,\
            edges_indices,\
            edges_weights,\
            timestamps,\
            features,\
            targets,\
            states_history,\
            infection_rates,\
            incubation_durations,\
            recovery_durations = data

            # Removing useless dimension targets
            targets = [t.squeeze() for t in targets]
            targets_np = np.array(targets) # Shape: [Total_Days, Num_Nodes]

            # Creating the final features of the nodes and the targets
            #for t in range(len(features)-1): # OLD VERSION BEFORE 22/09/2025 WHEN ONLY A FORECAST HORIZON OF 1 WAS POSSIBLE
            for t in range(len(features)-forecast_horizon):
                # One hot encoding the SEIR states
                curr_states = targets[t]
                SEIR_states_one_hot = one_hot_encoding_np(targets[t], num_classes=self.parameters_exp["num_states"])

                # Adding it to the final features
                x_t = np.concatenate([features[t], SEIR_states_one_hot], axis=1)
                self.data[data_split_type]["features"].append(x_t)

                # Final targets
                n_individuals = len(targets[t])
                if (forecast_horizon == 1):
                    self.data[data_split_type]["targets"].append(targets[t + 1])
                    window_targets = targets_np[t+1 : t+2, :] # [1, N]
                else: # In this case, we predict more than one day AFTER the current day
                    seir_targets = np.array(targets)[t+1:t+forecast_horizon+1, :].T
                    self.data[data_split_type]["targets"].append(seir_targets)
                    window_targets = targets_np[t+1 : t+forecast_horizon+1, :] # [W, N]

                # Generate Transition Targets (Anti-Identity Mapping Logic)
                # window_targets is [W, N]. We iterate over nodes.
                n_nodes = curr_states.shape[0]
                trans_labels = np.zeros(n_nodes, dtype=int)
                for i in range(n_nodes):
                    s_now = curr_states[i]
                    s_future_seq = window_targets[:, i] # Sequence of states for node i
                    
                    # Get unique states in window to find the trajectory
                    trajectory = [s_now]
                    for s_next in s_future_seq:
                        if (s_next != trajectory[-1]):
                            trajectory.append(s_next)
                    
                    # Define Specific "Correct" Transition Categories
                    # DETAILED TRANSITIONS
                    # Logic: [S=0, E=1, I=2, R=3] (Adjust indices based on your num_states mapping)
                    # if (len(trajectory) == 1):
                    #     trans_labels[i] = 0  # Stay
                    # elif (trajectory == [0, 1] or trajectory == [0, 2]):
                    #     trans_labels[i] = 1  # S -> E or S -> I (Infection)
                    # elif (trajectory == [1, 2]):
                    #     trans_labels[i] = 2  # E -> I (Progression)
                    # elif (trajectory == [2, 3]):
                    #     trans_labels[i] = 3  # I -> R (Recovery)
                    # elif (len(trajectory) > 2):
                    #     trans_labels[i] = 4  # Rapid Cycle (e.g., S -> E -> I)
                    # else:
                    #     trans_labels[i] = 5  # Other (R -> S, etc.)
                    # SIMPLIFIED VERSION
                    if (len(trajectory) == 1):
                        trans_labels[i] = 0  # Stay
                    elif (trajectory == [0, 1]) or (trajectory == [0, 2]) or (trajectory == [1, 2]) or (len(trajectory) > 2):
                        trans_labels[i] = 1  # S -> E or S -> I (Infection) or E -> I (Progression) or Rapid Cycle (e.g., S -> E -> I)
                    elif (trajectory == [2, 3]):
                        trans_labels[i] = 2  # I -> R (Recovery)
                    else:
                        trans_labels[i] = 3  # Other (R -> S, etc.)
                self.data[data_split_type]["transitions_targets"].append(trans_labels)
                

                # Other variables
                self.data[data_split_type]["nodes_ids"].append(nodes_ids[t])
                self.data[data_split_type]["edges_indices"].append(edges_indices[t])
                self.data[data_split_type]["edges_weights"].append(edges_weights[t])
                self.data[data_split_type]["timestamps"].append(timestamps[t])
                self.data[data_split_type]["states_history"].append(states_history[t])
                self.data[data_split_type]["infection_rates"].append(infection_rates[t])
                self.data[data_split_type]["incubation_durations"].append(incubation_durations[t])
                self.data[data_split_type]["recovery_durations"].append(recovery_durations[t])
                
        # Max number of nodes for training DS
        self.max_n_nodes_train = max([self.data["train"]["nodes_ids"][i].shape[0] for i in range(len(self.data["train"]["nodes_ids"]))])
        self.max_n_nodes_test = max([self.data["test"]["nodes_ids"][i].shape[0] for i in range(len(self.data["test"]["nodes_ids"]))])

        # Normalizing the timestamps and edges weights for more stable training
        if (self.parameters_exp["normalize_ds"]):
            self.normalizeDataset()

    def preprocessMurciaDataset(self, prefix_paths=""):
        """
            Pre-process a Murcia datasets (simulated)
            It creates an dictionary attribute. This dictionary containing 
            the following keys: EdgeIndexDicts, EdgeFeaturesDicts, 
            NodeFeaturesDicts, NodeTargetsDict, NodeTimestampsDicts,
            NodesIdsDicts. Each value is a list containing the data for
            each time step in the simulation.
            It also creates a non-normalized version of the train, val, 
            and test datasets.
        """           
        # Get the lists of dicts to create the Murcia Heterogeneous Dataset
        n_steps = len(self.h5_file["EdgeFeaturesDicts"])
        self.data = {
                        "EdgeIndexDicts": [None for _ in range(n_steps)],
                        "EdgeFeaturesDicts": [None for _ in range(n_steps)],
                        "NodeFeaturesDicts": [None for _ in range(n_steps)],
                        "NodeTargetsDict": [None for _ in range(n_steps)],
                        "NodeTargetsTransitionsDict": [None for _ in range(n_steps)],
                        "NodeTimestampsDicts": [None for _ in range(n_steps)],
                        "NodeInfRiskTargetsDict": [None for _ in range(n_steps)],
                        "NodesIdsDicts": [None for _ in range(n_steps)]
                    }
        for main_group in self.data:
            for str_step_ID in tqdm(self.h5_file[main_group]):
                step_ID = int(str_step_ID)
                keys = list(self.h5_file[main_group][str_step_ID])
                self.data[main_group][step_ID] = {}
                for key in keys:
                    if (len(key.split('-')) == 2):
                        new_key = tuple(key.split('-'))
                    else:
                        new_key = key
                    self.data[main_group][step_ID][new_key] = self.h5_file[main_group][str_step_ID][key][:]

        # Close HDF5 file
        self.h5_file.close()

        # Separating into training and testing data
        # Train
        train_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_TRAIN_STEP:LAST_TRAIN_STEP]}
        self.train_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=self.data["EdgeIndexDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            feature_dicts=self.data["NodeFeaturesDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            target_dicts=self.data["NodeTargetsDict"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            timestamps=self.data["NodeTimestampsDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            ids=self.data["NodesIdsDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            **train_other_attributes
                                                        )

        # Val
        val_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_VAL_STEP:LAST_VAL_STEP]}
        self.val_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=self.data["EdgeIndexDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            feature_dicts=self.data["NodeFeaturesDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            target_dicts=self.data["NodeTargetsDict"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            timestamps=self.data["NodeTimestampsDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            ids=self.data["NodesIdsDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            **val_other_attributes
                                                        )

        # Test
        test_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_TEST_STEP:LAST_TEST_STEP]}
        self.test_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=self.data["EdgeIndexDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            feature_dicts=self.data["NodeFeaturesDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            target_dicts=self.data["NodeTargetsDict"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            timestamps=self.data["NodeTimestampsDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            ids=self.data["NodesIdsDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                            **test_other_attributes
                                                        )
        
        # Normalizing the timestamps and edges weights for more stable training
        if (self.parameters_exp["normalize_ds"]):
            self.normalizeDataset()

        # Split percentages
        n_total_samples = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"\n\n=========>DATASET CREATED: {100*len(self.train_ds)/n_total_samples}% for training, {100*len(self.val_ds)/n_total_samples}% for validation, {100*len(self.test_ds)/n_total_samples}% for testing\n\n")


    def normalizeDataset(self):
        """
            Normalize the dataset by substracting the mean and dividing by
            the std
        """
        #======================================================================#
        #======================================================================#
        if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            # Edge weights
            train_edge_weights = np.concatenate(self.data["train"]["edges_weights"])
            mean_train_edge_weights = np.mean(train_edge_weights)
            std_train_edge_weights = np.std(train_edge_weights)
            min_train_edge_weights = np.min(train_edge_weights)
            max_train_edge_weights = np.max(train_edge_weights)
            KAPPA = 1 # Could be better chosen or even learnED
            # Train
            self.data["train"]["edges_weights"] = [1 - np.exp(- KAPPA * np.log( 1 + ( (edges_weights_snap-min_train_edge_weights)/(max_train_edge_weights - min_train_edge_weights) ) ) ) for edges_weights_snap in self.data["train"]["edges_weights"]]
            # Val
            self.data["val"]["edges_weights"] = [1 - np.exp(- KAPPA * np.log( 1 + ( (edges_weights_snap-min_train_edge_weights)/(max_train_edge_weights - min_train_edge_weights) ) ) ) for edges_weights_snap in self.data["val"]["edges_weights"]]
            # Test
            self.data["test"]["edges_weights"] = [1 - np.exp(- KAPPA * np.log( 1 + ( (edges_weights_snap-min_train_edge_weights)/(max_train_edge_weights - min_train_edge_weights) ) ) ) for edges_weights_snap in self.data["test"]["edges_weights"]]

            # Timestamps
            train_timestamps = np.concatenate(self.data["train"]["timestamps"])
            #timesteps_scaler = MinMaxScaler().fit(train_timestamps.reshape(-1, 1))
            timesteps_scaler = StandardScaler().fit(train_timestamps.reshape(-1, 1))
            # Train
            timesteps_to_normalize = np.array(self.data["train"]["timestamps"])
            self.data["train"]["timestamps"] = timesteps_scaler.transform(timesteps_to_normalize.reshape(-1, 1)).reshape(timesteps_to_normalize.shape)
            # Val
            timesteps_to_normalize = np.array(self.data["val"]["timestamps"])
            self.data["val"]["timestamps"] = timesteps_scaler.transform(timesteps_to_normalize.reshape(-1, 1)).reshape(timesteps_to_normalize.shape)
            # Test
            timesteps_to_normalize = np.array(self.data["test"]["timestamps"])
            self.data["test"]["timestamps"] = timesteps_scaler.transform(timesteps_to_normalize.reshape(-1, 1)).reshape(timesteps_to_normalize.shape)
            
        #======================================================================#
        #======================================================================#
        elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
            # Getting statistics of train dataset
            static_features_per_patient,\
            dynamic_features_per_patient,\
            dynamic_features_per_place = get_statistics_train_dataset_murcia(
                                                                                py_murcia_data_from_h5=self.data,
                                                                                train_steps_IDs=list(range(INIT_TRAIN_STEP, LAST_TRAIN_STEP)),
                                                                                patient_features_names_idx=self.patients_features_names_idx,
                                                                                place_features_names_idx=self.places_features_names_idx
                                                                            )
            
            # Creating the scalers
            # Ages: Standarization
            train_ages = np.array(list(static_features_per_patient['Age'].values()))
            scaler_age = StandardScaler().fit(train_ages)

            # Admission day: Min-max scaling
            train_admin_days = np.array(list(static_features_per_patient['AdmissionDay'].values()))
            scaler_admin_days = MinMaxScaler().fit(train_admin_days)

            # los_in_place: Min-max scaling
            train_los_in_place = np.array(list(dynamic_features_per_patient['los_in_place'].values()))
            scaler_los_in_place = MinMaxScaler().fit(train_los_in_place)

            # n_patients_same_place: Min-max scaling
            train_n_patients_same_place = np.array(list(dynamic_features_per_patient['n_patients_same_place'].values()))
            scaler_n_patients_same_place = MinMaxScaler().fit(train_n_patients_same_place)

            # nPatients for Place nods: Min-max scaling
            train_nPatients = np.array(list(dynamic_features_per_place['nPatients'].values()))
            scaler_nPatients = MinMaxScaler().fit(train_nPatients)

            # For timesteps
            train_timestamps = np.array(list(range(INIT_TRAIN_STEP, LAST_TRAIN_STEP)))
            #timesteps_scaler = MinMaxScaler().fit(train_timestamps.reshape(-1, 1))
            timesteps_scaler = StandardScaler().fit(train_timestamps.reshape(-1, 1))


            # Applying scalers to features
            n_steps = len(self.data["NodeFeaturesDicts"])
            for step_ID in range(n_steps):
                # Changing the types of the arrays to have float instead of ints (as normalization will create floats)
                self.data["NodeFeaturesDicts"][step_ID]['Patient'] = self.data["NodeFeaturesDicts"][step_ID]['Patient'].astype(np.float64)
                self.data["NodeFeaturesDicts"][step_ID]['Place'] = self.data["NodeFeaturesDicts"][step_ID]['Place'].astype(np.float64)
                
                # Age normalization
                init_age_idx, end_age_idx = self.patients_features_names_idx['Age'][0], self.patients_features_names_idx['Age'][1]
                data_to_scale = deepcopy(self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_age_idx:end_age_idx])
                scaled_feature = scaler_age.transform(data_to_scale)
                self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_age_idx:end_age_idx] = deepcopy(scaled_feature)

                # Admission day normalization
                init_admin_day_idx, end_admin_day_idx = self.patients_features_names_idx['AdmissionDay'][0], self.patients_features_names_idx['AdmissionDay'][1]
                data_to_scale = deepcopy(self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_admin_day_idx:end_admin_day_idx])
                scaled_feature = scaler_admin_days.transform(data_to_scale)
                self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_admin_day_idx:end_admin_day_idx] = deepcopy(scaled_feature)

                # los_in_place normalization
                init_los_in_place_idx, end_los_in_place_idx = self.patients_features_names_idx['los_in_place'][0], self.patients_features_names_idx['los_in_place'][1]
                data_to_scale = deepcopy(self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_los_in_place_idx:end_los_in_place_idx])
                scaled_feature = scaler_los_in_place.transform(data_to_scale)
                self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_los_in_place_idx:end_los_in_place_idx] = deepcopy(scaled_feature)

                # n_patients_same_place normalization
                init_n_patients_same_place_idx, end_n_patients_same_place_idx = self.patients_features_names_idx['n_patients_same_place'][0], self.patients_features_names_idx['n_patients_same_place'][1]
                data_to_scale = deepcopy(self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_n_patients_same_place_idx:end_n_patients_same_place_idx])
                scaled_feature = scaler_n_patients_same_place.transform(data_to_scale)
                self.data["NodeFeaturesDicts"][step_ID]['Patient'][:, init_n_patients_same_place_idx:end_n_patients_same_place_idx] = deepcopy(scaled_feature)

                # nPatients in place normalization
                init_nPatients_idx, end_nPatients_idx = self.places_features_names_idx['nPatients'][0], self.places_features_names_idx['nPatients'][1]
                data_to_scale = deepcopy(self.data["NodeFeaturesDicts"][step_ID]['Place'][:, init_nPatients_idx:end_nPatients_idx])
                scaled_feature = scaler_nPatients.transform(data_to_scale)
                self.data["NodeFeaturesDicts"][step_ID]['Place'][:, init_nPatients_idx:end_nPatients_idx] = deepcopy(scaled_feature)

                # Normalizing the timestamps
                for node_type in ['Patient', 'Place']:
                    timesteps_to_normalize = deepcopy(self.data['NodeTimestampsDicts'][step_ID][node_type])
                    self.data['NodeTimestampsDicts'][step_ID][node_type] = timesteps_scaler.transform(timesteps_to_normalize.reshape(-1, 1)).reshape(timesteps_to_normalize.shape)

            # Creating the normalized train and test datasets
            # Train
            train_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_TRAIN_STEP:LAST_TRAIN_STEP]}
            self.train_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=self.data["EdgeIndexDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            feature_dicts=self.data["NodeFeaturesDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            target_dicts=self.data["NodeTargetsDict"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            timestamps=self.data["NodeTimestampsDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            ids=self.data["NodesIdsDicts"][INIT_TRAIN_STEP:LAST_TRAIN_STEP],
                                                            **train_other_attributes
                                                        )
            # Val
            val_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_VAL_STEP:LAST_VAL_STEP]}
            self.val_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=self.data["EdgeIndexDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            feature_dicts=self.data["NodeFeaturesDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            target_dicts=self.data["NodeTargetsDict"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            timestamps=self.data["NodeTimestampsDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            ids=self.data["NodesIdsDicts"][INIT_VAL_STEP:LAST_VAL_STEP],
                                                            **val_other_attributes
                                                        )

            # Test
            test_other_attributes = {"transitions_targets": self.data["NodeTargetsTransitionsDict"][INIT_TEST_STEP:LAST_TEST_STEP]}
            self.test_ds = DynamicHeteroGraphTemporalSignal(
                                                                edge_index_dicts=self.data["EdgeIndexDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                edge_weight_dicts=self.data["EdgeFeaturesDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                feature_dicts=self.data["NodeFeaturesDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                target_dicts=self.data["NodeTargetsDict"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                timestamps=self.data["NodeTimestampsDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                ids=self.data["NodesIdsDicts"][INIT_TEST_STEP:LAST_TEST_STEP],
                                                                **test_other_attributes
                                                            )

        elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
            # For the HUG dataset, the normalization is implemented in src/DataManipulation/hug_covid/data_utils.py and done directly in the pre-processing function
            pass   
                
        else:
            raise ValueError()
        

    def createTorchDatasets(self, verbose=True, prefix_paths=""):
        """
            Create the torch datasets associated needed to train and evaluate the model
        """
        if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            if (self.parameters_exp['continual_training']):
                raise NotImplementedError()
            else:
                # Getting the pre-processed data
                self.preprocessSocioPatternsDataset(prefix_paths, forecast_horizon=self.parameters_exp['forecast_horizon'])

                # Creating the datasets
                # Train
                train_other_attributes = {
                                            "ids": self.data["train"]["nodes_ids"],
                                            "timestamps": self.data["train"]["timestamps"],
                                            "infection_rates": self.data["train"]["infection_rates"],
                                            "transitions_targets": self.data["train"]["transitions_targets"],
                                            "incubation_rates": [1/inc_dur for inc_dur in self.data["train"]["incubation_durations"]],
                                            "recovery_rates": [1/rec_dur for rec_dur in self.data["train"]["recovery_durations"]],
                                        }
                self.train_ds = DynamicGraphTemporalSignal(
                                                                edge_indices=self.data["train"]["edges_indices"],
                                                                edge_weights=self.data["train"]["edges_weights"],
                                                                features=self.data["train"]["features"],
                                                                targets=self.data["train"]["targets"],
                                                                **train_other_attributes
                                                            )

                # Validation
                val_other_attributes = {
                                            "ids": self.data["val"]["nodes_ids"],
                                            "timestamps": self.data["val"]["timestamps"],
                                            "infection_rates": self.data["val"]["infection_rates"],
                                            "transitions_targets": self.data["val"]["transitions_targets"],
                                            "incubation_rates": [1/inc_dur for inc_dur in self.data["val"]["incubation_durations"]],
                                            "recovery_rates": [1/rec_dur for rec_dur in self.data["val"]["recovery_durations"]],
                                        }
                self.val_ds = DynamicGraphTemporalSignal(
                                                            edge_indices=self.data["val"]["edges_indices"],
                                                            edge_weights=self.data["val"]["edges_weights"],
                                                            features=self.data["val"]["features"],
                                                            targets=self.data["val"]["targets"],
                                                            **val_other_attributes
                                                        )

                # Test
                test_other_attributes = {
                                            "ids": self.data["test"]["nodes_ids"],
                                            "timestamps": self.data["test"]["timestamps"],
                                            "infection_rates": self.data["test"]["infection_rates"],
                                            "transitions_targets": self.data["test"]["transitions_targets"],
                                            "incubation_rates": [1/inc_dur for inc_dur in self.data["test"]["incubation_durations"]],
                                            "recovery_rates": [1/rec_dur for rec_dur in self.data["test"]["recovery_durations"]],
                                            
                                        }
                self.test_ds = DynamicGraphTemporalSignal(
                                                            edge_indices=self.data["test"]["edges_indices"],
                                                            edge_weights=self.data["test"]["edges_weights"],
                                                            features=self.data["test"]["features"],
                                                            targets=self.data["test"]["targets"],
                                                            **test_other_attributes
                                                        )
        elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
            # Getting the pre-processed data (datasets created inside)
            if (self.parameters_exp['continual_training']):
                raise NotImplementedError()
            else:
                self.preprocessMurciaDataset(prefix_paths)

        elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
            # IMPORTANT: HERE THE DISTINCTION BETWEEN CLASSICAL AND CONTINUAL TRAINING IS DONE BEFORE THANKS TO data_splits_thresholds_IDs_dict
            # IMPORTANT: SO THERE IS NO NEED TO IDENTIFY BETWEEN THE TWO CASES
            self.train_ds,\
            self.val_ds,\
            self.test_ds = preprocessHUGCovidDataset(
                                                        h5_file=self.h5_file,
                                                        parameters_exp=self.parameters_exp,
                                                        patients_features_names_idx=self.original_patients_features_names_idx,
                                                        places_features_names_idx=self.original_places_features_names_idx,
                                                        data_splits_thresholds_IDs_dict=self.data_splits_thresholds_IDs_dict,
                                                        prefix_paths=prefix_paths
                                                    )
            # Re-open HDF5 file as it is closed inside preprocessHUGCovidDataset
            self.h5_file = h5py.File(self.hdf5_dataset_filename, 'r')

        else:
            raise ValueError()
        
        # Empty val dataset?
        if ((type(self.val_ds) == list) and (len(self.val_ds) > 0)) or\
           (type(self.val_ds) == DynamicGraphTemporalSignal) or\
           (type(self.val_ds) == DynamicHeteroGraphTemporalSignal):
            self.empty_val_ds = False
        else:
            self.empty_val_ds = True

        
        # Input channels for the different models
        if (self.remove_epi_state):
            if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                self.parameters_exp['in_channels'] =  {
                                                            "Patient": 2,
                                                            "Place": 2
                                                        }
            else:
                raise NotImplementedError(f"Remove epidemiological state for {self.parameters_exp['dataset_name']} dataset is not implemented yet.")
        else:
            tmp_snapshot = self.train_ds[0]
            self.parameters_exp['in_channels'] = None
            try:
                self.parameters_exp["in_channels"] = {}
                x = {key: torch.tensor(tmp_snapshot[key].x).to(self.device).float() for key in tmp_snapshot.node_types}
                for key in tmp_snapshot.node_types:
                    self.parameters_exp['in_channels'][key] = x[key].shape[1]
            except:
                x = torch.tensor(tmp_snapshot.x).to(self.device).float()
                self.parameters_exp['in_channels'] = x.shape[1]


    def computeClassWeightsLoss(self):
        def compute_weights(labels, num_classes):
            labels = labels.flatten()
            mask = labels != -1
            filtered_labels = labels[mask]
            
            counts = Counter(filtered_labels.tolist())
            
            # Use the fixed number of classes to avoid index out of bounds
            weights = torch.zeros(num_classes)
            for i in range(num_classes):
                count = counts.get(i, 0)
                # Add a small epsilon or max clip to avoid division by zero
                weights[i] = 1.0 / count if count > 0 else 0.0
            
            if weights.sum() > 0:
                # Standardize so the total loss magnitude stays consistent
                weights = (weights / weights.sum()) * num_classes
                
            return weights.to(self.device)
        
        # Output variable
        self.class_weights_to_use = {}

        # Made with the help of Gemini
        # Get the number of classes
        # Assuming SIR (3 classes), SEIR (4 classes) or SEIRD-NS (6 classes)
        if (self.parameters_exp['dataset_name'].lower() == 'hug'):
            num_classes = 3
        elif (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            num_classes = 4
        elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
            num_classes = 6

        # Initialize Count Matrix [Num_Classes, Num_Classes]
        counts = torch.zeros((num_classes, num_classes))

        # Collect Transition Statistics from Training Set
        if (self.parameters_exp["predict_state_transitions"]):
            all_transitions_labels = []
        for snapshot in self.train_ds:
            # Extract starting states for all patients [N]
            if (self.parameters_exp['dataset_name'].lower() in ['murcia', 'hug']):
                if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                    start_idx, end_idx = self.patients_features_names_idx['State']
                else:
                    start_idx, end_idx = self.patients_features_names_idx['current_state']
                y_start = snapshot['Patient'].x[:, start_idx:end_idx].argmax(dim=1)
                y_window = snapshot['Patient'].y # Shape [N, W]
                if (self.parameters_exp["predict_state_transitions"]):
                    y_transitions = snapshot['Patient'].transitions_targets
                
            else: # SocioPatterns
                y_start = snapshot.x[:, -4:].argmax(dim=1)
                y_window = snapshot.y # Shape [N, W]
                if (self.parameters_exp["predict_state_transitions"]):
                    y_transitions = snapshot.transitions_targets # Shape [N, W]
            if (self.parameters_exp["predict_state_transitions"]):
                all_transitions_labels.append(y_transitions)

            # Iterate through every step in the forecast window W
            if (len(y_window.shape) == 1): # i.e. Forecast Horizon of size 1
                y_window = y_window.unsqueeze(dim=1) # Now of dimension (N, W)
            N, W = y_window.shape
            for w in range(W):
                y_step = y_window[:, w]
                
                # Mask invalid labels (-1)
                mask = (y_step != -1)
                y_s_filtered = y_start[mask]
                y_w_filtered = y_step[mask]

                # Populate counts for this specific window step
                for i in range(len(y_s_filtered)):
                    counts[y_s_filtered[i], y_w_filtered[i]] += 1

        # Compute Weights using Smoothed Inverse Frequency
        # We add 1 to counts to handle zero-occurrence transitions gracefully
        weights = 1.0 / (counts + 1.0)

        # 4. Apply "Identity Protection"
        # Even if transitions are rare, we don't want them to be 1000x heavier
        # than stability, or the model will ignore the 'Identity Mapping' loss.
        for i in range(num_classes):
            diag_val = weights[i, i]
            # Cap the penalty for any transition at 25x the cost of a stability error
            weights[i, :] = torch.clamp(weights[i, :], max=diag_val * 25.0)

        # 5. Normalize and Register
        # Normalizing by the mean keeps the loss magnitude predictable
        self.transition_weights = (weights / weights.mean()).to(self.device)


        # Class weights for transition classification
        if (self.parameters_exp["predict_state_transitions"]):
            all_transitions_labels = torch.cat(all_transitions_labels, axis=0).numpy()
            num_classes_transitions = len(np.unique(all_transitions_labels))
            self.class_weights_to_use['TransitionLabel'] = compute_weights(all_transitions_labels, num_classes_transitions)
            # Number of transitions states
            self.n_transitions_types = num_classes_transitions


    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            deep learning model
        """
        if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns') or\
           (self.parameters_exp['dataset_name'].lower() == 'murcia') or\
           (self.parameters_exp['dataset_name'].lower() == 'hug'):
             # For the datasets that we use now, we do not need data loaders
             self.train_loader = self.train_ds
             self.val_loader = self.val_ds
             self.test_loader = self.test_ds
        else:
            raise ValueError()
        
    def createLossFunction(self):
        """
            Create the loss function to use for optimization
        """
        # Loss for state prediction
        self.criterion_states = TemporalTransitionAwareLoss(
                                                                weights_matrix=self.transition_weights,
                                                                temporal_decay=True,
                                                                ignore_index=-1
                                                            )
        
        # State transition loss
        if (self.parameters_exp["predict_state_transitions"]):
            self.criterion_transitions = torch.nn.CrossEntropyLoss(
                                                                            weight=self.class_weights_to_use['TransitionLabel'],
                                                                            ignore_index=-1 # Handles your "invalid" dates automatically
                                                                        )

        # Regularization lossses
        # Weight of the classification loss function 
        if ('lambda_classif_seir_states' not in self.parameters_exp):
            self.parameters_exp['lambda_classif_seir_states'] = 1

        # Epidemio-informed regularization
        if (self.parameters_exp['epidemio_informed']):
            # Loss function
            if ('loss_function_reg' not in self.parameters_exp):
                self.parameters_exp['loss_function_reg'] = "MSE"
            
            if (self.parameters_exp['loss_function_reg'].lower() == "mse"):
                self.criterion_reg = torch.nn.MSELoss()
            elif (self.parameters_exp['loss_function_reg'].lower() == "hubert"):
                self.criterion_reg = torch.nn.HuberLoss(delta=1.0)
            elif (self.parameters_exp['loss_function_reg'].lower() == "ce"):
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    self.criterion_reg = torch.nn.CrossEntropyLoss()
                else:
                    raise ValueError("\nCross Entropy loss is not valid for the epidemio-informed term for other methods than ODEs_Next_State_Residuals")
            else:
                raise ValueError()
            
            # Weights for regularization terms
            if ('lambda_reg' not in self.parameters_exp):
                self.parameters_exp['lambda_reg'] = 1e-1
            self.initial_lambda_reg = self.parameters_exp['lambda_reg']
            if ('lambda_reg_S' not in self.parameters_exp):
                self.parameters_exp['lambda_reg_S'] = 1e-2
            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                if ('lambda_reg_E' not in self.parameters_exp):
                    self.parameters_exp['lambda_reg_E'] = 1e-2
            if ('lambda_reg_I' not in self.parameters_exp):
                self.parameters_exp['lambda_reg_I'] = 1
            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                if ('lambda_reg_R' not in self.parameters_exp):
                    self.parameters_exp['lambda_reg_R'] = 1
            if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                if ('lambda_reg_D' not in self.parameters_exp):
                    self.parameters_exp['lambda_reg_D'] = 1
                if ('lambda_reg_NS' not in self.parameters_exp):
                    self.parameters_exp['lambda_reg_NS'] = 1

        # Loss function to constrain transitions between consecutive days
        # for the cases where we have a forecast horizon >= 2
        if (self.parameters_exp['forecast_horizon'] >= 2):
            # Transitions constraints loss
            if ("constrain_transitions_loss" not in self.parameters_exp):
                #self.parameters_exp["constrain_transitions_loss"] = False
                self.parameters_exp["constrain_transitions_loss"] = True
            if (self.parameters_exp["constrain_transitions_loss"]):
                if ("lambda_constrain_transitions_loss" not in self.parameters_exp):
                    self.parameters_exp["lambda_constrain_transitions_loss"] = 1e-2
            # Monotonicity
            if ("monotonicity_loss" not in self.parameters_exp):
                #self.parameters_exp["monotonicity_loss"] = False
                self.parameters_exp["monotonicity_loss"] = True
            if (self.parameters_exp["monotonicity_loss"]):
                if ("lambda_monotonicity_loss" not in self.parameters_exp):
                    self.parameters_exp["lambda_monotonicity_loss"] = 1e-2
        else:
            self.parameters_exp["constrain_transitions_loss"] = False # This loss can only be used for forecast horizon >= 2

    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        # Creating the main model
        super().modelCreation()

        #==============================================================#
        #==============================================================#
        # Create for multi-class classifier
        if (self.parameters_exp['epidemio_informed']) and (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
            in_channels = self.parameters_exp["out_channels_node_encoder"]+1 # +1 because we add time dimension
        else:
            in_channels = self.parameters_exp["out_channels_node_encoder"]

        if (self.parameters_exp['forecast_horizon'] == 1):
            model_compart_pred = MulticlassClassifier(
                                                        in_channels=in_channels,
                                                        out_channels=self.parameters_exp["out_channels"], 
                                                        hidden_sizes=[], 
                                                        metadata=self.parameters_exp['metadata'], 
                                                        device=torch.device(str(self.device))
                                                    )
        else:
            model_compart_pred = MultistepClassifier(
                                                        in_channels=in_channels,
                                                        num_classes=self.parameters_exp['num_states'],
                                                        forecast_horizon=self.parameters_exp['forecast_horizon'],
                                                        hidden_sizes=[], 
                                                        metadata=self.parameters_exp['metadata'], 
                                                        device=torch.device(str(self.device))
                                                    )
            
        # #==============================================================#
        # #==============================================================#
        # Model for state transitions prediction
        if (self.parameters_exp["predict_state_transitions"]):
            if (self.n_transitions_types is None):
                if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                    self.n_transitions_types = 3
                elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                    self.n_transitions_types = 4
                elif (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                    self.n_transitions_types = 4
            model_transitions_pred = MulticlassClassifier(
                                                            in_channels=in_channels,
                                                            out_channels=self.n_transitions_types, 
                                                            hidden_sizes=[], 
                                                            metadata=self.parameters_exp['metadata'], 
                                                            device=torch.device(str(self.device))
                                                        )
        
            
        #==============================================================#
        #==============================================================#
        # Create model for epidemics parameters
        if (self.parameters_exp['epidemio_informed']):
            # Epidemic parameters per model
            if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    n_epi_params = 11
                elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    n_epi_params = 5
            elif (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                n_epi_params = 3
            elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    n_epi_params = 9 # For SIR model with immigration/emmigration
                elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    n_epi_params = 9 # For SIR model with immigration/emmigration
                    
            # Model creation
            if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                model_epi_params = PerIndEpidemioParamsRegressor(
                                                                        n_epi_params=n_epi_params, 
                                                                        device=self.device
                                                                    )
            elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                model_epi_params = ODENextStatePredictor(
                                                                n_epi_params=n_epi_params, 
                                                                device=self.device
                                                            )
            else:
                raise ValueError(f"\nEpdemio-informed approach {self.parameters_exp['epidemio_informed_method']} is not valid.\n")

        #==============================================================#
        #==============================================================#
        # Intialize params
        model_compart_pred.apply(init_weights)
        if (self.parameters_exp['epidemio_informed']):
            if (self.init_epi_params_predefined_vals):
                if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                    # True paremeters used in simulation (DO NOT USE THEM AS PREDICTIONS CAN BE BIASED)
                    # beta = 0.5
                    # sigma = 1/5 
                    # gamma = 1/14
                    # Realistic initialization of the parameters WITHOUT using the exact parameters of the simulation
                    beta = 0.5
                    sigma = 1/7 
                    gamma = 1/12
                    with torch.no_grad():
                        model_epi_params.epi_params[0] = safe_inverse_softplus(beta) # beta
                        model_epi_params.epi_params[1] = safe_inverse_softplus(sigma) # sigma
                        model_epi_params.epi_params[2] = safe_inverse_softplus(gamma) # gamma
                elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                    if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                        beta = 1/(0.435 + 0.24)
                        A = 18.603
                        A_S = 0.997
                        A_E = 1e-7 # Cannot be 0 because it create NaN params
                        A_I = 0.002
                        A_R = 1e-7 # Cannot be 0 because it create NaN params
                        A_NS = 0.001
                        dis_rate = 1/4.254
                        alpha = 1/2.5
                        mu = 0.027
                        gamma = 1 - mu
                        with torch.no_grad():
                            model_epi_params.epi_params[0] = safe_inverse_softplus(beta) # beta
                            model_epi_params.epi_params[1] = safe_inverse_softplus(A) # A
                            model_epi_params.epi_params[2] = safe_inverse_softplus(A_S) # A_S
                            model_epi_params.epi_params[3] = safe_inverse_softplus(A_E) # A_E
                            model_epi_params.epi_params[4] = safe_inverse_softplus(A_I) # A_I
                            model_epi_params.epi_params[5] = safe_inverse_softplus(A_R) # A_R
                            model_epi_params.epi_params[6] = safe_inverse_softplus(A_NS) # A_NS
                            model_epi_params.epi_params[7] = safe_inverse_softplus(dis_rate) # dis_rate
                            model_epi_params.epi_params[8] = safe_inverse_softplus(alpha) # alpha
                            model_epi_params.epi_params[9] = safe_inverse_softplus(gamma) # gamma
                            model_epi_params.epi_params[10] = safe_inverse_softplus(mu) # mu
                    elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                        beta = 1/(0.435 + 0.24) 
                        dis_rate = 1/4.254 
                        alpha = 1/2.5 
                        mu = 0.027                        
                        gamma = 1 - mu 
                        with torch.no_grad():
                            model_epi_params.epi_params[0] = safe_inverse_softplus(beta) # beta
                            model_epi_params.epi_params[1] = safe_inverse_softplus(dis_rate) # dis_rate
                            model_epi_params.epi_params[2] = safe_inverse_softplus(alpha) # alpha
                            model_epi_params.epi_params[3] = safe_inverse_softplus(gamma) # gamma
                            model_epi_params.epi_params[4] = safe_inverse_softplus(mu) # mu
                elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                    beta = 0.5
                    gamma = 0.1
                    lambda_S = 0.02
                    lambda_I = 0.001
                    lambda_R = 0.001
                    mu_S = 0.01
                    mu_I = 0.02
                    mu_R = 0.005
                    epsilon = 0.001
                    with torch.no_grad():
                        model_epi_params.epi_params[0] = safe_inverse_softplus(beta) # beta
                        model_epi_params.epi_params[1] = safe_inverse_softplus(gamma) # gamma
                        model_epi_params.epi_params[2] = safe_inverse_softplus(lambda_S) # lambda_S
                        model_epi_params.epi_params[3] = safe_inverse_softplus(lambda_I) # lambda_I
                        model_epi_params.epi_params[4] = safe_inverse_softplus(lambda_R) # lambda_R
                        model_epi_params.epi_params[5] = safe_inverse_softplus(mu_S) # mu_S
                        model_epi_params.epi_params[6] = safe_inverse_softplus(mu_I) # mu_I
                        model_epi_params.epi_params[7] = safe_inverse_softplus(mu_R) # mu_R
                        model_epi_params.epi_params[8] = safe_inverse_softplus(epsilon) # epsilon
                # Check if it requires gradients
                print(f"Requires Grad: {model_epi_params.epi_params.requires_grad}") 
                # Check if it is a 'Leaf' tensor (only leaves can be updated by optimizers)
                print(f"Is Leaf: {model_epi_params.epi_params.is_leaf}")
                # Check if it's in the model parameters list
                print(f"In Model Params: {any(p is model_epi_params.epi_params for p in model_epi_params.parameters())}")

            else:
                model_epi_params.apply(init_weights)
        if (self.parameters_exp["predict_state_transitions"]):
            model_transitions_pred.apply(init_weights)


        #==============================================================#
        #==============================================================#
        # Adding the models to the list of models
        self.models["model_compart_pred"] = model_compart_pred
        if (self.parameters_exp['epidemio_informed']):
            self.models["model_epi_params"] = model_epi_params
        if (self.parameters_exp["predict_state_transitions"]):
            self.models["model_transitions_pred"] = model_transitions_pred

        print(f"\n\n===>New model_compart_pred and model_epi_params created !\n\n")


    # Function to compute the derivatitves of the predicted SEIR probabilities per node
    def compute_time_derivatives_seir(
                                        self, 
                                        seir_pred_states,
                                        timestamps,
                                        global_SEIR=False,
                                        method_global_SEIR_states_count="Softmax"
                                    ):
        """
            Compute the time derivatives of the SEIR probabilities
            of each state for each node.

            Parameters:
            -----------
            seir_pred_states: torch.tensor
                Predicted probabilities of the SEIR states of shape (num_nodes, num_states)
                or (num_nodes, num_states, forecast_horizon)
            timestamps: torch.tensor
                Timestamps of each measurment.
            global_SEIR: bool
                If true, we compute the derivatives as a global SEIR model behavious
                and not per individual (using average infection, incubation and recovery
                rates).
            method_global_SEIR_states_count: str
                Method to use to approximate the number of patients per compartment (as
                we have probabilities and not values per compartment, and the argmax
                function is not differentiable). Two options: GumbelSoftmax and Softmax

            Returns:
            --------
            dt_out: torch.tensor
                Theoretical derivatives of S, E, I and R using the SEIR network-based
                differential equations.
        """
        grads = []
        if (not global_SEIR):
            for i in range(seir_pred_states.shape[1]):
                grad_outputs = torch.zeros_like(seir_pred_states)
                grad_outputs[:, i] = 1.0
                grad = torch.autograd.grad(
                                                outputs=seir_pred_states,
                                                inputs=timestamps,
                                                grad_outputs=grad_outputs,
                                                create_graph=True,  # IMPORTANT FOR BACKPROPAGATION LATER WHEN COMPUTING THE LOSS FUNCTION
                                                retain_graph=True
                                        )[0]
                grads.append(grad)
            dt_out = torch.cat(grads, dim=1)  # shape (403, 4)
        else:
            # IMPORTANT: ARGMAX IS NOT DIFFERENTIABLE, SO TO COUNT THE NUMBER OF PATIENTS PER COMPARTMENT WE HAVE TO DO IT IN
            # IMPORTANT: ANOTHER (DIFFERENTIABLE) WAY. FOR INSTANCE, AFTER USING SOFTMAX WE CAN JUST SUM
            if (method_global_SEIR_states_count.lower() == "gumbelsoftmax"):
                '''
                # Hyperparameter, the smaller to closer to argmax
                temperature = 0.5
                # Gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(seir_pred_states)))
                y = (seir_pred_states + gumbel_noise) / temperature
                # Gumbel-softmax
                probs = F.softmax(y, dim=1)  # shape: (num_patients, 4)
                expected_counts_per_compartment = torch.sum(probs, dim=0)  # shape: (4,)
                '''
                pass
            elif (method_global_SEIR_states_count.lower() == "softmax"):
                expected_counts_per_compartment = torch.sum(seir_pred_states, dim=0)
            else:
                raise ValueError("\nMethod {} to approximate argmax for compartment counting is no valid\n".format(method_global_SEIR_states_count))
            
            # Computing derivatives
            for i in range(expected_counts_per_compartment.shape[0]):
                grad_outputs = torch.zeros_like(expected_counts_per_compartment)
                grad_outputs[i] = 1.0
                grad = torch.autograd.grad(
                                                outputs=expected_counts_per_compartment,
                                                inputs=timestamps,
                                                grad_outputs=grad_outputs,
                                                create_graph=True,  # IMPORTANT FOR BACKPROPAGATION LATER WHEN COMPUTING THE LOSS FUNCTION
                                                retain_graph=True
                                        )[0]
                grads.append(grad)
            dt_out = torch.stack(grads)  # shape (4)

        return dt_out

    def teacher_forcing_scheduler(self, epoch_nb, lambda_max=1.0, ramp_epochs=10):
        """
            Gives the value of the weight of the epidemio-informed loss
            function and the probability of using teacher forcing
            for the computation of the epidemio-informed loss
        """
        ratio = epoch_nb / float(ramp_epochs)
        p_teacher_forcing = min(1.0, ratio)
        lambda_reg = lambda_max * min(1.0, ratio)

        return p_teacher_forcing, lambda_reg

    def compute_transition_constraint_loss(self, logits, y):
        """
            Done with the help of Gemini.
            Computes the states transitions loss.

            Arguments:
            ----------
                logits: [N, N_STATES, H]
                    Model output.
                y: [N, H]
                    Ground truth labels.
        """
        # 1. Select the correct transition matrix
        if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            T = self.T_SEIR.to(self.device)
        elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
            T = self.T_SEIRD_NS.to(self.device)
        elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
            T = self.T_SIR.to(self.device)
        else:
            raise ValueError("Unknown dataset type")

        # Batch size, classes/states, horizon
        N, C, H = logits.shape
        probs = torch.softmax(logits, dim=1) # [N, C, H]
        
        # Get state at t and predictions at t+1
        y_t = y[:, :-1]          # Ground truth at step h: [N, H-1]
        p_next = probs[:, :, 1:] # Predicted prob at step h+1: [N, C, H-1]

        # Create the forbidden mask
        # y_t has values 0 to C-1. We use it to index the rows of T.
        # We ignore -1 (invalid labels) to avoid IndexErrors
        y_t_safe = y_t.clone()
        y_t_safe[y_t == -1] = 0 
        
        # T[y_t_safe] gives [N, H-1, C]. Values are 1 for valid, 0 for illegal.
        # 1 - T flips this: 1 for illegal, 0 for valid.
        forbidden_transitions = 1 - T[y_t_safe] # [N, H-1, C]
        forbidden_transitions = forbidden_transitions.permute(0, 2, 1) # [N, C, H-1]

        # Calculate penalty
        # Element-wise multiplication of predicted probs and the forbidden mask
        illegal_prob_mass = p_next * forbidden_transitions # [N, C, H-1]
        
        # Masking and Normalization
        # Ignore time steps where ground truth was -1
        valid_label_mask = (y_t != -1).float().unsqueeze(1) # [N, 1, H-1]
        
        # Sum of all illegal probabilities in valid slots
        loss = (illegal_prob_mass * valid_label_mask).sum()
        
        # Normalize by the number of valid transitions checked
        denominator = valid_label_mask.sum() * C

        return loss / (denominator + 1e-7)
    
    def compute_monotonicity_penalty(self, logits):
        # probs: [N, C, H]
        probs = torch.softmax(logits, dim=1)
        
        # Calculate the "Expected State Index" at each step
        # S=0, E=1, I=2, R=3. We want this value to be non-decreasing over H.
        state_indices = torch.arange(logits.shape[1], device=logits.device).float()
        expected_state = torch.einsum('nch,c->nh', probs, state_indices) # [N, H]
        
        # Penalize if expected_state at h+1 < expected_state at h
        diffs = expected_state[:, 1:] - expected_state[:, :-1]
        monotonicity_loss = torch.relu(-diffs).mean() 
        
        return monotonicity_loss


    def computeForwardPass(self, batch, epoch_nb, batch_ID=None):
        #======================================================================#
        #======================================================================#
        #======================================================================#
        if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns') or\
           (self.parameters_exp['dataset_name'].lower() == 'murcia') or\
           (self.parameters_exp['dataset_name'].lower() == 'hug'):
            #======================================================================#
            #======================================================================#
            # IN THIS CASE ONE BATCH IS ONE SNAPSHOT
            snapshot = batch
            # Get inputs and labels
            if (self.models['model_node_enc'].hetero_mode):
                x = {key: torch.tensor(snapshot[key].x).to(self.device).float() for key in snapshot.node_types}
                y = {key: torch.tensor(snapshot[key].y).to(self.device).long() for key in snapshot.node_types}
                if (self.parameters_exp["predict_state_transitions"]):
                    y_transitions = {key: torch.tensor(snapshot[key]["transitions_targets"]).to(self.device).long() for key in snapshot.node_types}
                edge_index = {key: torch.tensor(snapshot[key].edge_index).to(self.device).long() for key in snapshot.edge_types}
                edge_attr = {key: torch.tensor(snapshot[key].edge_attr).to(self.device).float() for key in snapshot.edge_types}
                timestamps = {key: torch.tensor(snapshot[key].timestamps).to(self.device).float() for key in snapshot.node_types}
                true_nodes_ids = {key: torch.tensor(snapshot[key].ids).to(self.device) for key in snapshot.node_types} 
            else:
                x = torch.tensor(snapshot.x).to(self.device).float()
                y = torch.tensor(snapshot.y).to(self.device).long()
                if (self.parameters_exp["predict_state_transitions"]):
                    y_transitions = torch.tensor(snapshot.transitions_targets).to(self.device).long()
                edge_index = torch.tensor(snapshot.edge_index).to(self.device).long()
                edge_attr = torch.tensor(snapshot.edge_attr).to(self.device).float()
                timestamps = torch.tensor(snapshot.timestamps).to(self.device).float()
                true_nodes_ids = torch.tensor(snapshot.ids).to(self.device)

            
            # Getting the correspondence between the local node IDs (in the current snapshot) and the true ones (in the whole dataset)
            # IMPORTANT: FOR STM-GNN DO NO USE THIS AS INPUT BUT THE ORIGINAL edge_index AS SOMETHING SIMILAR TO WHAT IS DONE HERE IS
            # IMPORTANT: DONE INSIDE THE MODEL
            if (self.models['model_node_enc'].hetero_mode):
                mapping_true_to_local_node_ID = {node_type:{} for node_type in snapshot.node_types}
                mapped_edges_index = {edge_type:None for edge_type in snapshot.edge_types}
                for node_type in snapshot.node_types:
                    n_nodes = true_nodes_ids[node_type].shape[0]
                    for local_node_ID in range(n_nodes):
                        mapping_true_to_local_node_ID[node_type][int(true_nodes_ids[node_type][local_node_ID])] = local_node_ID
                for edge_type in snapshot.edge_types:
                    mapped_edges_index[edge_type] = torch.empty(edge_index[edge_type].shape)
                    for local_edge_ID in range(edge_index[edge_type].shape[1]):
                        node_i_true_ID = int(edge_index[edge_type][0, local_edge_ID])
                        node_i_type = edge_type[0]
                        node_j_true_ID = int(edge_index[edge_type][1, local_edge_ID])
                        node_j_type = edge_type[2]
                        mapped_edges_index[edge_type][0, local_edge_ID] = mapping_true_to_local_node_ID[node_i_type][node_i_true_ID]
                        mapped_edges_index[edge_type][1, local_edge_ID] = mapping_true_to_local_node_ID[node_j_type][node_j_true_ID]
                    mapped_edges_index[edge_type] = mapped_edges_index[edge_type].to(self.device).long()
            else:
                mapping_true_to_local_node_ID = {}
                n_nodes = true_nodes_ids.shape[0]
                for local_node_ID in range(n_nodes):
                    mapping_true_to_local_node_ID[int(true_nodes_ids[local_node_ID])] = local_node_ID
                mapped_edges_index = torch.empty(edge_index.shape)
                for local_edge_ID in range(edge_index.shape[1]):
                    node_i_true_ID = int(edge_index[0, local_edge_ID])
                    node_j_true_ID = int(edge_index[1, local_edge_ID])
                    mapped_edges_index[0, local_edge_ID] = mapping_true_to_local_node_ID[node_i_true_ID]
                    mapped_edges_index[1, local_edge_ID] = mapping_true_to_local_node_ID[node_j_true_ID]
                mapped_edges_index = mapped_edges_index.to(self.device).long()

            
            #======================================================================#
            #======================================================================#
            # Gets prediction
            # Nodes embeddings
            current_epi_state = None
            if (self.parameters_exp['model_to_use'].lower() in ["stm", "tgn", "graphsage", "gat", "gcn"]):
                # IMPORTANT: HERE WE MUST NOT USE mapped_edges_index AS THE MODEL HANDLES IT USING true_nodes_ids and edge_index
                if (self.remove_epi_state):
                    if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        current_epi_state = deepcopy(x['Patient'][:, 1:4])
                        indices_to_keep = [0, 4]
                        x['Patient'] = x['Patient'][:, indices_to_keep]
                    else:
                        raise NotImplementedError(f"Remove epidemiological state for {self.parameters_exp['dataset_name']} dataset is not implemented yet.")

                node_embed = self.models["model_node_enc"](x, true_nodes_ids, edge_index, edge_attr, timestamps)   
            elif (self.parameters_exp['model_to_use'].lower().lower() == "mlp"):
                node_embed = self.models["model_node_enc"](x)   
            elif (self.parameters_exp['model_to_use'].lower().lower() == "simplegnn"):
                node_embed = self.models["model_node_enc"](x, mapped_edges_index)
            else:
                node_embed = self.models["model_node_enc"](x, mapped_edges_index, edge_attr)
            
            # Raise error if any of the models has NaN in their weights
            for model_type in self.models:
                for name, param in self.models[model_type].named_parameters():
                    if (torch.isnan(param).any()):
                        print(f"NaN found in model {model_type} for parameter: {name} (epoch: {epoch_nb}, batch ID: {batch_ID})")
                        raise RuntimeError(f"NaN found in model {model_type} for parameter: {name}")

            # Raise error if some of the embeddings are NaN
            if (self.models['model_node_enc'].hetero_mode):
                for node_type in node_embed:
                    if (node_embed[node_type].isnan().any()):
                        raise RuntimeError("At least one value of node_embed is NaN")

            # Add the time dimension to the node embeddings
            # Only needed if doing auto-differentiation with respect to time
            if (self.parameters_exp['epidemio_informed']):
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    if (self.models['model_node_enc'].hetero_mode):
                        shared_timestamp = torch.nn.Parameter(torch.tensor(timestamps['Patient'][0]), requires_grad=True)
                        for node_type in timestamps.keys():
                            timestamps[node_type] = shared_timestamp.expand(node_embed[node_type].shape[0], 1)  # shape: [num_nodes, 1]
                            node_embed[node_type] = torch.cat([timestamps[node_type], node_embed[node_type]], axis=1)
                    else:
                        shared_timestamp = torch.nn.Parameter(torch.tensor(timestamps[0]), requires_grad=True)
                        timestamps = shared_timestamp.expand(node_embed.shape[0], 1)  # shape: [num_nodes, 1]
                        node_embed = torch.cat([timestamps, node_embed], axis=1)
                elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    # IMPORTANT: no need to add time step value in the feature vector as we are not going to differentiate with respect to time!
                    pass
                else:
                    raise ValueError(f"\nEpdemio-informed approach {self.parameters_exp['epidemio_informed_method']} is not valid.\n")


            # Getting the prediction
            epi_states_scores = self.models["model_compart_pred"](node_embed)
            if (self.parameters_exp["predict_state_transitions"]):
                transitions_scores = self.models["model_transitions_pred"](node_embed)
            # Predictor of the states of the patients and the transitions(nodes)
            if (self.models['model_node_enc'].hetero_mode):
                if (self.parameters_exp['dataset_name'].lower() in ['murcia', 'hug']):
                    # We ONLY USE THE PATIENT NODES FOR PREDICTION
                    node_embed = node_embed['Patient']
                    epi_states_scores = epi_states_scores['Patient']
                    y = y['Patient']
                    if (self.parameters_exp["predict_state_transitions"]):
                        transitions_scores = transitions_scores['Patient']
                        y_transitions = y_transitions['Patient']
            if (self.parameters_exp['forecast_horizon'] >= 2):
                # Nothing to do as we already have the forecast horizon dimension
                pass
            else:
                epi_states_scores = epi_states_scores.unsqueeze(2)
                if (len(y.shape) > 1): # HUG-COVID dataset already forecast dimension
                    pass 
                else:
                    y = y.unsqueeze(1)

            # Getting the output probabilities
            epi_states_probs = torch.nn.functional.softmax(epi_states_scores, dim=1)
            if (self.parameters_exp["predict_state_transitions"]):
                transitions_probs = torch.nn.functional.softmax(transitions_scores, dim=1)
            

            #======================================================================#
            #======================================================================#
            # Parameters for teacher forcing if asked
            if (self.parameters_exp['epidemio_informed']):
                if (self.parameters_exp['teacher_forcing_epidemio']):
                    p_teacher_forcing, lambda_reg = self.teacher_forcing_scheduler(
                                                                                    epoch_nb=epoch_nb,
                                                                                    #lambda_max=1.0,
                                                                                    lambda_max=self.initial_lambda_reg,
                                                                                    #ramp_epochs=10
                                                                                    ramp_epochs=self.parameters_exp['ramp_epochs_teacher_forcing_sched']
                                                                                )
                    self.parameters_exp['lambda_reg'] = lambda_reg
                else:
                    p_teacher_forcing = 0
                random_val_teacher_forcing = random.random() # For teacher forcing
                if (batch_ID is not None) and (batch_ID == 0):
                    print(f"\n=========> Lambda for the regularization loss at epoch {epoch_nb}: {self.parameters_exp['lambda_reg']}\n")

            #======================================================================#
            #======================================================================#
            # Computing the derivatives (different terms of the ODEs) to compute
            # the residuals for the loss
            if (self.parameters_exp['epidemio_informed']):
                if (self.parameters_exp['teacher_forcing_epidemio']) or\
                (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    # One hot encoding of the current states
                    if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                        gt_current_states_probs = x[:, -4:] # Current state are one-hot encoded in the last 4 values of the feature vector

                    elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        first_curr_state_idx, last_curr_state_idx = self.patients_features_names_idx['current_state'][0], self.patients_features_names_idx['current_state'][1]
                        if (self.remove_epi_state):
                            gt_current_states_probs = current_epi_state 
                        else:
                            gt_current_states_probs = x['Patient'][:, first_curr_state_idx:last_curr_state_idx] # Current state are one-hot encoded in the last 4 values of the feature vector
                    
                    elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        first_curr_state_idx, last_curr_state_idx = self.patients_features_names_idx['State'][0], self.patients_features_names_idx['State'][1] # According to the Yaml files
                        if (self.remove_epi_state):
                            gt_current_states_probs = current_epi_state 
                        else:
                            gt_current_states_probs = x['Patient'][:, first_curr_state_idx:last_curr_state_idx] # Current state are one-hot encoded in the last values 1 to 3 in the feature vector
                    
                    # Getting the rest of the states necessary to compute the left-hand side of the ODEs
                    # Number of classes
                    if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                        n_classes = 4
                    elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        n_classes = 6
                    elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        n_classes = 3 # For SIR epidemic model

                    # Initial current state
                    gt_current_states_probs = [gt_current_states_probs.cpu().numpy()]

                    # Other states, in the ground truth
                    # IMPORTANT: We go until forecast_horizon - 1 because we are going to use this
                    # IMPORTANT: to compute the ODEs by discretization, so we do not need the last
                    # IMPORTANT: future state.
                    for hor in range(self.parameters_exp['forecast_horizon']-1):
                        tmp_gt_next_state_probs = one_hot_encoding_np(y[:, hor].cpu().numpy(), num_classes=n_classes)
                        gt_current_states_probs.append(tmp_gt_next_state_probs)
                    gt_current_states_probs = torch.from_numpy(np.stack(gt_current_states_probs, axis=2)).float().to(self.device)

                    # Add artificial batch dim for consistency with the model
                    gt_current_states_probs = gt_current_states_probs.unsqueeze(0)

                if (not self.parameters_exp['teacher_forcing_epidemio']) or\
                   (random_val_teacher_forcing >= p_teacher_forcing):
                    if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                        # Use predicted state at time t+1
                        pred_future_states_probs = epi_states_probs.unsqueeze(0) # Add artificial batch dim for consistency with the model
                
            # Computing the derivatives
            if (self.parameters_exp['epidemio_informed']):
                # Get adjacency matrix
                if (self.models['model_node_enc'].hetero_mode):
                    n_patients = x['Patient'].shape[0]
                    if (n_patients == 1): # We have only one patient
                        # We do not consider self-loops
                        mapped_edges_index_undirected, edge_attr_undirected = None, None 
                    else:
                        if (('Patient', 'to', 'Patient') in mapped_edges_index):
                            mapped_edges_index_undirected, edge_attr_undirected = to_undirected(
                                                                                                    edge_index=mapped_edges_index[('Patient', 'to', 'Patient')],
                                                                                                    edge_attr=edge_attr[('Patient', 'to', 'Patient')]
                                                                                                )
                        else:
                            # Get the list of  patients per place                        
                            unique_mapped_places_nodes_ID = np.unique(mapped_edges_index[('Patient', 'to', 'Place')][1, :].detach().cpu().numpy())
                            patients_nodes_IDs_per_place = {int(place_node_ID):[] for place_node_ID in unique_mapped_places_nodes_ID}
                            for edge_ID in range(mapped_edges_index[('Patient', 'to', 'Place')].shape[1]):
                                patient_node_ID = int(mapped_edges_index[('Patient', 'to', 'Place')][0, edge_ID])
                                place_node_ID = int(mapped_edges_index[('Patient', 'to', 'Place')][1, edge_ID])
                                patients_nodes_IDs_per_place[place_node_ID].append(patient_node_ID)

                            # Get contact between patients based on the places nodes
                            mapped_edges_index_patients = []
                            mapped_edges_attr_patients = []
                            seen_pairs = set()
                            for place_node_ID in patients_nodes_IDs_per_place:
                                n_patients_place = len(patients_nodes_IDs_per_place[place_node_ID])
                                for i in range(n_patients_place - 1):
                                    for j in range(i+1, n_patients_place):
                                        tmp_edge = [patients_nodes_IDs_per_place[place_node_ID][i], patients_nodes_IDs_per_place[place_node_ID][j]]
                                        mapped_edges_index_patients.append(tmp_edge)
                                        mapped_edges_attr_patients.append(1.0)
                            mapped_edges_index_patients = torch.tensor(mapped_edges_index_patients).to(self.device)
                            mapped_edges_attr_patients = torch.tensor(mapped_edges_attr_patients).to(self.device)

                            # Creating the undirected mapped edges indices and attributes tensors
                            if (self.parameters_exp['dataset_name'].lower() in ['murcia', 'hug']):
                                mapped_edges_index_patients = mapped_edges_index_patients.T
                            mapped_edges_index_undirected, edge_attr_undirected = to_undirected(
                                                                                                edge_index=mapped_edges_index_patients,
                                                                                                edge_attr=mapped_edges_attr_patients
                                                                                            )
                        

                else:
                    mapped_edges_index_undirected, edge_attr_undirected = to_undirected(
                                                                                            edge_index=mapped_edges_index,
                                                                                            edge_attr=edge_attr
                                                                                        )
                if (mapped_edges_index_undirected is None) or (edge_attr_undirected is None):
                    adj_mat = torch.tensor([[0.]]).float().to(self.device) # 0 because we do not have self-loops
                else:
                    adj_mat = to_dense_adj(
                                                edge_index=mapped_edges_index_undirected,
                                                edge_attr=edge_attr_undirected,
                                                max_num_nodes=epi_states_probs.shape[0],
                                            ).squeeze()
                
                # Compute derivatives
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    if (self.parameters_exp['teacher_forcing_epidemio']) and\
                       (random_val_teacher_forcing < p_teacher_forcing):
                            # IMPORTANT: doing this does not create data leakage as we use it just to compute the approximated derivate, but it is not
                            # IMPORTANT: fed to any model
                            if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                                delta_t = torch.tensor(1)
                                n_classes = 4
                            elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                                delta_t = torch.tensor(0.33)
                                n_classes = 6
                            elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                                delta_t = torch.tensor(1)
                                #n_classes = 2 # For SI epidemic model
                                n_classes = 3 # For SIR epidemic model
                            n_nodes, forecast_hor = y.shape
                            y_one_hot = []
                            for hor in range(forecast_hor):
                                y_one_hot.append(one_hot_encoding_np(y[:, hor].cpu().numpy(), num_classes=n_classes))
                            y_one_hot = torch.from_numpy(np.stack(y_one_hot, axis=2)).to(self.device)
                            # Left-hand side
                            dX_dT_left = (y_one_hot-gt_current_states_probs[0, ...])/delta_t
                            dX_dT_left = dX_dT_left.float()

                            # Right-hand side
                            dX_dT_right = self.models["model_epi_params"](gt_current_states_probs, adj_mat, normalize_adj_mat=False, use_future_adj_mats=False)
                    else:       
                        # Left-hand side
                        n_nodes, n_states, forecast_hor = epi_states_probs.shape
                        dX_dT_left = []
                        for hor in range(forecast_hor):
                            if (self.models['model_node_enc'].hetero_mode):
                                dX_dT_left.append(self.compute_time_derivatives_seir(epi_states_probs[:, :, hor], timestamps['Patient'], global_SEIR=False, method_global_SEIR_states_count="Softmax"))
                            else:
                                dX_dT_left.append(self.compute_time_derivatives_seir(epi_states_probs[:, :, hor], timestamps, global_SEIR=False, method_global_SEIR_states_count="Softmax"))
                        dX_dT_left = torch.stack(dX_dT_left, axis=2)

                        # Right-hand side
                        dX_dT_right = self.models["model_epi_params"](pred_future_states_probs, adj_mat, normalize_adj_mat=False, use_future_adj_mats=False)

                    dX_dT_right = dX_dT_right.squeeze(dim=0).transpose(0, 1) # Need to specify the squeeze dim as for the HUG dataset, some graphs can have only 1 patient

                    # Nomalizing derivatives before computing the loss
                    # Using all the nodes in the graph
                    epsilon = 1e-8
                    # Left-hand side
                    dX_dT_left = dX_dT_left.unsqueeze(dim=0)

                    # Normalize (only if there are more than one sample in the batch)
                    # Left-hand side
                    if (dX_dT_left.shape[0] > 1):
                        dX_dT_left = (dX_dT_left-dX_dT_left.mean(dim=0))/(dX_dT_left.std(dim=0)+epsilon)
                        
                    # Right-hand side (create new variables to avoid in-place modifications of the variable as it is used for gradient computation)
                    if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        # Normalize (only if there are more than one sample in the batch)
                        if (dX_dT_right.shape[0] > 1):
                            dS_dT_right = (dX_dT_right[:, 0, :]-dX_dT_right[:, 0, :].mean(dim=0))/(dX_dT_right[:, 0, :].std(dim=0) + epsilon)
                            dI_dT_right = (dX_dT_right[:, 1, :]-dX_dT_right[:, 1, :].mean(dim=0))/(dX_dT_right[:, 1, :].std(dim=0) + epsilon)
                            dR_dT_right = (dX_dT_right[:, 2, :]-dX_dT_right[:, 2, :].mean(dim=0))/(dX_dT_right[:, 2, :].std(dim=0) + epsilon)
                        else:
                            dS_dT_right = dX_dT_right[:, 0, :]
                            dI_dT_right = dX_dT_right[:, 1, :]
                            dR_dT_right = dX_dT_right[:, 2, :]
                    else:
                        # Normalize (only if there are more than one sample in the batch)
                        if (dX_dT_right.shape[0] > 1):
                            dS_dT_right = (dX_dT_right[:, 0, :]-dX_dT_right[:, 0, :].mean(dim=0))/(dX_dT_right[:, 0, :].std(dim=0) + epsilon)
                            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                                dE_dT_right = (dX_dT_right[:, 1, :]-dX_dT_right[:, 1, :].mean(dim=0))/(dX_dT_right[:, 1, :].std(dim=0) + epsilon)
                            dI_dT_right = (dX_dT_right[:, 2, :]-dX_dT_right[:, 2, :].mean(dim=0))/(dX_dT_right[:, 2, :].std(dim=0) + epsilon)
                            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                                dR_dT_right = (dX_dT_right[:, 3, :]-dX_dT_right[:, 3, :].mean(dim=0))/(dX_dT_right[:, 3, :].std(dim=0) + epsilon)
                            if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                                dD_dT_right = (dX_dT_right[:, 4, :]-dX_dT_right[:, 4, :].mean(dim=0))/(dX_dT_right[:, 4, :].std(dim=0) + epsilon)
                                dNS_dT_right = (dX_dT_right[:, 5, :]-dX_dT_right[:, 5, :].mean(dim=0))/(dX_dT_right[:, 5, :].std(dim=0) + epsilon)
                        else:
                            dS_dT_right = dX_dT_right[:, 0, :]
                            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                                dE_dT_right = dX_dT_right[:, 1, :]
                            dI_dT_right = dX_dT_right[:, 2, :]
                            if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                                dR_dT_right = dX_dT_right[:, 3, :]
                            if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                                dD_dT_right = dX_dT_right[:, 4, :]
                                dNS_dT_right = dX_dT_right[:, 5, :]

                elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    # IMPORTANT: doing this does not create data leakage as we use it just to compute the approximated derivate, but it is not
                    # IMPORTANT: fed to any model
                    if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                        delta_t = torch.tensor(1)
                        n_classes = 4
                    elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        delta_t = torch.tensor(0.33)
                        n_classes = 6
                    elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        delta_t = torch.tensor(1)
                        n_classes = 3 # For SIR epidemic model

                    # Getting the predictions of the probabilities of the next states based on the ODEs with Hazard waiting time
                    odes_next_states = self.models["model_epi_params"](gt_current_states_probs, adj_mat, delta_t, normalize_adj_mat=False, use_future_adj_mats=False)
                    odes_next_states = odes_next_states.squeeze(dim=0).T # Need to specify the squeeze dim as for the HUG dataset, some graphs can have only 1 patient
                else:
                    raise ValueError(f"\nEpdemio-informed approach {self.parameters_exp['epidemio_informed_method']} is not valid.\n")

            

            #======================================================================#
            #======================================================================#
            # Compute loss function
            # Get the current epidemiological state (can be useful to stratify evaluation)
            if (current_epi_state is None):
                if (self.parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                    current_epi_state = x[:, -4:] # Current state are one-hot encoded in the last 4 values of the feature vector

                elif (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                    first_curr_state_idx, last_curr_state_idx = self.patients_features_names_idx['current_state'][0], self.patients_features_names_idx['current_state'][1]
                    current_epi_state = x['Patient'][:, first_curr_state_idx:last_curr_state_idx] # Current state are one-hot encoded in the last 4 values of the feature vector
                
                elif (self.parameters_exp['dataset_name'].lower() == 'hug'):
                    first_curr_state_idx, last_curr_state_idx = self.patients_features_names_idx['State'][0], self.patients_features_names_idx['State'][1] # According to the Yaml files
                    current_epi_state = x['Patient'][:, first_curr_state_idx:last_curr_state_idx] # Current state are one-hot encoded in the last values 1 to 3 in the feature vector
                    
            # Compute loss functions
            # Clasificaiton loss              
            # The first argument of the forward pass of TemporalTransitionAwareLoss expect a tensor of shape [N, W, C] and epi_states_scores is of shape [N, C, W]
            loss_classif = self.criterion_states(epi_states_scores.permute(0, 2, 1), y, current_epi_state.argmax(dim=1)) 

            # States transition loss
            if (self.parameters_exp["predict_state_transitions"]):
                loss_transitions = self.criterion_transitions(transitions_scores, y_transitions)
                
            # Raise error if NaN values in loss_classif
            if (loss_classif.isnan().any()):
                raise RuntimeError("The value of loss_classif is NaN.")
            
            # Epidemio-informed loss
            if (self.parameters_exp['epidemio_informed']):
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):     
                    # Raise error if any NaN values in dX_dT_left or dX_dT_right
                    if (dX_dT_left.isnan().any()):
                        raise RuntimeError("NaN values in dX_dT_left")
                    if (dX_dT_right.isnan().any()):
                        raise RuntimeError("NaN values in dX_dT_right")
                                
                    if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                        #print("\n\n TODO: verify that the shapes of (dx_dt_left[0, :, i, :] and ds_dt_right are coherent with the input shapes expected by self.criterion_reg for all forecast horizon (1 and > 1)")
                        if (len(dX_dT_left.shape) == 3):
                            dX_dT_left = dX_dT_left.unsqueeze(dim=0)
                        # dX_dT_left is of shape (1, num_nodes, num_states, forecast_hor)

                        # Susceptible
                        loss_reg_S = self.criterion_reg(dX_dT_left[0, :, 0, :], dS_dT_right)
                        # Infectious
                        loss_reg_I = self.criterion_reg(dX_dT_left[0, :, 1, :], dI_dT_right)
                        # Recovered
                        loss_reg_R = self.criterion_reg(dX_dT_left[0, :, 2, :], dR_dT_right)
                    else:
                        if (len(dX_dT_left.shape) == 3):
                            dX_dT_left = dX_dT_left.unsqueeze(dim=0)
                        # Susceptible
                        loss_reg_S = self.criterion_reg(dX_dT_left[0, :, 0, :], dS_dT_right)
                        # Exposed
                        loss_reg_E = self.criterion_reg(dX_dT_left[0, :, 1, :], dE_dT_right)
                        # Infectious
                        loss_reg_I = self.criterion_reg(dX_dT_left[0, :, 2, :], dI_dT_right)
                        # Recovered
                        loss_reg_R = self.criterion_reg(dX_dT_left[0, :, 3, :], dR_dT_right)
                        # Deceased and Non-Susceptible for SEIRD-NS models
                        if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                            # Deceased
                            loss_reg_D = self.criterion_reg(dX_dT_left[:, :, 4, :], dD_dT_right)
                            # Non-Susceptible
                            loss_reg_NS = self.criterion_reg(dX_dT_left[:, :, 5, :], dNS_dT_right)
                   
                    # Total regularization loss
                    loss_S = self.parameters_exp['lambda_reg_S']*loss_reg_S
                    loss_I = self.parameters_exp['lambda_reg_I']*loss_reg_I
                    loss_R = self.parameters_exp['lambda_reg_R']*loss_reg_R
                    loss_reg = loss_S + loss_I + loss_R
                    if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                        loss_E = self.parameters_exp['lambda_reg_E']*loss_reg_E
                        loss_reg = loss_reg + loss_E
                    if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        loss_D = self.parameters_exp['lambda_reg_D']*loss_reg_D
                        loss_NS = self.parameters_exp['lambda_reg_NS']*loss_reg_NS
                        loss_reg = loss_reg + loss_D + loss_NS
                    # Raise error if NaN values in any of these losses
                    if (loss_S.isnan().any()):
                        raise RuntimeError("The value of loss_S is NaN.")
                    if (loss_I.isnan().any()):
                        raise RuntimeError("The value of loss_I is NaN.")
                    if (loss_R.isnan().any()):
                        raise RuntimeError("The value of loss_R is NaN.")
                    if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                        if (loss_E.isnan().any()):
                            raise RuntimeError("The value of loss_E is NaN.")
                    if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        if (loss_D.isnan().any()):
                            raise RuntimeError("The value of loss_D is NaN.")
                        if (loss_NS.isnan().any()):
                            raise RuntimeError("The value of loss_NS is NaN.")
                    
                elif (self.parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    # Epdemio-informed loss
                    if (self.parameters_exp['teacher_forcing_epidemio']):
                        # Permuting dims to be coherent with y_one_hot and epi_states_probs if forecast horizon >= 2
                        odes_next_states = odes_next_states.permute(1, 2, 0) 

                        # Computing y_one_hot if necessary
                        if (random_val_teacher_forcing < p_teacher_forcing):
                            # The labels of the samples (y) correspond to the FUTURE STATES
                            n_nodes, forecast_hor = y.shape
                            y_one_hot = []
                            for hor in range(forecast_hor):
                                y_one_hot.append(one_hot_encoding_np(y[:, hor].cpu().numpy(), num_classes=n_classes))
                            y_one_hot = torch.from_numpy(np.stack(y_one_hot, axis=2)).float().to(self.device)
                            loss_reg = self.criterion_reg(odes_next_states, y_one_hot)
                        else:
                            loss_reg = self.criterion_reg(odes_next_states, epi_states_probs)

                        # Raise error if NaN values in loss_reg
                        if (loss_reg.isnan().any()):
                            raise RuntimeError("The value of loss_reg is NaN.")
                else:
                    raise ValueError(f"\nEpdemio-informed approach {self.parameters_exp['epidemio_informed_method']} is not valid.\n")

            else:
                loss_reg = 0.0

            # Loss for the constraint on the transitions (only can be used for the case forecast_horizon >= 2)
            if (self.parameters_exp['forecast_horizon'] >= 2):
                if (self.parameters_exp["constrain_transitions_loss"]):
                    # IMPORTANT: We suppose that all the first states at the forecast window are always valid.
                    # Computing loss
                    loss_constr_transitions = self.compute_transition_constraint_loss(
                                                                                        logits=epi_states_scores,
                                                                                        y=y
                                                                                    )

                    # Raise error if NaN values in loss_constr_transitions
                    if (loss_constr_transitions.isnan().any()):
                        raise RuntimeError("The value of loss_constr_transitions is NaN.")
                
                if (self.parameters_exp["monotonicity_loss"]):
                    loss_mono = self.compute_monotonicity_penalty(logits=epi_states_scores)
    
            # Total loss
            # Classification loss
            total_loss = self.parameters_exp['lambda_classif_seir_states'] * loss_classif

            # Transitions loss
            if (self.parameters_exp["predict_state_transitions"]):
                total_loss = total_loss + self.parameters_exp["lambda_transitions"] * loss_transitions

            # Epidemio-informed part
            if (self.parameters_exp['epidemio_informed']):
                total_loss = total_loss + self.parameters_exp['lambda_reg'] * loss_reg

            # Constrain future states loss
            if (self.parameters_exp['forecast_horizon'] >= 2):
                # States constraints
                if (self.parameters_exp["constrain_transitions_loss"]):
                    total_loss = total_loss + self.parameters_exp["lambda_constrain_transitions_loss"] * loss_constr_transitions

                # Stability
                if (self.parameters_exp["monotonicity_loss"]):
                    total_loss = total_loss + self.parameters_exp["lambda_monotonicity_loss"] * loss_mono

            #======================================================================#
            #======================================================================#
            # Loss
            loss = {
                        "total_loss": total_loss, # DO NOT DO .item() AS WE ARE GOING TO DO THE BACKWARD PASS OUTSIDE THIS METHOD
                        "loss_classif": loss_classif.item(),
                    }
            if (self.parameters_exp["predict_state_transitions"]):
                loss["loss_transitions"] = loss_transitions.item()
            if (self.parameters_exp['epidemio_informed']):
                loss["loss_reg"] = loss_reg.item()
                if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    loss["loss_total_S"] = loss_S.item()
                    if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                        loss["loss_total_E"] = loss_E.item()
                    loss["loss_total_I"] = loss_I.item()
                    loss["loss_total_R"] = loss_R.item()
                    if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        loss["loss_total_D"] = loss_D.item()
                        loss["loss_total_NS"] = loss_NS.item()
            if (self.parameters_exp['forecast_horizon'] >= 2):
                if (self.parameters_exp["constrain_transitions_loss"]):
                    loss["loss_constr_transitions"] = self.parameters_exp["lambda_constrain_transitions_loss"] * loss_constr_transitions
                if (self.parameters_exp["monotonicity_loss"]):
                    loss["loss_monotonicity"] = self.parameters_exp["lambda_monotonicity_loss"] * loss_mono
            

            #======================================================================#
            #======================================================================#
            # Predictions
            preds = {
                        "true_epi_states": y,
                        "pred_epi_states_probs": epi_states_probs,
                        "current_epi_state": current_epi_state
                    }
            
            if (self.parameters_exp["predict_state_transitions"]):
                preds["true_transitions"] = y_transitions
                preds["pred_transitions_probs"] = transitions_probs
        else:
            raise ValueError()
        
        return loss, preds
    

    def get_nodes_embeddings(self):
        """
            Get the nodes embeddings obtained using a GNN encoder for all the samples in the datasets
        """
        raise NotImplementedError("Not implemented yet (see previous versions of the code to have an idea how to do it).")
                                            

        
    def singleTrain(self, rep_ID, reinitialize_model_weights=True):
        """
            Trains a model one time during self.n_epochs epochs
        """
        # Doing all on the hdf5 files result
        with h5py.File(self.repetitions_results_fn, 'a') as h5f_rep_results:
            # Initialization
            self.initSingleTrain(reinitialize_model_weights)

            # Data structures for the losses and predictions
            # Base group name
            if (self.using_multiple_DS):
                base_group_h5file = f"Rep-{rep_ID}_Dataset-{self.current_dataset_ID}"
            else:
                base_group_h5file = f"Rep-{rep_ID}"
            # Losses
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss")
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Train")
            if (not self.empty_val_ds):
                h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Val")
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Test")
            # Predictions
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds")
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train")
            if (not self.empty_val_ds):
                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val")
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test")

            # Epochs
            seen_loss_types = []
            last_done_epoch_saved = False
            for epoch in tqdm(range(self.n_epochs)):
                # Reinitialize memory
                self.reinitialize_memory_model()
                # Know if results saved in HDF5
                res_saved_hdf5 = False
                # Training
                self.test_mode = False
                for submodel in self.models:
                    self.models[submodel].train()
                tmp_train_losses = {}
                tmp_train_preds = []
                #for batch in self.train_loader:
                # IN THIS CASE, AT LEAST FOR THE SOCIOPATTERNS AND MURCIA DATASETS, ONE BATCH IS ONE SNAPSHOT
                batch_ID = 0 
                for batch in tqdm(self.train_loader):
                    # Compute forward pass
                    train_loss, train_preds = self.updateModel(batch, epoch, batch_ID=batch_ID)

                    # Detach model components (memory) from the computational graph for STM models (memory components)
                    if (self.parameters_exp['model_to_use'].lower().lower() in ["stm", "tgn"]):
                        self.models["model_node_enc"].detach()
            
                    # Detach recurrent weights from graph to prevent backprop through time
                    if (type(self.models["model_node_enc"]) == SEIRPredictor):
                        self.models["model_node_enc"].recurrent.weight.detach_()
                    elif (type(self.models["model_node_enc"]) == SEIRPredictorDeeper):
                        self.models["model_node_enc"].recurrent_1.weight.detach_()
                        self.models["model_node_enc"].recurrent_2.weight.detach_()
                        self.models["model_node_enc"].recurrent_3.weight.detach_()
                        self.models["model_node_enc"].recurrent_4.weight.detach_()
                        self.models["model_node_enc"].recurrent_5.weight.detach_()

                    # Add predictions and losses to list
                    if (train_loss is not None):
                        tmp_train_preds.append(train_preds)
                        for loss_type in train_loss:
                            if (loss_type not in tmp_train_losses):
                                tmp_train_losses[loss_type] = []
                            try:
                                tmp_train_losses[loss_type].append(train_loss[loss_type].detach().data.cpu().numpy())
                            except:
                                tmp_train_losses[loss_type].append(train_loss[loss_type])

                    batch_ID += 1

                # Mean performance in the epoch
                for loss_type in tmp_train_losses:
                    if (f"{base_group_h5file}/Loss/Train/{loss_type}" not in h5f_rep_results):
                        h5f_rep_results.create_dataset(f"{base_group_h5file}/Loss/Train/{loss_type}", shape=(self.n_epochs,), dtype='f')
                        seen_loss_types.append(loss_type)

                    # Raise error if NaN value in loss
                    if (any(np.isnan(a).any() for a in tmp_train_losses[loss_type])):
                        raise RuntimeError(f"There are NaN values for the loss {loss_type}")
                    h5f_rep_results[f"{base_group_h5file}/Loss/Train/{loss_type}"][epoch] = np.mean(tmp_train_losses[loss_type])
                # Preds
                if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.n_epochs-1):
                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}")
                    for day in range(len(tmp_train_preds)):
                        h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}/Day-{day}")
                        for res_name in tmp_train_preds[day]:
                            h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_train_preds[day][res_name].detach().cpu().numpy())
                    res_saved_hdf5 = True

                # Reinitialize memory
                self.reinitialize_memory_model()

                # Validation
                if (not self.empty_val_ds):
                    self.test_mode = False # As the validation set has the same characteristics as the training set
                    tmp_val_losses, tmp_val_preds = self.evalCurrentModel(self.val_loader, epoch)
                    # Loss
                    for loss_type in tmp_val_losses:
                            if (f"{base_group_h5file}/Loss/Val/{loss_type}" not in h5f_rep_results):
                                h5f_rep_results.create_dataset(f"{base_group_h5file}/Loss/Val/{loss_type}", shape=(self.n_epochs,), dtype='f')
                            h5f_rep_results[f"{base_group_h5file}/Loss/Val/{loss_type}"][epoch] = np.mean(tmp_val_losses[loss_type])

                    # Preds
                    if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.n_epochs-1):
                        if (not self.empty_val_ds):
                            h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}")
                            for day in range(len(tmp_val_preds)):
                                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}/Day-{day}")
                                for res_name in tmp_val_preds[day]:
                                    h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_val_preds[day][res_name].detach().cpu().numpy())

                # Test the model
                self.test_mode = True
                tmp_test_losses, tmp_test_preds = self.evalCurrentModel(self.test_loader, epoch)
                # Loss
                for loss_type in tmp_test_losses:
                    if (f"{base_group_h5file}/Loss/Test/{loss_type}" not in h5f_rep_results):
                        h5f_rep_results.create_dataset(f"{base_group_h5file}/Loss/Test/{loss_type}", shape=(self.n_epochs,), dtype='f')
                    h5f_rep_results[f"{base_group_h5file}/Loss/Test/{loss_type}"][epoch] = np.mean(tmp_test_losses[loss_type])
                # Preds
                if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.n_epochs-1):
                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}")
                    for day in range(len(tmp_test_preds)):
                        h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}/Day-{day}")
                        for res_name in tmp_test_preds[day]:
                            h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_test_preds[day][res_name].detach().cpu().numpy())

                print("================================================================================")
                print("LOSS AND METRICS\n")
                for loss_type in seen_loss_types:
                    print(f"\n=========> Loss type {loss_type} <=========")
                    print("\tTrain loss at epoch {} is {}".format(epoch, h5f_rep_results[f"{base_group_h5file}/Loss/Train/{loss_type}"][epoch]))
                    if (not self.empty_val_ds):
                        print("\t\tVal loss at epoch {} is {}".format(epoch, h5f_rep_results[f"{base_group_h5file}/Loss/Val/{loss_type}"][epoch]))
                    print("\t\tTest loss at epoch {} is {}".format(epoch, h5f_rep_results[f"{base_group_h5file}/Loss/Test/{loss_type}"][epoch]))
                # TODO: IMPLEMENT OTHER METRICS SUCH AS
                print("================================================================================\n\n")

                # Early stopping if asked
                if (not self.empty_val_ds):
                    if (self.parameters_exp['early_stopping']):
                        self.early_stopping(np.mean(tmp_val_losses['total_loss']), self.models)
                        if self.early_stopping.early_stop:
                            # Saving the current results if not done
                            if (not res_saved_hdf5):
                                # Train
                                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}")
                                for day in range(len(tmp_train_preds)):
                                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}/Day-{day}")
                                    for res_name in tmp_train_preds[day]:
                                        h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Train/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_train_preds[day][res_name].detach().cpu().numpy())
                                # Val
                                if (not self.empty_val_ds):
                                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}")
                                    for day in range(len(tmp_val_preds)):
                                        h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}/Day-{day}")
                                        for res_name in tmp_val_preds[day]:
                                            h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Val/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_val_preds[day][res_name].detach().cpu().numpy())

                                # Test
                                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}")
                                for day in range(len(tmp_test_preds)):
                                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}/Day-{day}")
                                    for res_name in tmp_test_preds[day]:
                                        h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/Test/Epoch-{epoch}/Day-{day}/{res_name}", data=tmp_test_preds[day][res_name].detach().cpu().numpy())

                            # Stopping
                            print(f"\n\n\n\n==========>Early stopping triggered AT EPOCH {epoch}.\n\n\n\n")
                            break 

    def singleTrainContinual(self, rep_ID):
        """
            Trains a model one time during self.n_epochs epochs
            on a continual training paradigm, where for each 
            timestep, we retrain the whole model and evaluate
            on the next step
        """
        losses = {'Train': {}, 'Val': {}, 'Test': {}}
        predictions = {'Train': {}, 'Val': {}, 'Test': {}}
        if (self.parameters_exp['epidemio_informed']):
            learned_epi_params_per_test_threshold_day = {} # Parameters learn per test threshold day
        for thresholds_IDs_dict_idx in range(len(self.list_data_splits_thresholds_IDs_dict)):
            data_splits_thresholds_IDs_dict = self.list_data_splits_thresholds_IDs_dict[thresholds_IDs_dict_idx]
            # Reinitialize the model weights JUST FOR THE FIRST MODEL IF ASKED
            if (not self.parameters_exp['reinitialize_model_weights']):
                if (thresholds_IDs_dict_idx == 0):
                    reinitialize_model_weights = True
                else:
                    reinitialize_model_weights = False

            # Update data spluts thresholds
            self.data_splits_thresholds_IDs_dict = data_splits_thresholds_IDs_dict

            # Do the single training
            self.singleTrain(rep_ID=rep_ID, reinitialize_model_weights=reinitialize_model_weights)

            # For the predictions ONLY KEEP THE LAST EPOCH PREDICTION OF THE PREVIOUS TRAIN
            with h5py.File(self.repetitions_results_fn, 'a') as h5f_rep_results:
                # Base group name
                if (self.using_multiple_DS):
                    base_group_h5file = f"Rep-{rep_ID}_Dataset-{self.current_dataset_ID}"
                else:
                    base_group_h5file = f"Rep-{rep_ID}"
            
                # Iterating over the splits
                for data_split in h5f_rep_results[base_group_h5file]['Loss']:
                    # Loss function
                    for loss_type in h5f_rep_results[base_group_h5file]['Loss'][data_split]:
                        if (loss_type not in losses[data_split]):
                            losses[data_split][loss_type] = []
                        losses[data_split][loss_type].extend(h5f_rep_results[f"{base_group_h5file}/Loss/{data_split}/{loss_type}"][:].tolist())

                    # For the predictions
                    epochs_list = [int(epoch_str.split('-')[-1]) for epoch_str in h5f_rep_results[base_group_h5file]['Preds'][data_split]]
                    last_epoch = max(epochs_list)
                    days_list = sorted([int(day_str.split('-')[-1]) for day_str in h5f_rep_results[base_group_h5file]['Preds'][data_split][f'Epoch-{last_epoch}']])
                    if (data_split == 'Test'):
                        days_list = [day+self.data_splits_thresholds_IDs_dict['init_test_step'] for day in days_list]
                    for day in days_list:
                        if (f'Day-{day}' not in predictions[data_split]): # Use true day
                            predictions[data_split][f'Day-{day}'] = {}
                        if (data_split == 'Test'):
                            # Remove test idx offset for correct indexint 
                            local_day = day - self.data_splits_thresholds_IDs_dict['init_test_step']
                        else:
                            local_day = day
                        for res_name in h5f_rep_results[base_group_h5file]['Preds'][data_split][f'Epoch-{last_epoch}'][f'Day-{local_day}']:
                            if (res_name not in predictions[data_split][f'Day-{day}']):
                                predictions[data_split][f'Day-{day}'][res_name] = h5f_rep_results[base_group_h5file]['Preds'][data_split][f'Epoch-{last_epoch}'][f'Day-{local_day}'][res_name][:]


                # Erasing the created groups and datasets in the HDF5 file for the next iteration
                del h5f_rep_results[f"{base_group_h5file}/Loss"]
                del h5f_rep_results[f"{base_group_h5file}/Preds"]

            # Learned parameters
            if (self.parameters_exp['epidemio_informed']):
                test_treshold_day = self.mapping_thresholds_IDs_list_to_test_day[thresholds_IDs_dict_idx]
                learned_epi_params_per_test_threshold_day[test_treshold_day] = self.models['model_epi_params'].epi_params.clone().detach().numpy()

        # Write the new HDF5 file
        with h5py.File(self.repetitions_results_fn, 'a') as h5f_rep_results:
            # Losses
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss")
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Train")
            if (not self.empty_val_ds):
                h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Val")
            h5f_rep_results.create_group(f"{base_group_h5file}/Loss/Test")
            # Predictions
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds")
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Train")
            if (not self.empty_val_ds):
                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Val")
            h5f_rep_results.create_group(f"{base_group_h5file}/Preds/Test")
            # Learned parameters
            if (self.parameters_exp['epidemio_informed']):
                h5f_rep_results.create_group(f"{base_group_h5file}/EpiParams")
                for test_treshold_day in learned_epi_params_per_test_threshold_day:
                    h5f_rep_results.create_dataset(f"{base_group_h5file}/EpiParams/TestThresholdDay-{test_treshold_day}", data=learned_epi_params_per_test_threshold_day[test_treshold_day])
            # Write data
            # Losses and predictions
            for data_split in losses:
                # Losses
                for loss_type in losses[data_split]:
                    h5f_rep_results.create_dataset(f"{base_group_h5file}/Loss/{data_split}/{loss_type}", data=losses[data_split][loss_type], dtype='f') 
            
                # Predictions
                h5f_rep_results.create_group(f"{base_group_h5file}/Preds/{data_split}/Epoch-Last")
                for day_str in predictions[data_split]:
                    h5f_rep_results.create_group(f"{base_group_h5file}/Preds/{data_split}/Epoch-Last/{day_str}")
                    for res_name in predictions[data_split][day_str]:
                        h5f_rep_results.create_dataset(f"{base_group_h5file}/Preds/{data_split}/Epoch-Last/{day_str}/{res_name}", data=predictions[data_split][day_str][res_name])


    def evalCurrentModel(self, dataloader, epoch):
        # Reinitialize memory if it is a memory based model
        self.reinitialize_memory_model()

        # Evaluation
        for submodel in self.models:
            self.models[submodel].eval()
        tmp_losses = {}
        tmp_preds = []
        # IMPORTANT: HERE WE DO NOT USE WITH TORC NO GRAD AS WE NEED TO COMPUTE THE GRADIENTS FOR THE VALIDATION AND TEST LOSSES FOR EPIDEMIO-INFORMED MODELS
        batch_ID = 0
        for batch in tqdm(dataloader):
            # Forward pass
            loss, preds = self.computeForwardPass(batch, epoch, batch_ID=batch_ID)


            # Detach model components (memory) from the computational graph for STM models (memory components)
            if (self.parameters_exp['model_to_use'].lower().lower() in ["stm", "tgn"]):
                self.models["model_node_enc"].detach()
    
            # Detach recurrent weights from graph to prevent backprop through time
            if (type(self.models["model_node_enc"]) == SEIRPredictor):
                self.models["model_node_enc"].recurrent.weight.detach_()
            elif (type(self.models["model_node_enc"]) == SEIRPredictorDeeper):
                self.models["model_node_enc"].recurrent_1.weight.detach_()
                self.models["model_node_enc"].recurrent_2.weight.detach_()
                self.models["model_node_enc"].recurrent_3.weight.detach_()
                self.models["model_node_enc"].recurrent_4.weight.detach_()
                self.models["model_node_enc"].recurrent_5.weight.detach_()

            # Add predictions and losses to list
            if (preds is not None):
                tmp_preds.append(preds)
                for loss_type in loss:
                    if (loss_type not in tmp_losses):
                        tmp_losses[loss_type] = []
                    try:
                        tmp_losses[loss_type].append(loss[loss_type].detach().data.cpu().numpy())
                    except:
                        tmp_losses[loss_type].append(loss[loss_type])
            batch_ID += 1

        # Seing the transitions performance
        # Permute preds['pred_epi_states_probs'] to be of shape [N, W, C]
        print(f"\n\n\n=========> Transitions performance at epoch {epoch}:")
        true_epi_states_batch = np.concatenate([tmp_preds[i]['true_epi_states'].cpu().detach().numpy() for i in range(len(tmp_preds))], axis=0)
        pred_epi_states_probs_batch = np.concatenate([tmp_preds[i]['pred_epi_states_probs'].permute(0, 2, 1).cpu().detach().numpy() for i in range(len(tmp_preds))], axis=0)
        current_epi_state_batch = np.concatenate([tmp_preds[i]['current_epi_state'].argmax(dim=1).cpu().detach().numpy() for i in range(len(tmp_preds))], axis=0)
        if (self.parameters_exp['forecast_horizon'] == 2):
            true_epi_states_batch = true_epi_states_batch.squeeze(axis=2)
        transitions_metrics = compute_transition_metrics(
                                                            true_states=true_epi_states_batch,
                                                            pred_probs=pred_epi_states_probs_batch,
                                                            current_states=current_epi_state_batch
                                                        )
        check_prediction_bias(pred_epi_states_probs_batch, current_epi_state_batch)
        for key in transitions_metrics:
            print(f"\n\t=========> {key}: {transitions_metrics[key]}")
        print("\n\n\n")

        self.optimizer.zero_grad()

        return tmp_losses, tmp_preds
    
    
def modify_parameters_exp_from_optuna(parameters_exp, best_params):
    # Learning rate
    # Search LR in OPTUNA for the NEW EXPERIMENT
    if ('optuna_search_all_lrs' not in parameters_exp): 
        if ('use_optuna' not in parameters_exp):
            parameters_exp['use_optuna'] = False 
            parameters_exp['optuna_search_all_lrs'] = False
        else:
            if (parameters_exp['use_optuna']):
                parameters_exp['optuna_search_all_lrs'] = True
                #parameters_exp['optuna_search_all_lrs'] = False
            else:
                parameters_exp['optuna_search_all_lrs'] = False
    # Update learning rates
    try:
        parameters_exp["lr"] = best_params["lr"]
    except:
        parameters_exp["lr"] = best_params["learning_rate"]
    try:
        parameters_exp["lr_model_compart_pred"] = best_params["lr_model_compart_pred"]
    except:
        parameters_exp["lr_model_compart_pred"] = parameters_exp["lr"]
    try:
        parameters_exp["lr_model_epi_params"] = best_params["lr_model_epi_params"]
    except:
        parameters_exp["lr_model_epi_params"] = parameters_exp["lr"]
    lrs = {
                    'model_node_enc': parameters_exp['lr'],
                    'model_compart_pred': parameters_exp['lr_model_compart_pred'],
                    'model_epi_params': parameters_exp['lr_model_epi_params']
                }
    
    if (parameters_exp["predict_state_transitions"]):
        try:
            parameters_exp["lr_model_transitions_pred"] = best_params["lr_model_transitions_pred"]
        except:
            parameters_exp["lr_model_transitions_pred"] = parameters_exp["lr"]
        if ("lambda_transitions"):
            parameters_exp["lambda_transitions"] = best_params["lambda_transitions"]
    
    # Other params
    #parameters_exp["weight_decay"] = best_params["weight_decay", 1e-5, 1e-1, log=True)
    #parameters_exp["optimizer"] = best_params["optimizer", ["Adam", "AdamW"])
    if (parameters_exp['epidemio_informed']):
        #if ("teacher_forcing_epidemio" in best_params):
            #parameters_exp["teacher_forcing_epidemio"] = best_params["teacher_forcing_epidemio", [True, False])
        if (parameters_exp["teacher_forcing_epidemio"]):
            if ("ramp_epochs_teacher_forcing_sched" in best_params):
                parameters_exp["ramp_epochs_teacher_forcing_sched"] = best_params["ramp_epochs_teacher_forcing_sched"]
        if ("lambda_reg" in best_params):
            parameters_exp["lambda_reg"] = best_params["lambda_reg"]
        if (parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
            if ("lambda_reg_S" in best_params):
                parameters_exp["lambda_reg_S"] = best_params["lambda_reg_S"]
            if (parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                if ("lambda_reg_E" in best_params):
                    parameters_exp["lambda_reg_E"] = best_params["lambda_reg_E"]
            if ("lambda_reg_I" in best_params):
                parameters_exp["lambda_reg_I"] = best_params["lambda_reg_I"]
            if ("lambda_reg_R" in best_params):
                parameters_exp["lambda_reg_R"] = best_params["lambda_reg_R"]
            if (parameters_exp['dataset_name'].lower() == 'murcia'):
                if ("lambda_reg_D" in best_params):
                    parameters_exp["lambda_reg_D"] = best_params["lambda_reg_D"]
                if ("lambda_reg_NS" in best_params):
                    parameters_exp["lambda_reg_NS"] = best_params["lambda_reg_NS"]

    # Other parameters for training when the prediciton horizon is greater than 1
    if ("lambda_classif_seir_states" in best_params):
        parameters_exp['lambda_classif_seir_states'] = best_params["lambda_classif_seir_states"]
    if (parameters_exp['constrain_transitions_loss']):
        if ("lambda_constrain_transitions_loss" in best_params):
            parameters_exp['lambda_constrain_transitions_loss'] = best_params["lambda_constrain_transitions_loss"]
    if (parameters_exp['monotonicity_loss']):
        if ("lambda_monotonicity_loss" in best_params):
            parameters_exp['lambda_monotonicity_loss'] = best_params["lambda_monotonicity_loss"]


    # Define per-model hyper-parameters to tune
    # IMPORTANT: ARCHITECTURES HYPER-PARAMS ARE ONLY FIXED FOR NOT EPIDEMIO-INFORMED MODELS
    if (parameters_exp['model_to_use'].lower() == "stm"):
        if ("hidden_channels" in best_params):
            tmp_hidden_channels = best_params["hidden_channels"]
            parameters_exp["hidden_channels"] = tmp_hidden_channels

            parameters_exp["out_channels_node_encoder"] = tmp_hidden_channels//2
            parameters_exp["args_temporal"]["hidden_channels"] = tmp_hidden_channels
            parameters_exp["args_temporal"]["out_channels"] = tmp_hidden_channels

            parameters_exp["args_spatial"]["hidden_channels"] = tmp_hidden_channels
            parameters_exp["args_spatial"]["out_channels"] = tmp_hidden_channels
        if ("dropout" in best_params):
            tmp_dropout = best_params["dropout"]
            parameters_exp["dropout"] = tmp_dropout
            parameters_exp["args_temporal"]["dropout"] = tmp_dropout
            parameters_exp["args_spatial"]["dropout"] = tmp_dropout
        if ("num_layers" in best_params):
            tmp_num_layers = best_params["num_layers"]
            parameters_exp["args_temporal"]["num_layers"] = tmp_num_layers
            parameters_exp["args_spatial"]["num_layers"] = tmp_num_layers
        if ("heads" in best_params):
            tmp_heads = best_params["heads"]
            parameters_exp["args_temporal"]["heads"] = tmp_heads
            parameters_exp["args_spatial"]["heads"] = tmp_heads
        if ("time_dim" in best_params):
            parameters_exp["time_dim"] = best_params["time_dim"]
    
    elif (parameters_exp['model_to_use'].lower() == "tgn"):
        if ("hidden_channels" in best_params):
            parameters_exp["hidden_channels"] = best_params["hidden_channels"]
            parameters_exp["time_dim"] = parameters_exp["hidden_channels"]
            parameters_exp["out_channels_node_encoder"] = parameters_exp["hidden_channels"]
            parameters_exp['memory_dim'] = parameters_exp["out_channels_node_encoder"]
        if ("dropout" in best_params):
            parameters_exp['dropout'] = best_params["dropout"]
        if ("dim_enc_nodes_feaures" in best_params):
            parameters_exp['dim_enc_nodes_feaures'] = best_params["dim_enc_nodes_feaures"]
        
    elif (parameters_exp['model_to_use'].lower() == "gat"):
        if ("hidden_channels" in best_params):
            parameters_exp["hidden_channels"] = best_params["hidden_channels"]
        if ("heads" in best_params):
            parameters_exp["heads"] = best_params["heads"]
        if ("dropout" in best_params):
            parameters_exp["dropout"] = best_params["dropout"]
        if ("num_layers_encoder" in best_params):
            parameters_exp["num_layers_encoder"] = best_params["num_layers_encoder"]
        if ("out_channels_node_encoder" in best_params):
            parameters_exp["out_channels_node_encoder"] = best_params["out_channels_node_encoder"]
    
    elif (parameters_exp['model_to_use'].lower() in ["mlp", "gcn", "graphsage"]):
        if ("hidden_channels" in best_params):
            parameters_exp["hidden_channels"] = best_params["hidden_channels"]
        if ("num_layers_encoder" in best_params):
            parameters_exp["num_layers_encoder"] = best_params["num_layers_encoder"]
        if ("dropout" in best_params):
            parameters_exp["dropout"] = best_params["dropout"]
        if ("out_channels_node_encoder" in best_params):
            parameters_exp["out_channels_node_encoder"] = best_params["out_channels_node_encoder"]
    
    elif (parameters_exp['model_to_use'].lower() == "simplegnn"):
        if ("hidden_channels" in best_params):
            parameters_exp["hidden_channels"] = best_params["hidden_channels"]
        if ("out_channels_node_encoder" in best_params):
            parameters_exp["out_channels_node_encoder"] = best_params["out_channels_node_encoder"]
    
    elif (parameters_exp['model_to_use'].lower() == "seirpredictor"):
        if ("out_channels_node_encoder" in best_params):
            parameters_exp["out_channels_node_encoder"] = best_params["out_channels_node_encoder"]
    
    elif (parameters_exp['model_to_use'].lower() == "seirpredictordeeper"):
        if ("hidden_channels" in best_params):
            parameters_exp["hidden_channels"] = best_params["hidden_channels"]
        if ("out_channels_node_encoder" in best_params):
            parameters_exp["out_channels_node_encoder"] = best_params["out_channels_node_encoder"]
    else:
        raise ValueError(f"\nModel to use {parameters_exp['model_to_use'].lower()} is not valid for Optuna hyper-parameter tuning\n.")

    return parameters_exp





#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    #==========================================================================#
    # IGNOEING WARNING
    import warnings
    warnings.filterwarnings("ignore")
    #==========================================================================#
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    #seed = 42
    # seed = 7
    seed = 1
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True, warn_only=False) # For reproducibility purposes
    torch.use_deterministic_algorithms(True, warn_only=True) # For reproducibility purposes
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--parameters_file', required=True, help="Parameters for the experiment", type=str)
    ap.add_argument('--use_debug', help="Activate to access some specific breakpoints in the code for debug.", action="store_true")
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    parameters_file_original = parameters_file
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)

    # Debug mode
    use_debug = args['use_debug']

    # Verifying some important values that should be in parameters_exp
    if ('use_optuna' not in parameters_exp):
        parameters_exp['use_optuna'] = False

    #==========================================================================#
    # Creating an instance of the experiment
    exp = InfectionRiskPred(parameters_exp, use_debug=use_debug)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H%M%S")
    # If execution is from src
    resultsFolder = './results/InfectionRiskPred/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.makedirs(resultsFolder, exist_ok=True)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Dataset Loading
    if ('continual_training' not in parameters_exp) or (not parameters_exp['continual_training']):
        exp.createTorchDatasets()

    # Creating directories for the trained models, the training and testing metrics
    # and the parameters of the model (i.e. the training parameters and the network
    # architecture)
    os.mkdir(resultsFolder + '/model/')
    os.mkdir(resultsFolder + '/params_exp/')
    os.mkdir(resultsFolder + '/metrics/')

    # Normalizing the dataset
    #exp.computeDatasetMeanStd()
    #exp.normalizeDataset()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.yaml')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.yaml'
    try:
        with open(parameters_file, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(parameters_exp, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Optuna or classical holdout
    if (parameters_exp['use_optuna']):
        exp.optuna_tunning() 
    else:
    # Doing holdout evaluation
        if (len(parameters_exp['multiple_hdf5_dataset_filenames']) > 1):
            exp.repeatedHoldoutMultipleDS(save_results=True)
            #exp.repeatedHoldoutMultipleDS(save_results=False)
        else:
            exp.holdoutTrain(save_results=True)
            #exp.holdoutTrain(save_results=False)

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.yaml')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.yaml'
    # To avoid format problems with YAML for some Python variables, we are going to transform them into str
    final_params_to_save = deepcopy(exp.parameters_exp)
    final_params_to_save['metadata'] = str(final_params_to_save['metadata'])
    try: # For Murcia and HUG datasets
        final_params_to_save['edge_attr_dims'] = str(final_params_to_save['edge_attr_dims'])
    except: # For SocioPattern dataset
        pass
    # Save into YAML
    try:
        with open(parameters_file, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(final_params_to_save, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    #==========================================================================#
    # Writing or modifying params files if Optuna was used
    if ('use_optuna' in parameters_exp):
        if (parameters_exp['use_optuna']):
            # Get prediction horizon
            forecast_horizon = exp.parameters_exp['forecast_horizon']

            # Save original files if not already done (the ones I wrote)
            base_params_folder = '/'.join(parameters_file_original.split('/')[:-2])
            if (not os.path.exists(base_params_folder + "_ORIGINAL")):
                shutil.copytree(base_params_folder, base_params_folder + "_ORIGINAL")

            # Getting the parameters of the best trial
            best_params = exp.study.best_params

            # Create the names of the new files to where the results are going to be stored 
            if ("NotEpidemioInformed_Optuna" in parameters_file_original):
                files_to_modify = [
                                    base_params_folder + "/NotEpidemioInformed/NotEpidemioInformed.json",
                                    base_params_folder + "/NotEpidemioInformed/NotEpidemioInformedContinual.json",
                                    base_params_folder + "/EpidemioInformed/AutoDiffWrtTimeReg_Optuna.json",
                                    base_params_folder + "/EpidemioInformed/AutoDiffWrtTimeReg.json",
                                    base_params_folder + "/EpidemioInformed/AutoDiffWrtTimeRegContinual.json",
                                    base_params_folder + "/EpidemioInformed/ODEsNextStatePredRed_Optuna.json",
                                    base_params_folder + "/EpidemioInformed/ODEsNextStatePredRed.json",
                                    base_params_folder + "/EpidemioInformed/ODEsNextStatePredRedContinual.json",
                                  ]
                
            elif ("AutoDiffWrtTimeReg_Optuna" in parameters_file_original):
                files_to_modify = [
                                    base_params_folder + "/EpidemioInformed/AutoDiffWrtTimeReg.json",
                                    base_params_folder + "/EpidemioInformed/AutoDiffWrtTimeRegContinual.json",
                                  ]
            elif ("ODEsNextStatePredRed_Optuna" in parameters_file_original):
                files_to_modify = [
                                    base_params_folder + "/EpidemioInformed/ODEsNextStatePredRed.json",
                                    base_params_folder + "/EpidemioInformed/ODEsNextStatePredRedContinual.json",
                                  ]
            

            # Modifying necessary files
            for file_to_modify in files_to_modify:
                print(f"\n\n =========> MODIFYING FILE {file_to_modify} <=========")
                try:
                    # Open file
                    with open(file_to_modify) as jf:
                        tmp_new_params = json.load(jf)

                    # Modify the parameters
                    tmp_new_params = modify_parameters_exp_from_optuna(tmp_new_params, best_params)

                    # Modify if continual
                    if ('continual' in file_to_modify.lower()):
                        tmp_new_params['continual_training'] = True
                        tmp_new_params['nb_repetitions'] = 1

                    # Store the new values in the same file
                    with open(file_to_modify, "w") as jf:
                        json.dump(tmp_new_params, jf)
                except Exception as e:
                    print(f"\n\nProblem with file {file_to_modify}, file not modified. Error: {e}\n\n")


    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()