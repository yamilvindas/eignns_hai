#!/usr/bin/env python3
"""
    This code implements some basic classes and functions for a generic deep
    learning experiment
    This code CANNOT BE USE BY ITSELF, as there is no main function because
    the loss function to use will depend on what we want to do, so it cannot
    be generic.
"""
import os
import re
import yaml
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from pathlib import Path

import h5py

import pandas as pd

import numpy as np


import torch

import optuna

from src.Utils.GCE import GeneralizedCrossEntropy
from src.Utils.plot_metrics import get_metrics_per_rep, evaluate_infection_onset
from src.Models.modules import SEIRPredictorDeeper,\
                               SimpleGNN,\
                               SEIRPredictor,\
                               GraphSageModule,\
                               GATModule,\
                               GCNModule,\
                               init_weights
from src.Models.TGN import TGN, MLPMessage, MeanAggregator
from src.Models.STM import STM
    

# Mapping of the states
SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
INV_MAPPING = {v: k for k, v in MAPPING.items()}

# Class for early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
            GENERATED WITH CHAT-GPT

            Parameters:
            -----------
            patience: int
                How many epochs to wait after last improvement.
            min_delta: float
                Minimum change in the monitored metric to qualify as improvement.
            restore_best_weights: bool
                Whether to restore the model weights from the best epoch.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dicts = {}

    def __call__(self, val_loss, models):
        score = -val_loss  # since lower loss is better

        if self.best_score is None:
            self.best_score = score
            for submodel in models:
                self.best_state_dicts[submodel] = models[submodel].state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    for submodel in models:
                        models[submodel].load_state_dict(self.best_state_dicts[submodel])
        else:
            self.best_score = score
            for submodel in models:
                self.best_state_dicts[submodel] = models[submodel].state_dict()
            self.counter = 0

# Abstract class
class GenericExperiment(ABC):
    def __init__(self, parameters_exp, use_debug=False):
        """
            Class for a generic DL experiment (train, validate, test
            a certain model on a given dataset)

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment.
            use_debug: bool
                Activate to access some specific breakpoints in the code for debug.
        """
        # Defining some attributes of the experiment
        self.exp_id = parameters_exp['exp_id']
        self.results_folder = None

        # Dataset type
        # Main dataset
        if ('dataset_name' not in parameters_exp):
            parameters_exp['dataset_name'] = 'SocioPatterns'
        self.dataset_name = parameters_exp['dataset_name']
        # Subdataset
        if ('subdataset' not in parameters_exp):
            parameters_exp['subdataset'] = None
        self.subdataset = parameters_exp['subdataset']

        # HDF5 file name
        if ('hdf5_dataset_filename' not in parameters_exp):
            if ('multiple_hdf5_dataset_filenames' not in parameters_exp):
                raise RuntimeError("\nAt least one of 'hdf5_dataset_filename' or 'multiple_hdf5_dataset_filenames' should be specified in the parameters file\n.")
            else:
                parameters_exp['hdf5_dataset_filename'] =  parameters_exp['multiple_hdf5_dataset_filenames'][0]
        self.hdf5_dataset_filename = parameters_exp['hdf5_dataset_filename']

        # Get right path for the data (depends if local or clusteer execution)
        if (not os.path.exists(self.hdf5_dataset_filename)):
            raise RuntimeError(f"\nThe HDF5 dataset file {self.hdf5_dataset_filename} does not exist.\n")
            

        self.h5_file = h5py.File(self.hdf5_dataset_filename, 'r')
        if ('multiple_hdf5_dataset_filenames' not in parameters_exp):
            parameters_exp['multiple_hdf5_dataset_filenames'] = [self.hdf5_dataset_filename]
        # Using several datasets or just one (important to save files)
        if (len(parameters_exp['multiple_hdf5_dataset_filenames']) > 1):
            self.using_multiple_DS = True 
        else:
            self.using_multiple_DS = False

        # Forecast horizon for SocioPattern (computed dynamically)
        if (parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            if ('forecast_horizon' not in parameters_exp):
                parameters_exp['forecast_horizon'] = 1
        elif (parameters_exp['dataset_name'].lower() in ['murcia', 'hug']):
            forecast_horizon_in_name = False
            split_hdf5_ds_name = parameters_exp['hdf5_dataset_filename'].lower().split('_')
            for el in split_hdf5_ds_name:
                if ('forecasthorizon' in el):
                    tmp_forecast_horizon = int(el.split('forecasthorizon-')[-1])
                    forecast_horizon_in_name = True
            if (not forecast_horizon_in_name):
                parameters_exp['forecast_horizon'] = 1
                print(f"\n\nWARNING: No forecast horizon was found in the name of the Murcia HDF5 datasets, using a FORECAST HORIZON OF 1 by default.\n\n")
            else:
                if ('forecast_horizon' in parameters_exp):
                    if (parameters_exp['forecast_horizon'] != tmp_forecast_horizon):
                        raise RuntimeError(f"The forecast_horizon value given in the parameters file ({parameters_exp['forecast_horizon']}) is not consisten with that of the Murcia HDF5 files ({tmp_forecast_horizon})")
                else:
                    parameters_exp['forecast_horizon'] = tmp_forecast_horizon

        # Continual training
        if ('continual_training' not in parameters_exp):
            parameters_exp['continual_training'] = False
        
        # Reinitialize model weights for CL
        if (parameters_exp['continual_training']):
            if ('reinitialize_model_weights' not in parameters_exp):
                parameters_exp['reinitialize_model_weights'] = False
        else:
            parameters_exp['reinitialize_model_weights'] = True # IN THIS CASE WE ALWAYS WANT TO REINITIALIZE THE MODEL WEIGHTS FOR EACH REPETITION


        # Other parameters for datasets
        if (parameters_exp['dataset_name'].lower() == 'murcia'):
            # Caracteristics of the DS
            is_fully_connected = parameters_exp['hdf5_dataset_filename'].split('/')[-1].split('_')[1].split('-')[1]
            if (is_fully_connected.lower() == 'true'):
                is_fully_connected = True
            elif (is_fully_connected.lower() == 'false'):
                is_fully_connected = False
            else:
                raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_fully_connected})\n")
            is_uniform_patTopat_edges = parameters_exp['hdf5_dataset_filename'].split('/')[-1].split('_')[2].split('-')[1]
            if (is_uniform_patTopat_edges.lower() == 'true'):
                is_uniform_patTopat_edges = True
            elif (is_uniform_patTopat_edges.lower() == 'false'):
                is_uniform_patTopat_edges = False
            else:
                raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_uniform_patTopat_edges})\n")

            # Nodes and edge types
            node_types = ["Patient", "Place"]
            edge_types = [('Patient', 'to', 'Place'), ('Place', 'to', 'Patient')]
            edge_attr_dims = {('Patient', 'to', 'Place'): 1, ('Place', 'to', 'Patient'): 1}
            if (is_fully_connected):
                edge_types.append(('Patient', 'to', 'Patient'))
                edge_attr_dims[('Patient', 'to', 'Patient')] = 1
            parameters_exp['metadata'] = (node_types, edge_types)
            parameters_exp['edge_attr_dims'] = edge_attr_dims
            
            # Position of each feature in the feature vector for patients and places nodes
            fn_patient_features_names_idx = "/".join(self.hdf5_dataset_filename.split("/")[:-1]) + "/patient_features_names_idx_0.pkl"
            with open(fn_patient_features_names_idx, mode='rb') as pf:
                self.patients_features_names_idx = pickle.load(pf)
            fn_place_features_names_idx = "/".join(self.hdf5_dataset_filename.split("/")[:-1]) + "/place_features_names_idx_0.pkl"
            with open(fn_place_features_names_idx, mode='rb') as pf:
                self.places_features_names_idx = pickle.load(pf)


        elif (parameters_exp['dataset_name'].lower() == 'hug'):
            # Caracteristics of the DS
            is_fully_connected = parameters_exp['hdf5_dataset_filename'].split('/')[-1].split('_')[4].split('-')[1]
            if (is_fully_connected.lower() == 'true'):
                is_fully_connected = True
            elif (is_fully_connected.lower() == 'false'):
                is_fully_connected = False
            else:
                raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_fully_connected})\n")
            is_uniform_patTopat_edges = parameters_exp['hdf5_dataset_filename'].split('/')[-1].split('_')[5].split('-')[1].split('.hdf5')[0]
            if (is_uniform_patTopat_edges.lower() == 'true'):
                is_uniform_patTopat_edges = True
            elif (is_uniform_patTopat_edges.lower() == 'false'):
                is_uniform_patTopat_edges = False
            else:
                raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_uniform_patTopat_edges})\n")
            
            # Nodes and edge types
            node_types = ["Patient", "Place"]
            edge_types = [('Patient', 'to', 'Place'), ('Place', 'to', 'Patient')]
            edge_attr_dims = {('Patient', 'to', 'Place'): 1, ('Place', 'to', 'Patient'): 1}
            if (is_fully_connected):
                edge_types.append(('Patient', 'to', 'Patient'))
                edge_attr_dims[('Patient', 'to', 'Patient')] = 1
            parameters_exp['metadata'] = (node_types, edge_types)
            parameters_exp['edge_attr_dims'] = edge_attr_dims

            # Position of each feature in the feature vector for patients and places nodes
            fn_patients_features_names_idx = "/".join(self.hdf5_dataset_filename.split("/")[:-1]) + "/patients_features_names_idx_0.yaml"
            with open(fn_patients_features_names_idx, 'r') as file:
                self.original_patients_features_names_idx = yaml.safe_load(file)
                self.patients_features_names_idx = deepcopy(self.original_patients_features_names_idx)
            fn_places_features_names_idx = "/".join(self.hdf5_dataset_filename.split("/")[:-1]) + "/places_features_names_idx_0.yaml"
            with open(fn_places_features_names_idx, 'r') as file:
                self.original_places_features_names_idx = yaml.safe_load(file)
                self.places_features_names_idx = deepcopy(self.original_places_features_names_idx)

            # Thresholds for the train and test sets
            # Re-constructing the list_node_timestamps_dicts for data_splits_thresholds_IDs_dict
            tmp_h5 = h5py.File(self.hdf5_dataset_filename, 'r')
            self.list_node_timestamps_dicts = []
            sorted_step_IDs = sorted([int(step_ID) for step_ID in tmp_h5['NodeTimestampsDicts'] ])
            sorted_step_IDs = [str(step_ID) for step_ID in sorted_step_IDs]
            all_dates_str = set()
            for step_ID in tqdm(sorted_step_IDs):
                nodes_timestamps_dicts = {}
                for node_type in tmp_h5['NodeTimestampsDicts'][step_ID]:
                    if (node_type not in nodes_timestamps_dicts):
                        nodes_timestamps_dicts[node_type] = []
                    for tmp_date_str in tmp_h5['NodeTimestampsDicts'][step_ID][node_type]:
                        nodes_timestamps_dicts[node_type].append(pd.to_datetime(tmp_date_str.decode('utf-8')).date())
                        all_dates_str.add(tmp_date_str)
                # Sorting nodes_timestamps_dicts by increasing dates
                for node_type in nodes_timestamps_dicts:
                    nodes_timestamps_dicts[node_type] = sorted(nodes_timestamps_dicts[node_type])
                # Convertig back into str
                for node_type in nodes_timestamps_dicts:
                    nodes_timestamps_dicts[node_type] = [tmp_date.strftime('%Y-%m-%d') for tmp_date in nodes_timestamps_dicts[node_type]]
                # Add to the list
                self.list_node_timestamps_dicts.append(nodes_timestamps_dicts)
            all_dates_str = list(all_dates_str)
                
            # Get the dictionary with the thresholds for testing
            if ('threshold_train_date' not in parameters_exp):
                threshold_train_date = "2020-04-07" # 827 positive samples in the Test set ==> Selected to have confidence interval of 95% with a half-width of d = 0.03
                parameters_exp['threshold_train_date'] = threshold_train_date
            if ('threshold_val_date' not in parameters_exp):
                threshold_val_date = None
                parameters_exp['threshold_val_date'] = threshold_val_date


            # For continual learning
            if (parameters_exp['continual_training']):
                # Continual training datasets
                test_days = []

                # Sorting by increasing order all the possible dates
                all_dates_datetimes = [pd.to_datetime(tmp_date.decode('utf-8')) for tmp_date in all_dates_str]
                sorted_idx = np.argsort(all_dates_datetimes)
                all_dates_str = [all_dates_str[i] for i in sorted_idx]

                # Dates to use for test
                date_start_eval = parameters_exp['threshold_train_date']
                date_start_eval = pd.to_datetime(date_start_eval).date()
                for day_datetime in all_dates_datetimes:
                    if (day_datetime.date() >= date_start_eval):
                        test_days.append(day_datetime.date())


                # Getting the thresholds IDs between train and test datasets for all the couples of splits
                thresholds_IDs_list = []
                self.mapping_thresholds_IDs_list_to_test_day = {}
                for test_treshold_day in test_days:
                    # Create train and test datasets
                    # Get the threshold step ID between train and test
                    for step_ID in range(len(self.list_node_timestamps_dicts)):
                        tmp_date_str = self.list_node_timestamps_dicts[step_ID]['Patient'][0]
                        if (pd.to_datetime(tmp_date_str).date() <= test_treshold_day): # <= because in Python, when doing selection in a list, the last idx is excluded
                            threshold_step_ID = step_ID
                    thresholds = {
                                    "init_train_step": 0,
                                    "last_train_step": threshold_step_ID,
                                    "init_val_step": None,
                                    "last_val_step": None,
                                    "init_test_step": threshold_step_ID,
                                    "last_test_step": threshold_step_ID + 1,
                                    "Date": test_treshold_day.strftime('%Y-%m-%d') 
                                }
                    thresholds_IDs_list.append(thresholds)
                    self.mapping_thresholds_IDs_list_to_test_day[len(thresholds_IDs_list)-1] = test_treshold_day

                # Getting a list of data_splits_thresholds_IDs_dict for each dataset for continual training
                self.list_data_splits_thresholds_IDs_dict = []
                for thresholds_IDs_list_idx in tqdm(range(len(thresholds_IDs_list))):
                    thresholds_IDs = thresholds_IDs_list[thresholds_IDs_list_idx]
                    # Getting the thresholds IDs
                    data_splits_thresholds_IDs_dict = {
                                                            "init_train_step": thresholds_IDs['init_train_step'],
                                                            "last_train_step": thresholds_IDs['last_train_step'],
                                                            "init_val_step": thresholds_IDs['init_val_step'],
                                                            "last_val_step": thresholds_IDs['last_val_step'],
                                                            "init_test_step": thresholds_IDs['init_test_step'],
                                                            "last_test_step": thresholds_IDs['last_test_step']
                                                    }
                    self.list_data_splits_thresholds_IDs_dict.append(data_splits_thresholds_IDs_dict)
                # Sort by increasing init_test_step order TO AVOID DATA LEAKAGE
                self.list_data_splits_thresholds_IDs_dict = sorted(self.list_data_splits_thresholds_IDs_dict, key=lambda data_splits_thresholds_IDs_dict: data_splits_thresholds_IDs_dict['init_test_step'])

            else:
                threshold_train_date = pd.to_datetime(parameters_exp['threshold_train_date']).date()
                if (parameters_exp['threshold_val_date'] is not None):
                    threshold_val_date = pd.to_datetime(parameters_exp['threshold_val_date']).date()
                else:
                    threshold_val_date = None
                threshold_val_step_ID = None # By default is None unless we find it in the time steps (case where threshold_val_date is given)
                for step_ID in range(len(self.list_node_timestamps_dicts)):
                    tmp_date_str = self.list_node_timestamps_dicts[step_ID]['Patient'][0]
                    if (pd.to_datetime(tmp_date_str).date() <= threshold_train_date): # <= because in Python, when doing selection in a list, the last idx is excluded
                        threshold_train_step_ID = step_ID
                    if (threshold_val_date is not None):
                        # In this case there is a validation set
                        if (pd.to_datetime(tmp_date_str).date() > threshold_train_date) and (pd.to_datetime(tmp_date_str).date() <= threshold_val_date):
                            threshold_val_step_ID = step_ID
                self.data_splits_thresholds_IDs_dict = {
                                                            "init_train_step": 0,
                                                            "last_train_step": threshold_train_step_ID
                                                       }
                if (threshold_val_step_ID is not None):
                    self.data_splits_thresholds_IDs_dict["init_val_step"] = threshold_train_step_ID
                    self.data_splits_thresholds_IDs_dict["last_val_step"] = threshold_val_step_ID

                    self.data_splits_thresholds_IDs_dict["init_test_step"] = threshold_val_step_ID
                    self.data_splits_thresholds_IDs_dict["last_test_step"] = None 
                else:
                    self.data_splits_thresholds_IDs_dict["init_val_step"] = None
                    self.data_splits_thresholds_IDs_dict["last_val_step"] = None

                    self.data_splits_thresholds_IDs_dict["init_test_step"] = threshold_train_step_ID
                    self.data_splits_thresholds_IDs_dict["last_test_step"] = None 
                
        else:
            parameters_exp['metadata'] = None


        # Save predictions every X epochs (to reduce storage requirements)
        if ("epochs_step_save_preds" not in parameters_exp):
            parameters_exp["epochs_step_save_preds"] = 10
        self.epochs_step_save_preds = parameters_exp["epochs_step_save_preds"]

        # Weight for the state prediction loss
        if ("lambda_classif_seir_states" not in parameters_exp):
            parameters_exp["lambda_classif_seir_states"] = 1e-1
        else:
            if (parameters_exp["lambda_classif_seir_states"] is None):
                parameters_exp["lambda_classif_seir_states"] = 1.0

        # Number of classes
        parameters_exp['num_classes'] = parameters_exp['out_channels']

        # Number of states (different than number of classes if the forecast horizon is greater than 1)
        if (parameters_exp['dataset_name'].lower() == 'sociopatterns'):
            parameters_exp['num_states'] = 4
        elif (parameters_exp['dataset_name'].lower() == 'murcia'):
            parameters_exp['num_states'] = 6
        elif (parameters_exp['dataset_name'].lower() == 'hug'):
            #parameters_exp['num_states'] = 2 # For SI epidemic model
            parameters_exp['num_states'] = 3 # For SIR epidemic model
        else:
            raise ValueError(f"\nDataset {parameters_exp['dataset_name']} is not supported\n")

        # Parameter to know if we have to normalize or not the dataset
        if ("normalize_ds" not in parameters_exp):
            parameters_exp['normalize_ds'] = False
        self.normalize_ds = parameters_exp['normalize_ds']

        # Dataset mean and std (for normalization)
        if ('dataset_mean' not in parameters_exp):
            self.dataset_mean = None
        else:
            self.dataset_mean = parameters_exp['dataset_mean']
        if ('dataset_std' not in parameters_exp):
            self.dataset_std = None
        else:
            self.dataset_std = parameters_exp['dataset_std']

        # Model type to use (Transformer, GNN, etc.)
        if ('model_type' not in parameters_exp):
            parameters_exp['model_type'] = 'GNN'
        self.model_type = parameters_exp['model_type']

        # Precise model to use
        if ('model_to_use' not in parameters_exp):
            if (parameters_exp['model_type'].lower() == 'GNN'):
                parameters_exp['model_to_use'] = 'stm'
            else:
                raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
        self.model_to_use = parameters_exp['model_to_use']
        # Parameters for STM GNN
        if (parameters_exp['model_to_use'].lower() == 'stm'):
            # Skip connection for spatial module
            if ('skip_connection_spatial' not in parameters_exp):
                parameters_exp['skip_connection_spatial'] = False # Same behaviour as AIME paper
            
        # Parameters for the TGN
        if (parameters_exp['model_to_use'].lower() == "tgn"):
            if ("dim_enc_nodes_feaures" not in parameters_exp):
                parameters_exp["dim_enc_nodes_feaures"] = 64
            if ("input_proj_homo" not in parameters_exp):
                parameters_exp["input_proj_homo"] = False
                
        # Optuna params
        if ('use_optuna' not in parameters_exp):
            parameters_exp['use_optuna'] = False
        if (parameters_exp['use_optuna']):
            # N trials to do
            if ('n_trials' not in parameters_exp):
                parameters_exp['n_trials'] = 30
            # Metric to optimize
            if ('metric_optimize' not in parameters_exp):
                parameters_exp['metric_optimize'] = 'SE_to_I_AUC'
            # File path to Optuna DB to continue optimization
            if ('optuna_starting_point_fn' not in parameters_exp):
                parameters_exp['optuna_starting_point_fn'] = None
            # Learning rates optimization
            if ('optuna_search_all_lrs' not in parameters_exp):
                parameters_exp['optuna_search_all_lrs'] = False

        # Predict transitions
        if ("predict_state_transitions" not in parameters_exp):
            parameters_exp["predict_state_transitions"] = True

        # Training params
        if ('device' not in parameters_exp):
            parameters_exp['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(parameters_exp['device'])
        if (not parameters_exp['use_optuna']):
            if ('lr' not in parameters_exp):
                parameters_exp['lr'] = 1e-2
                
            if ('lr_model_compart_pred' not in parameters_exp):
                parameters_exp['lr_model_compart_pred'] = parameters_exp['lr']

            if ('lr_model_epi_params' not in parameters_exp):
                parameters_exp['lr_model_epi_params'] = parameters_exp['lr']

            if (parameters_exp["predict_state_transitions"]):
                if ('lr_model_transitions_pred' not in parameters_exp):
                    parameters_exp['lr_model_transitions_pred'] = 1e-2
            else:
                parameters_exp['lr_model_transitions_pred'] = None

            self.lrs = {
                            'model_node_enc': parameters_exp['lr'],
                            'model_compart_pred': parameters_exp['lr_model_compart_pred'],
                            'model_epi_params': parameters_exp['lr_model_epi_params'],
                            'model_transitions_pred': parameters_exp['lr_model_transitions_pred']
                       }
            
        # Weight decays
        if ('weight_decay' not in parameters_exp):
            parameters_exp['weight_decay'] = 1e-5
            
        if ('weight_decay_model_compart_pred' not in parameters_exp):
            parameters_exp['weight_decay_model_compart_pred'] = parameters_exp['weight_decay']

        if ('weight_decay_model_epi_params' not in parameters_exp):
            parameters_exp['weight_decay_model_epi_params'] = parameters_exp['weight_decay']

        if (parameters_exp["predict_state_transitions"]):
            if ('weight_decay_model_transitions_pred' not in parameters_exp):
                parameters_exp['weight_decay_model_transitions_pred'] = parameters_exp['weight_decay']

        self.weight_decays = {
                                'model_node_enc': parameters_exp['weight_decay'],
                                'model_compart_pred': parameters_exp['weight_decay_model_compart_pred'],
                                'model_epi_params': parameters_exp['weight_decay_model_epi_params']
                            }
        if (parameters_exp["predict_state_transitions"]):
            self.weight_decays['model_transitions_pred'] = parameters_exp['weight_decay_model_transitions_pred']


        self.nb_repetitions = parameters_exp['nb_repetitions']
        self.batch_size_train = parameters_exp['batch_size_train']
        self.batch_size_val = parameters_exp['batch_size_val']
        self.batch_size_test = parameters_exp['batch_size_test']
        self.n_epochs = parameters_exp['n_epochs']

        # Results file
        self.repetitions_results_fn = None


        # Optimizer
        if ('optimizer' not in parameters_exp):
            parameters_exp['optimizer'] = "Adam"
    

        # Holdout train ID (in case we use the method several times)
        self.holdout_train_id = -1

        # Early stopping
        if ('early_stopping' not in parameters_exp):
            parameters_exp['early_stopping'] = False
        if (parameters_exp['early_stopping']):
            if ('patience' not in parameters_exp):
                parameters_exp['patience'] = 5
            if ('min_delta' not in parameters_exp):
                parameters_exp['min_delta'] = 1e-4

        # For use_debug
        self.use_debug = use_debug

        # Parameters of the exp
        self.parameters_exp = parameters_exp

    @abstractmethod
    def createTorchDatasets(self, verbose=True, prefix_paths=""):
        """
            Create the torch datasets associated needed to train and evaluate the model
        """
        pass

    @abstractmethod
    def computeClassWeightsLoss(self):
        """
            Compute class weights and redefines the loss function using
            these class weights to handle imbalanced classes.
        """
        pass

    @abstractmethod
    def normalizeDataset(self):
        """
            Normalize the dataset by substracting the mean and dividing by
            the std
        """
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            deep learning model
        """
        # TODO
        raise NotImplementedError()

    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        #==============================================================#
        #==============================================================#
        if (self.parameters_exp['model_type'].lower() == "gnn"):
            if (self.parameters_exp['model_to_use'].lower() == "stm"):
                # Other params
                model_uses_memory = True

                # Hidden channels
                if ('hidden_channels' not in self.parameters_exp):
                    self.parameters_exp['hidden_channels'] = 8

                # Parameters for the temporal and spatial modules
                self.parameters_exp['args_spatial']['in_channels'] = self.parameters_exp["hidden_channels"]
                self.parameters_exp['args_spatial']['out_channels'] = self.parameters_exp['hidden_channels']
                self.parameters_exp['args_temporal']['in_channels'] = self.parameters_exp["hidden_channels"]
                self.parameters_exp['args_temporal']['out_channels'] = self.parameters_exp['hidden_channels']

                # Create encoder model
                model_node_enc = STM(
                                    in_channels=self.parameters_exp["in_channels"], 
                                    hidden_channels=self.parameters_exp["hidden_channels"],
                                    out_channels=self.parameters_exp["out_channels_node_encoder"],
                                    temporal_module_params={'Type': self.parameters_exp['temporal_module'], 'Args': self.parameters_exp['args_temporal'], 'Metadata': self.parameters_exp['metadata']}, 
                                    spatial_module_params={'Type': self.parameters_exp['spatial_module'], 'Args': self.parameters_exp['args_spatial'], 'Metadata': self.parameters_exp['metadata']}, 
                                    temporal_augments=self.parameters_exp["temporal_augments"],
                                    spatial_augments=self.parameters_exp["spatial_augments"],
                                    time_dim=self.parameters_exp["time_dim"],
                                    dropout=self.parameters_exp["dropout"],
                                    metadata=self.parameters_exp['metadata'], 
                                    skip_connection_spatial=self.parameters_exp['skip_connection_spatial'],
                                    device=str(self.device),
                            )
            elif (self.parameters_exp['model_to_use'].lower() == "tgn"):
                model_uses_memory = True
                if (self.parameters_exp['metadata'] is not None): # Heterogeneous graphs case
                    message_module = MLPMessage(
                                                    raw_msg_dim=self.parameters_exp["dim_enc_nodes_feaures"], 
                                                    memory_dim=self.parameters_exp['memory_dim'], 
                                                    time_dim=self.parameters_exp["time_dim"], 
                                                    hidden_channels=self.parameters_exp["hidden_channels"], 
                                                    dropout=self.parameters_exp['dropout'], 
                                                    device=str(self.device)
                                                ) 
                else: # Homogeneous graphs case
                    message_module = MLPMessage(
                                                    raw_msg_dim=self.parameters_exp["in_channels"], 
                                                    memory_dim=self.parameters_exp['memory_dim'], 
                                                    time_dim=self.parameters_exp["time_dim"], 
                                                    hidden_channels=self.parameters_exp["hidden_channels"], 
                                                    dropout=self.parameters_exp['dropout'], 
                                                    device=str(self.device)
                                                )

                if (self.parameters_exp["time_dim"] != self.parameters_exp["hidden_channels"]) and (self.parameters_exp['metadata'] is not None):
                    raise ValueError(f"\nFor TGN heterogeneous model, time_dim and hidden_channels should have the same value (current values {self.parameters_exp['time_dim']} and {self.parameters_exp['hidden_channels']})")

                model_node_enc = TGN(
                                        in_channels=self.parameters_exp["in_channels"], 
                                        hidden_channels=self.parameters_exp["hidden_channels"],
                                        time_dim=self.parameters_exp["time_dim"], 
                                        out_channels=self.parameters_exp["out_channels_node_encoder"],
                                        num_nodes=self.parameters_exp['num_nodes'], 
                                        message_module=message_module,
                                        aggregator_module=MeanAggregator(),
                                        metadata=self.parameters_exp['metadata'],
                                        dim_enc_nodes_feaures=self.parameters_exp['dim_enc_nodes_feaures'],
                                        input_proj_homo=self.parameters_exp['input_proj_homo'],
                                        device=str(self.device)
                                    ).to(self.device)

            elif (self.parameters_exp['model_to_use'].lower() == "gat"):
                model_uses_memory = False
                model_node_enc = GATModule(
                                            in_channels=self.parameters_exp["in_channels"],
                                            hidden_channels=self.parameters_exp["hidden_channels"],
                                            out_channels=self.parameters_exp["out_channels_node_encoder"],
                                            heads=self.parameters_exp["heads"],
                                            dropout=self.parameters_exp["dropout"],
                                            num_layers=self.parameters_exp["num_layers_encoder"],
                                            act=self.parameters_exp['activation'],
                                            aggr=self.parameters_exp['aggr'],
                                            metadata=self.parameters_exp['metadata']
                                          ).to(self.device)
            
            elif (self.parameters_exp['model_to_use'].lower() in ["mlp", "gcn", "graphsage"]):
                # Params relative to the model
                model_uses_memory = False
                
                # Create encoder model
                kwargs = {}
                if (self.parameters_exp['model_to_use'].lower() == "mlp"):
                    raise NotImplementedError()
                elif (self.parameters_exp['model_to_use'].lower() == "gcn"):
                    model_arch = GCNModule
                    if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                        raise RuntimeError(f"\nGCN model cannot be used in Murcia heterogeneous graphs as GCN works on heterogeneous graphs only if every edge type connects nodes of the same type, which is not the case here.")
                elif (self.parameters_exp['model_to_use'].lower() == "graphsage"):
                    model_arch = GraphSageModule
                
                model_node_enc = model_arch(
                                                in_channels=self.parameters_exp["in_channels"],
                                                hidden_channels=self.parameters_exp["hidden_channels"],
                                                out_channels=self.parameters_exp["out_channels_node_encoder"],
                                                num_layers=self.parameters_exp["num_layers_encoder"],
                                                dropout=self.parameters_exp["dropout"],
                                                act=self.parameters_exp['activation'],
                                                metadata=self.parameters_exp['metadata']
                                            ).to(self.device)
            else:
                if (self.parameters_exp['model_to_use'].lower() == "simplegnn"):
                    model_node_enc = SimpleGNN(
                                                in_channels=self.parameters_exp["in_channels"],
                                                hidden_channels=self.parameters_exp["hidden_channels"],
                                                out_channels=self.parameters_exp["out_channels_node_encoder"],
                                                metadata=self.parameters_exp['metadata']
                                              ).to(self.device) # BEST MODEL
                elif (self.parameters_exp['model_to_use'].lower() == "seirpredictor"):
                    model_node_enc = SEIRPredictor(
                                                    in_channels=self.parameters_exp["in_channels"],
                                                    out_channels=self.parameters_exp["out_channels_node_encoder"]
                                                  ).to(self.device)
                elif (self.parameters_exp['model_to_use'].lower() == "seirpredictordeeper"):
                    model_node_enc = SEIRPredictorDeeper(
                                                            in_channels=self.parameters_exp["in_channels"],
                                                            hidden_dim=self.parameters_exp["hidden_channels"],
                                                            out_channels=self.parameters_exp["out_channels_node_encoder"]
                                                        ).to(self.device)

            model_node_enc.apply(init_weights)
        else:
            raise ValueError()

        self.models = {"model_node_enc": model_node_enc}

        print(f"\n\n===>New model_node_enc created !\n\n")
    
    def reinitialize_memory_model(self):
        # Reinitialize memory if it is a memory based model
        if (self.parameters_exp['model_to_use'].lower() in ['stm', 'tgn']):
            self.models["model_node_enc"].clear()

    def createOptimizer(self):
        # NEW VERSION: AFTER 20/11/2025
        # Creating groups of parameters
        optimizer_groups = []
        for submodel in self.models:
            group = {
                        "params": self.models[submodel].parameters(),
                        "lr": self.lrs[submodel],
                        "weight_decay": self.weight_decays[submodel],
                    }
            optimizer_groups.append(group)

        # Creating the optimizer
        if (self.parameters_exp['optimizer'].lower() == 'adam'):
            self.optimizer = torch.optim.Adam(optimizer_groups)
        elif (self.parameters_exp['optimizer'].lower() == 'adamw'):
            self.optimizer = torch.optim.AdamW(optimizer_groups)
        else:
            raise ValueError(f"\nOptimizer {self.parameters_exp['optimizer']} is not valid")

        
        # Learning rate scheduler
        # Noam already has a learning rate scheduler
        if (self.parameters_exp['optimizer'].lower() in ['adam', 'adamw']):
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',\
                                                               factor=0.1, patience=5,\
                                                               threshold=1e-4, threshold_mode='rel',\
                                                               cooldown=0, min_lr=0, eps=1e-08)
                                                               # cooldown=0, min_lr=0, eps=1e-04)
        
    def computeForwardPass(self, batch, epoch_nb, batch_ID=None):
        # Loss
        loss = None
        preds = None

        return loss, preds
    
    def updateModel(self, batch, epoch, batch_ID=None):
        """
            Updates the weights of a model
        """
        # Zero the parameters gradients
        self.optimizer.zero_grad()


        # There is a kernel specific-bug, which makes some gradients NaN (when using fast kernels), therefore
        # we use another more robust (but slower kernel) to avoid these errors
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            # Forward pass
            loss, preds = self.computeForwardPass(batch, epoch, batch_ID=batch_ID)
            
            # Backward pass for the gradient computation
            if (loss is not None):
                if (type(loss) == dict):
                    loss["total_loss"].backward()
                else:
                    loss.backward()

                # Applying gradient clipping to solve some gradient problems
                for submodel in self.models:
                    torch.nn.utils.clip_grad_norm_(self.models[submodel].parameters(), max_norm=5.0)

                # Updating the weights
                self.optimizer.step()

        return loss, preds

    def createLossFunction(self):
        """
            Create the loss function to use for optimization
        """
        # Main loss function
        if ('loss_function' not in self.parameters_exp):
            self.parameters_exp['loss_function'] = "GCE"
        if (self.parameters_exp['loss_function'].lower() == "ce"):
            self.criterion = torch.nn.CrossEntropyLoss()
            self.regression_criterion = torch.nn.MSELoss()
        elif (self.parameters_exp['loss_function'].lower() == "sce"):
            raise NotImplementedError()
        elif (self.parameters_exp['loss_function'].lower() == "gce"):
            self.criterion = GeneralizedCrossEntropy()
            self.regression_criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"\nLoss function {self.parameters_exp['loss_function']} is not valid\n")

    def initSingleTrain(self, reinitialize_model_weights=True):
        """
            Initialize the parameters for a single train
        """
        # Create dataset (if continul training used)
        if (self.parameters_exp['continual_training']):
            self.createTorchDatasets()

        # Creating the dataloaders
        self.dataloadersCreation()

        # Compute class weights
        self.computeClassWeightsLoss()

        # Create loss function
        self.createLossFunction()

        # Creating the model
        # IMPORTANT: TO DO AFTER self.computeClassWeightsLoss as the number of possible transitions is computed inside and it is necessary to create the state transitions models
        if (reinitialize_model_weights):
            self.modelCreation()

        # Creating the optimizer
        self.createOptimizer()
        
        # Early stopping if asked
        if (self.parameters_exp['early_stopping']):
            if (type(self.val_ds) == list) and (len(self.val_ds) == 0):
                RuntimeError("When using early stopping, a validation dataset is needed!")
            else:
                self.early_stopping = EarlyStopping(
                                                        patience=self.parameters_exp['patience'],
                                                        min_delta=self.parameters_exp['min_delta'],
                                                        restore_best_weights=True
                                                        #restore_best_weights=False
                                                    )

    @abstractmethod
    def singleTrain(self, rep_ID, reinitialize_model_weights=True):
        """
            Trains a model one time during self.n_epochs epochs
        """
        pass

    @abstractmethod
    def singleTrainContinual(self, rep_ID):
        """
            Trains a model one time during self.n_epochs epochs
            on a continual training paradigm, where for each 
            timestep, we retrain the whole model and evaluate
            on the next step
        """
        pass

    def evalCurrentModel(self, dataloader, epoch):
        # Reinitialize memory if it is a memory based model
        self.reinitialize_memory_model()

        # Evaluation
        for submodel in self.models:
            self.models[submodel].eval()
        tmp_losses = []
        tmp_preds = []
        with torch.no_grad():
            batch_ID = 0
            for batch in tqdm(dataloader):
                loss, preds = self.computeForwardPass(batch, epoch, batch_ID=batch_ID)
                if (loss is not None):
                    tmp_losses.append(loss.detach().data.cpu().numpy())
                    tmp_preds.append(preds)
                batch_ID += 1

        return tmp_losses, tmp_preds

    def holdoutTrain(self, save_results=True):
        """
            Does a holdout training repeated self.nb_repetitions times
        """
        # Holdout ID
        self.holdout_train_id += 1 # The initial value is initialize in the constructor, and is -1

        # Creating HDF5 file for the repetitions over time
        if (self.repetitions_results_fn is None):
            self.repetitions_results_fn = self.results_folder + '/metrics/final_results_all_repetitions_0.hdf5'
            with h5py.File(self.repetitions_results_fn, 'w') as h5file:
                for rep_id in range(self.nb_repetitions):
                    if (self.using_multiple_DS):
                        for tmp_dataset_ID in range(len(self.parameters_exp['multiple_hdf5_dataset_filenames'])):
                            h5file.create_group(f"Rep-{rep_id}_Dataset-{tmp_dataset_ID}")
                    else:
                        h5file.create_group(f"Rep-{rep_id}")
        
        # Iterating over the repetitions
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n\n\n======================================================================")
            print("=======> Repetitions {} <=======".format(nb_repetition))
            print("======================================================================")
            # Doing single train
            self.repetition_id = nb_repetition
            if (self.parameters_exp['continual_training']):
                self.singleTrainContinual(nb_repetition) 
            else:
                self.singleTrain(nb_repetition, reinitialize_model_weights=True)

            # Saving the final model and the results
            # Model
            if (self.models is not None):
                for submodel in self.models:
                    # Model file to save
                    model_file = self.results_folder + '/model/final_model-{}_submodel-{}_rep-{}_holdout-{}'.format(self.exp_id, submodel, self.repetition_id, self.holdout_train_id)
                    if (self.using_multiple_DS):
                        model_file = model_file + f"_Dataset-{self.current_dataset_ID}_.pth"
                    else:
                        model_file = model_file + ".pth"
                    # Save
                    torch.save({
                                    'model_state_dict': self.models[submodel].state_dict(),
                                    'model': self.models[submodel]
                                }, model_file)

    def repeatedHoldoutMultipleDS(self, save_results=True):
        """
            Repeats a holdout experiment using multiple datasets (of the same)
            type
        """
        for hdf5_dataset_filename_ID in range(len(self.parameters_exp['multiple_hdf5_dataset_filenames'])):
            self.hdf5_dataset_filename = self.parameters_exp['multiple_hdf5_dataset_filenames'][hdf5_dataset_filename_ID]
            print("\n======================================================================")
            print("======================================================================")
            print(f"==============PROCESSING DATASET {self.hdf5_dataset_filename}==============")
            print("======================================================================")
            print("======================================================================")
            # Dataset ID
            self.current_dataset_ID = hdf5_dataset_filename_ID

            # Opening HDF5 file
            self.h5_file = h5py.File(self.hdf5_dataset_filename, 'r')

            # Dataset Loading
            self.createTorchDatasets()

            # Add class weights
            self.computeClassWeightsLoss()

            # Create loss function
            self.createLossFunction()

            # Holdout training
            self.holdoutTrain(save_results)

    def optuna_objective(self, trial):
        """
            Objective to use with Optuna for hyper-parameter optimization
        """
        #======================================================================#
        #=======================Hyper-parameters to tune=======================#
        #======================================================================#
        # Define global hyper-parameters to tune
        # Learning rate
        if (self.parameters_exp['optuna_search_all_lrs']): # Optimize every learning rate
            self.parameters_exp["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            self.parameters_exp['lr_model_compart_pred'] = trial.suggest_float("lr_model_compart_pred", 1e-4, 1e-1, log=True)
            if (self.parameters_exp['epidemio_informed']):
                self.parameters_exp['lr_model_epi_params'] = trial.suggest_float("lr_model_epi_params", 1e-4, 1e-1, log=True)
            else:
                self.parameters_exp['lr_model_epi_params'] = 0
            if (self.parameters_exp["predict_state_transitions"]):
                self.parameters_exp['lr_model_transitions_pred'] = trial.suggest_float("lr_model_transitions_pred", 1e-4, 1e-1, log=True)
            else:
                self.parameters_exp['lr_model_transitions_pred'] = 0
        else:
            self.parameters_exp["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            self.parameters_exp['lr_model_compart_pred'] = self.parameters_exp["lr"]
            if (self.parameters_exp['epidemio_informed']):
                self.parameters_exp['lr_model_epi_params'] = self.parameters_exp["lr"]
            else:
                self.parameters_exp['lr_model_epi_params'] = 0
            if (self.parameters_exp["predict_state_transitions"]):
                self.parameters_exp['lr_model_transitions_pred'] = self.parameters_exp["lr"]
            else:
                self.parameters_exp['lr_model_transitions_pred'] = 0

        self.lrs = {
                        'model_node_enc': self.parameters_exp['lr'],
                        'model_compart_pred': self.parameters_exp['lr_model_compart_pred'],
                        'model_epi_params': self.parameters_exp['lr_model_epi_params'],
                        'model_transitions_pred': self.parameters_exp['lr_model_transitions_pred']
                    }
            
        # Other params
        self.parameters_exp["lambda_classif_seir_states"] = None
        if (self.parameters_exp['forecast_horizon'] == 1):
            if (not self.parameters_exp['epidemio_informed']) and (not self.parameters_exp["predict_state_transitions"]):
                self.parameters_exp["lambda_classif_seir_states"] = 1.0
            else:
                self.parameters_exp["lambda_classif_seir_states"] = trial.suggest_float("lambda_classif_seir_states", 1e-3, 1e-1, log=True)    
        if (self.parameters_exp['forecast_horizon'] >= 2):
            if (not self.parameters_exp['constrain_transitions_loss']) and (not self.parameters_exp['monotonicity_loss']) and (not self.parameters_exp["predict_state_transitions"]):
                self.parameters_exp["lambda_classif_seir_states"] = 1.0
            else:
                # Lambda for classification
                self.parameters_exp["lambda_classif_seir_states"] = trial.suggest_float("lambda_classif_seir_states", 1e-3, 1e-1, log=True)    
                # Lambda for contraint loss
                if (self.parameters_exp['constrain_transitions_loss']):
                    self.parameters_exp["lambda_constrain_transitions_loss"] = trial.suggest_float("lambda_constrain_transitions_loss", 1e-3, 1e-1, log=True)    
                # Lambda for monotonicity loss
                if (self.parameters_exp['monotonicity_loss']):
                    self.parameters_exp["lambda_monotonicity_loss"] = trial.suggest_float("lambda_monotonicity_loss", 1e-3, 1e-1, log=True)  

        # Lambda for predict transitions
        if (self.parameters_exp["predict_state_transitions"]):
            self.parameters_exp["lambda_transitions"] = trial.suggest_float("lambda_transitions", 1e-3, 1e-1, log=True)

        if (self.parameters_exp['epidemio_informed']):
            if (self.parameters_exp["lambda_classif_seir_states"] is not None):
                self.parameters_exp["lambda_classif_seir_states"] = trial.suggest_float("lambda_classif_seir_states", 1e-3, 1e-1, log=True)
            #self.parameters_exp["teacher_forcing_epidemio"] = trial.suggest_categorical("teacher_forcing_epidemio", [True, False])
            if (self.parameters_exp["teacher_forcing_epidemio"]):
                self.parameters_exp["ramp_epochs_teacher_forcing_sched"] = trial.suggest_int("ramp_epochs_teacher_forcing_sched", 5, max(5, self.n_epochs-10))
            self.parameters_exp["lambda_reg"] = trial.suggest_float("lambda_reg", 1e-3, 1, log=True)
            if (self.parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                self.parameters_exp["lambda_reg_S"] = trial.suggest_float("lambda_reg_S", 1e-3, 1, log=True)
                if (self.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'murcia']):
                    self.parameters_exp["lambda_reg_E"] = trial.suggest_float("lambda_reg_E", 1e-3, 1, log=True)
                self.parameters_exp["lambda_reg_I"] = trial.suggest_float("lambda_reg_I", 1e-3, 1, log=True)
                self.parameters_exp["lambda_reg_R"] = trial.suggest_float("lambda_reg_R", 1e-3, 1, log=True)
                if (self.parameters_exp['dataset_name'].lower() == 'murcia'):
                    self.parameters_exp["lambda_reg_D"] = trial.suggest_float("lambda_reg_D", 1e-3, 1, log=True)
                    self.parameters_exp["lambda_reg_NS"] = trial.suggest_float("lambda_reg_NS", 1e-3, 1, log=True)


        # Define per-model hyper-parameters to tune
        # IMPORTANT: ARCHITECTURES HYPER-PARAMS ARE ONLY FIXED FOR NOT EPIDEMIO-INFORMED MODELS
        if (not self.parameters_exp['epidemio_informed']):
            if (self.parameters_exp['model_to_use'].lower() == "stm"):
                tmp_hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                tmp_dropout = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                tmp_num_layers = trial.suggest_categorical("num_layers", [1, 2, 4]) 
                tmp_heads = trial.suggest_categorical("heads", [2])
                self.parameters_exp["hidden_channels"] = tmp_hidden_channels
                self.parameters_exp["out_channels_node_encoder"] = tmp_hidden_channels//2
                self.parameters_exp["args_temporal"]["hidden_channels"] = tmp_hidden_channels
                self.parameters_exp["args_temporal"]["out_channels"] = tmp_hidden_channels
                self.parameters_exp["args_temporal"]["dropout"] = tmp_dropout
                self.parameters_exp["args_temporal"]["num_layers"] = tmp_num_layers
                self.parameters_exp["args_temporal"]["heads"] = tmp_heads
                self.parameters_exp["args_spatial"]["hidden_channels"] = tmp_hidden_channels
                self.parameters_exp["args_spatial"]["out_channels"] = tmp_hidden_channels
                self.parameters_exp["args_spatial"]["dropout"] = tmp_dropout
                self.parameters_exp["args_spatial"]["num_layers"] = tmp_num_layers
                self.parameters_exp["args_spatial"]["heads"] = tmp_heads
                self.parameters_exp["time_dim"] = trial.suggest_categorical("time_dim", [0, 8, 16, 32])
                self.parameters_exp["dropout"] = tmp_dropout
            
            elif (self.parameters_exp['model_to_use'].lower() == "tgn"):
                self.parameters_exp["hidden_channels"] = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                self.parameters_exp["time_dim"] = self.parameters_exp["hidden_channels"]
                self.parameters_exp["out_channels_node_encoder"] = self.parameters_exp["hidden_channels"]
                self.parameters_exp['memory_dim'] = self.parameters_exp["out_channels_node_encoder"]
                self.parameters_exp['dropout'] = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                if (self.parameters_exp['metadata'] is not None): # Heterogeneous graphs case
                    self.parameters_exp['dim_enc_nodes_feaures'] = trial.suggest_int("dim_enc_nodes_feaures", 16, 128, log=True)
                
            elif (self.parameters_exp['model_to_use'].lower() == "gat"):
                self.parameters_exp["hidden_channels"] = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                self.parameters_exp["heads"] = trial.suggest_categorical("heads", [2, 4, 8, 16])
                self.parameters_exp["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                self.parameters_exp["num_layers_encoder"] = trial.suggest_categorical("num_layers_encoder", [1, 2, 4])
                self.parameters_exp["out_channels_node_encoder"] = trial.suggest_categorical("out_channels_node_encoder", [4, 16, 32, 64])
            
            elif (self.parameters_exp['model_to_use'].lower() in ["mlp", "gcn", "graphsage"]):
                self.parameters_exp["hidden_channels"] = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                self.parameters_exp["num_layers_encoder"] = trial.suggest_categorical("num_layers_encoder", [1, 2, 4])
                self.parameters_exp["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                self.parameters_exp["out_channels_node_encoder"] = trial.suggest_categorical("out_channels_node_encoder", [4, 16, 32, 64])
            
            elif (self.parameters_exp['model_to_use'].lower() == "simplegnn"):
                self.parameters_exp["hidden_channels"] = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                self.parameters_exp["out_channels_node_encoder"] = trial.suggest_categorical("out_channels_node_encoder", [4, 16, 32, 64])
            
            elif (self.parameters_exp['model_to_use'].lower() == "seirpredictor"):
                self.parameters_exp["out_channels_node_encoder"] = trial.suggest_categorical("out_channels_node_encoder", [4, 16, 32, 64])
            
            elif (self.parameters_exp['model_to_use'].lower() == "seirpredictordeeper"):
                self.parameters_exp["hidden_channels"] = trial.suggest_categorical("hidden_channels", [8, 16, 32])
                self.parameters_exp["out_channels_node_encoder"] = trial.suggest_categorical("out_channels_node_encoder", [4, 16, 32, 64])
            else:
                raise ValueError(f"\nModel to use {self.parameters_exp['model_to_use'].lower()} is not valid for Optuna hyper-parameter tuning\n.")


        #======================================================================#
        #==================Creating HDF5 for temporary results==================#
        #======================================================================#
        # Creating HDF5 file to temporarily store the results
        self.repetitions_results_fn = self.results_folder + f'/metrics/tmp_optuna_res_file.hdf5'
        with h5py.File(self.repetitions_results_fn, 'w') as h5file:
            for rep_id in range(self.nb_repetitions):
                # IMPORTANT: For Optuna optimization, we do not use several datasets as it
                # IMPORTANT: to expensive to do it.
                h5file.create_group(f"Rep-{rep_id}")

        #======================================================================#
        #==========================Getting the metric==========================#
        #======================================================================#
        # Verify that validation dataset is available (necessary for hyper-parameter tuning)
        if (type(self.val_ds) == list) and (len(self.val_ds) == 0):
            if (self.parameters_exp['dataset_name'].lower() == 'hug'):
                print("\n\n=========> WARNING: FOR HUG DATASET, IF NOT VALIDATION SET AVAILABLE, OPTUNA HYPER-PARAMETER SEARCH IS DONE WITH THE TEST SET<=========\n\n")
            else:
                raise RuntimeError("\nHyper-parameter optimization with Optuna cannot be done without validation set\n")

        try:
            # Training the model
            for nb_repetition in range(self.nb_repetitions):
                self.singleTrain(nb_repetition, reinitialize_model_weights=True)

            # Load results HDF5 file
            results_h5_file = h5py.File(self.repetitions_results_fn, 'r')

            # Getting the validation metric
            n_repetitions = len(results_h5_file)
            epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[f"Rep-0"]["Preds"]["Train"].keys())])
            epochs_to_use = max(epochs_list)
            metrics = {
                            "MCC": None,
                            "F1Score": None,
                            "BalancedAccuracy": None,
                            "AUC": None,
                            "S_to_Infected_AUC": None,
                            "E_to_I_AUC": None
                        }
            

            # Getting the metrics for the epoch
            n_unique_labels = [i for i in range(self.parameters_exp['num_classes'])]
            tmp_loss, tmp_metrics, _ = get_metrics_per_rep(self.repetitions_results_fn, parameters_exp=self.parameters_exp)

            if (self.parameters_exp["dataset_name"].lower() == 'hug'):
                model_type = 'SIR'
            elif (self.parameters_exp["dataset_name"].lower() == 'sociopatterns'):
                model_type = 'SEIR'
            elif (self.parameters_exp["dataset_name"].lower() == 'murcia'):
                model_type = 'SEIRD-NS'
            inf_onset_res_per_rep = []
            for tmp_rep_ID in range(n_repetitions):
                inf_onset_res = evaluate_infection_onset(
                                                            h5_file_path=self.repetitions_results_fn,
                                                            rep_id=tmp_rep_ID,
                                                            model_type=model_type,
                                                            split='Val',
                                                            epoch_id=-1,
                                                            threshold=0.7
                                                        )
                inf_onset_res_per_rep.append(inf_onset_res)
            # Getting the metrics at the last epoch, averaged
            for metric_name in tmp_metrics:
                if ('perclass' not in metric_name.lower()):
                    data_dict = tmp_metrics[metric_name]
                    for split in data_dict:
                        epochs = sorted(data_dict[split].keys())
                        means = []
                        stds = []
                        for ep in epochs:
                            values = data_dict[split][ep]
                            means.append(np.mean(values))
                            stds.append(np.std(values))
                        means = np.array(means)
                        stds = np.array(stds)
                        tmp_metrics[metric_name][split] = means[-1] # Mean at the last epoch

            # Add to metrics list
            metrics["MCC"] = tmp_metrics["MCC"]['Val']
            metrics["BalancedAccuracy"] = tmp_metrics["BalancedAccuracy"]['Val']
            metrics["AUC"] = tmp_metrics["AUC"]['Val']
            metrics["S_to_Infected_AUC"] = np.mean([inf_onset_res_per_rep[tmp_rep_ID]["S_to_Infected_AUC"] for tmp_rep_ID in range(n_repetitions)])
            if (model_type != 'SIR'):
                metrics["E_to_I_AUC"] = np.mean([inf_onset_res_per_rep[tmp_rep_ID]["E_to_I_AUC"] for tmp_rep_ID in range(n_repetitions)])
            if (self.parameters_exp['predict_state_transitions']):
                metrics["MCCTransitions"] = tmp_metrics["MCCTransitions"]['Val']
                metrics["BalancedAccuracyTransitions"] = tmp_metrics["BalancedAccuracyTransitions"]['Val']
                metrics["AUCTransitions"] = tmp_metrics["AUCTransitions"]['Val']

            # Erase the results HDF5 file as it is not necessary for hyper-parameter tuning
            os.remove(self.repetitions_results_fn)

            # Increase the ID of the current trial
            self.repetition_id += 1

            # Selecting metric to use
            print(f"\n\n =========> Using metric {self.metric_use_optuna} for OPTUNA optimization.\n\n")
            if (self.metric_use_optuna.lower() == "mcc"):
                val_metric = metrics["MCC"]

            elif (self.metric_use_optuna.lower() == "mcctransitions"):
                val_metric = metrics["MCCTransitions"]

            elif (self.metric_use_optuna.lower() == "balancedaccuracy"):
                val_metric = metrics["BalancedAccuracy"]
            
            elif (self.metric_use_optuna.lower() == "balancedaccuracytransitions"):
                val_metric = metrics["BalancedAccuracyTransitions"]

            elif (self.metric_use_optuna.lower() == "auc"):
                val_metric = metrics["AUC"]

            elif (self.metric_use_optuna.lower() == "auctransitions"):
                val_metric = metrics["AUCTransitions"]

            elif (self.metric_use_optuna.lower() == "s_to_infected_auc"):
                val_metric = metrics["S_to_Infected_AUC"]

            elif (self.metric_use_optuna.lower() == "e_to_i_auc"):
                val_metric = metrics["E_to_I_AUC"]

            if (self.use_debug):
                breakpoint()

            return val_metric
        
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}. Returning 0.0 as metric.")

            return 0.0  # <- only works for metrics to maximize and > 0.0!
        
    def optuna_tunning(self):
        """
            Hyper-parameter optimization using Optuna
        """
        print(f"\n\n\n\n======================================================================")
        print(f"======================================================================")
        print(f"======================================================================")
        print(f"============DOING HYPER-PARAMETER OPTIMIZATION WITH OPTUNA============")
        print(f"======================================================================")
        print(f"======================================================================")
        print(f"======================================================================\n\n\n\n")
        # As we are going to do several sub-experiments, we are going to have
        # one ID per sub-experiment
        self.repetition_id = 0

        # Metric to use for hyper-parameter optimization
        if (self.parameters_exp['metric_optimize'].lower() in ['mcc', 'mcctransitions', 'balancedaccuracy', 'balancedaccuracytransitions', 'auc', 'auctransitions']):
            self.metric_use_optuna = self.parameters_exp['metric_optimize']
            
        elif (self.parameters_exp['metric_optimize'].lower() == 'se_to_i_auc'):
            if (self.parameters_exp["dataset_name"].lower() == 'hug'):
                self.metric_use_optuna = "S_to_Infected_AUC" 
            elif (self.parameters_exp["dataset_name"].lower() in ['sociopatterns', 'murcia']):       
                self.metric_use_optuna = "E_to_I_AUC" 

        else:
            raise ValueError(f"Metric to optimize {self.parameters_exp['metric_optimize']} in OPTUNA is not valid.")

        # Creating the study
        if (self.parameters_exp['optuna_starting_point_fn'] is None): # Create study from scratch
            print("\n\n=========> CREATING OPTUNA STUDY FROM SCRATCH <=========\n")
            db_dir = Path(self.results_folder) / 'metrics'
            db_dir.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{db_dir / 'optuna_results.db'}"
            self.study = optuna.create_study(
                                            direction="maximize",
                                            study_name=f'OptunaHyperParamOptim',
                                            #study_name=f'OptunaHyperParamOptim_{self.exp_id}',
                                            sampler=optuna.samplers.TPESampler(seed=42), # Fix seed for reproducibility
                                            storage=storage
                                        )
        else: # Continue study
            print(f"\n\n=========> LOADING OPTUNA STUDY TO CONTINUE IT ({self.parameters_exp['optuna_starting_point_fn']}) <=========\n")
            optuna_study_path = Path(self.parameters_exp['optuna_starting_point_fn'])
            self.study = optuna.load_study(
                                        study_name=f'OptunaHyperParamOptim',
                                        #study_name=f'OptunaHyperParamOptim_{self.exp_id}',
                                        sampler=optuna.samplers.TPESampler(seed=42), # Fix seed for reproducibility
                                        storage=f"sqlite:///{optuna_study_path}"
                                    )
        self.study.optimize(self.optuna_objective, n_trials=self.parameters_exp['n_trials'])
        print("Best Trial:")
        trial = self.study.best_trial
        print(f"{self.metric_use_optuna.upper()}: {trial.value:.4f}")
        print("\tParams:")
        for key, value in trial.params.items():
            print(f"\t\t{key}: {value}")


    def setResultsFolder(self, results_folder):
        """
            Set the folder where the results are going to be stored
        """
        self.results_folder = results_folder
