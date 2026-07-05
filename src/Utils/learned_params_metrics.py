"""
    Plot the metrics and results of an experiment
"""
import os
import yaml
from copy import deepcopy
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
import torch
from torch_geometric.utils import to_dense_adj, to_undirected
from src.Experiments.InfectionRiskPredMurcia import InfectionRiskPred
from src.DataManipulation.Murcia.data_exploration import estimate_epidemic_params_from_data,\
                                                         get_patients_data,\
                                                         load_movement_data,\
                                                         load_locations_data

def evaluate_learned_params(learned, ground_truth):
    """
    Evaluate learned vs. ground-truth parameters.
    
    Parameters:
    ------------
    learned: np.array
        Array contained the learned parameters.
    ground_truth: np.array
        Array containing the ground truth parameters.
    
    Returns:
        dict with cosine similarity, optimal scaling factor, and error norm
    """
    # Transforming into numpy arrays
    learned = np.asarray(learned).ravel()
    ground_truth = np.asarray(ground_truth).ravel()
    
    # Cosine similarity (scale-invariant)
    cos_sim = np.dot(learned, ground_truth) / ( np.linalg.norm(learned) * np.linalg.norm(ground_truth) )
    
    # Optimal scaling factor alpha* to map learned -> ground_truth
    alpha = np.dot(learned, ground_truth) / np.dot(learned, learned)
    
    # Residual error after scaling
    residual = ground_truth - alpha * learned
    error_norm = np.linalg.norm(residual)
    
    return {"cosine_similarity": cos_sim, "optimal_scale": alpha, "error_norm": error_norm}

# Function to get the adjacency matrix
def get_adj_matrix_snapshot(snapshot, parameters_exp):
    """
        Gets the adjacency matrix of a snapshot.

        Parameters:
        -----------
        snapshot: torch_geometric.data.data.Data
            Snapshot of a graph from which we want to get the
            adjacency matrix.

        Returns:
        --------
        adj_mat: torch.Tensor
            Tensor of shape (num_nodes_current_snapshot, num_nodes_current_snapshot) 
            representing the adjacency matrix of the current snapshot.
        parameters_exp: dict
            Dictionary containing the parameters used for the experiment.
    """
    # Getting the data of the snapshot
    if (parameters_exp['dataset_name'].lower() in ['murcia']):
        try:
            y = {key: snapshot[key].y.clone().detach() for key in snapshot.node_types}
        except:
            y = {key: snapshot[key].y for key in snapshot.node_types}
        edge_index = {key: snapshot[key].edge_index.clone().detach() for key in snapshot.edge_types}
        edge_attr = {key: snapshot[key].edge_attr.clone().detach() for key in snapshot.edge_types}
        timestamps = {key: snapshot[key].timestamps.clone().detach() for key in snapshot.node_types}
        true_nodes_ids = {key: snapshot[key].ids.clone().detach() for key in snapshot.node_types} 
    else:
        y = snapshot.y.clone().detach()
        edge_index = snapshot.edge_index.clone().detach()
        edge_attr = snapshot.edge_attr.clone().detach()
        timestamps = snapshot.timestamps.clone().detach()
        true_nodes_ids = snapshot.ids.clone().detach()

    # Get max number of nodes
    if (parameters_exp['dataset_name'].lower() in ['murcia']):
        max_num_nodes = true_nodes_ids['Patient'].shape[0]
    else:
        max_num_nodes = true_nodes_ids.shape[0]        

    # Getting the correspondence between the local node IDs (in the current snapshot) and the true ones (in the whole dataset)
    # IMPORTANT: FOR STM-GNN DO NO USE THIS AS INPUT BUT THE ORIGINAL edge_index AS SOMETHING SIMILAR TO WHAT IS DONE HERE IS
    # IMPORTANT: DONE INSIDE THE MODEL
    if (parameters_exp['dataset_name'].lower() in ['murcia']):
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
            mapped_edges_index[edge_type] = mapped_edges_index[edge_type]
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
        mapped_edges_index = mapped_edges_index

    # Get adjacency matrix
    if (parameters_exp['dataset_name'].lower() in ['murcia']):
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
            mapped_edges_index_patients = torch.tensor(mapped_edges_index_patients)
            mapped_edges_attr_patients = torch.tensor(mapped_edges_attr_patients)

            # Creating the undirected mapped edges indices and attributes tensors
            mapped_edges_index_undirected, edge_attr_undirected = to_undirected(
                                                                                edge_index=mapped_edges_index_patients,
                                                                                edge_attr=mapped_edges_attr_patients
                                                                            )
        
    else:
        mapped_edges_index_undirected, edge_attr_undirected = to_undirected(
                                                                                edge_index=mapped_edges_index,
                                                                                edge_attr=edge_attr
                                                                            )
    adj_mat = to_dense_adj(
                                edge_index=mapped_edges_index_undirected.long(),
                                edge_attr=edge_attr_undirected,
                                max_num_nodes=max_num_nodes,
                            ).squeeze()


    return adj_mat, mapped_edges_index


def get_local_infection_hazard(dataset, exp_folder, beta_gt, dataset_type, untrained_models=None, patient_features_names_idx=None, rep_ID=0, DS_ID=None):
    """
        Get the true and predicted local infection hazard (of
        force of infection) per day for all the patients.

        Parameters:
        -----------
        dataset: torch dataset
            Dataset to use to evaluate the model.
        exp_folder: str
            Path to the folder containing the results of the experiment
        beta_gt: float
            True value of the transmission rate.
        dataset_type: str
            Type of dataset used: Murcia or SocioPatterns
        untrained_models: dict
            Dictionary of initialized but untrained models. Only necessary when using
            models trained on heterogeneous data. This is because we have to load the
            model using state_dict to avoid FX-traced modules that create errors as 
            they depend on version and metadata
        patient_features_names_idx: dict
            Dict indicating, for each feature, the initial and last indices
            corresponding to that feature in patient_features. Only necessary
            for the Murcia dataset.
        rep_ID: int
            Repetition ID to use to select the trained models to use.
        DS_ID: int or None
            ID of the dataset used to train the model (as a single model can be trained
            repeated times with different weights initializations on differnt datasets)

        Returns:
        --------
        local_infection_hazard_all_days: dict
            Dictionary containing two keys: True and Pred.
            Each element is a list containing the computed true and predicted
            local infection hazard for each individual for each day.
        epi_params_metrics_over_time: dict
            Dictionary containing two keys: MSE and PCC.
            They containg, for each day, the MSE and Pearson correaltion coefficient
            between the true and predicted local infection hazard of all the patients
            at that given day.
    """
    with torch.no_grad():
        # Parameters of the experiment
        try:
            parameters_file = exp_folder + "/params_exp/params_0.pth"
            with open(parameters_file, 'rb') as pf:
                parameters_exp = pickle.load(pf)
        except:
            parameters_file = exp_folder + "/params_exp/params_0.yaml"
            with open(parameters_file, 'r') as file:
                parameters_exp = yaml.safe_load(file)
        parameters_exp['hdf5_dataset_filename'] = "../." + parameters_exp['hdf5_dataset_filename']
        
        # Loading the models files
        models_dicts = {}
        for model_file in os.listdir(exp_folder + "/model/"):
            if (DS_ID is None): 
                # We add all the models
                models_dicts[model_file] = torch.load(exp_folder + "/model/" + model_file, weights_only=False, map_location=torch.device('cpu'))
            else:
                # We only add the models trained on the requested DS
                DS_ID_current_model = int(model_file.split('Dataset-')[-1].split('_.pth')[0])
                if (DS_ID_current_model == DS_ID):
                    if (dataset_type.lower() == 'murcia'):
                        # VERY IMPORTANT: For HETEROGENEOUS GRAPH DS we have to load the model using state_dict to avoid FX-traced modules that create errors as they depend on version and metadata
                        # Create untrained model
                        if ('model_node_enc' in model_file.lower()):
                            #print(f"\n =========> MODEL NODE ENC FILE: {model_file}\n\n")
                            tmp_model = deepcopy(untrained_models['model_node_enc'])
                        if ('model_compart_pred' in model_file.lower()):
                            #print(f"\n =========> MODEL COMPART PRED FILE: {model_file}\n\n")
                            tmp_model = deepcopy(untrained_models['model_compart_pred'])
                        if ('model_epi_params' in model_file.lower()):
                            #print(f"\n =========> MODEL EPI PARAMS FILE: {model_file}\n\n")
                            tmp_model = deepcopy(untrained_models['model_epi_params'])
                        if ('model_inf_risk_pred' in model_file.lower()):
                            #print(f"\n =========> MODEL INFECTION RISK PREDICTOR FILE: {model_file}\n\n")
                            tmp_model = deepcopy(untrained_models['model_inf_risk_pred'])
                        if ('model_transitions_pred' in model_file.lower()):
                            #print(f"\n =========> MODEL TRANSITIONS PREDICTOR FILE: {model_file}\n\n")
                            tmp_model = deepcopy(untrained_models['model_transitions_pred'])

                        # Load weights
                        model_state_dict = torch.load(exp_folder + "/model/" + model_file, weights_only=False, map_location=torch.device('cpu'))['model_state_dict']
                        tmp_model.load_state_dict(model_state_dict)
                        tmp_model = tmp_model.to(torch.device('cpu'))
                        models_dicts[model_file] = {'model': deepcopy(tmp_model)}
                    else:
                        models_dicts[model_file] = torch.load(exp_folder + "/model/" + model_file, weights_only=False, map_location=torch.device('cpu'))

        # Getting the models
        for submodel in  models_dicts.keys():
            if (f'rep-{rep_ID}' in submodel.lower()):
                if ('model_node_enc' in submodel.lower()):
                    model_node_enc = models_dicts[submodel]['model']
                if ('model_compart_pred' in submodel.lower()):
                    model_compart_pred = models_dicts[submodel]['model']
                if ('model_epi_params' in submodel.lower()):
                    model_epi_params = models_dicts[submodel]['model']
                if ('model_inf_risk_pred' in submodel.lower()):
                    model_inf_risk_pred = models_dicts[submodel]['model']
        epi_params = model_epi_params.epi_params.clone().cpu().detach()
                    
        # Iterating over the snapshots
        epi_params_metrics_over_time = {}
        local_infection_hazard_all_days = {'True': [], 'Pred': []}
        for snapshot in tqdm(dataset):
            #======================================================================#
            #======================================================================#
            # Get inputs and labels
            if (parameters_exp['dataset_name'].lower() in ['murcia']):
                x = {key: torch.tensor(snapshot[key].x).to(torch.device('cpu')).float() for key in snapshot.node_types}
                y = {key: torch.tensor(snapshot[key].y).to(torch.device('cpu')).long() for key in snapshot.node_types}
                edge_index = {key: torch.tensor(snapshot[key].edge_index).to(torch.device('cpu')).long() for key in snapshot.edge_types}
                edge_attr = {key: torch.tensor(snapshot[key].edge_attr).to(torch.device('cpu')).float() for key in snapshot.edge_types}
                timestamps = {key: torch.tensor(snapshot[key].timestamps).to(torch.device('cpu')).float() for key in snapshot.node_types}
                true_nodes_ids = {key: torch.tensor(snapshot[key].ids).to(torch.device('cpu')) for key in snapshot.node_types} 
            else:
                x = torch.tensor(snapshot.x).to(torch.device('cpu')).float()
                y = torch.tensor(snapshot.y).to(torch.device('cpu')).long()
                edge_index = torch.tensor(snapshot.edge_index).to(torch.device('cpu')).long()
                edge_attr = torch.tensor(snapshot.edge_attr).to(torch.device('cpu')).float()
                timestamps = torch.tensor(snapshot.timestamps).to(torch.device('cpu')).float()
                true_nodes_ids = torch.tensor(snapshot.ids).to(torch.device('cpu'))
        
            # One hot encoding of the current states
            if (parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                gt_current_states_probs = x[:, -4:] # Current state are one-hot encoded in the last 4 values of the feature vector
        
            elif (parameters_exp['dataset_name'].lower() == 'murcia'):
                first_curr_state_idx, last_curr_state_idx = patient_features_names_idx['current_state'][0], patient_features_names_idx['current_state'][1]
                gt_current_states_probs = x['Patient'][:, first_curr_state_idx:last_curr_state_idx] # Current state are one-hot encoded in the last 4 values of the feature vector
            # Add artificial batch dim for consistency with the model
            gt_current_states_probs = gt_current_states_probs.unsqueeze(0)
        
            #======================================================================#
            # Get the adjacency matrix
            adj_mat, mapped_edges_index = get_adj_matrix_snapshot(snapshot, parameters_exp)
        
            #======================================================================#
            #======================================================================#
            # Get predictions by the model
            # Nodes embeddings
            model_node_enc = model_node_enc.to(torch.device('cpu'))
            # If model is STM-GNN then put in correct device the memory
            if (parameters_exp['model_to_use'].lower() == "stm"):
                try:
                    for key in model_node_enc.spatial_in:
                        model_node_enc.spatial_in[key] = model_node_enc.spatial_in[key].to(torch.device('cpu'))
                    for key in model_node_enc.last_temporal:
                        model_node_enc.last_temporal[key] = model_node_enc.last_temporal[key].to(torch.device('cpu'))
                    for key in model_node_enc.temporal_in:
                        model_node_enc.temporal_in[key] = model_node_enc.temporal_in[key].to(torch.device('cpu'))
                    for key in model_node_enc.last_spatial:
                        model_node_enc.last_spatial[key] = model_node_enc.last_spatial[key].to(torch.device('cpu'))
                except:
                    model_node_enc.spatial_in = model_node_enc.spatial_in.to(torch.device('cpu'))
                    model_node_enc.last_temporal = model_node_enc.last_temporal.to(torch.device('cpu'))
                    model_node_enc.temporal_in = model_node_enc.temporal_in.to(torch.device('cpu'))
                    model_node_enc.last_spatial = model_node_enc.last_spatial.to(torch.device('cpu'))
                model_node_enc.temporal_module = model_node_enc.temporal_module.to(torch.device('cpu'))
                model_node_enc.spatial_module = model_node_enc.spatial_module.to(torch.device('cpu'))
                try:
                    model_node_enc.time_encoder = model_node_enc.time_encoder.to(torch.device('cpu'))
                except:
                    print(f"\n===>The loaded STM-GNN trained model does not have any time encoder (becasue time_dim in its parameters was set to 0 during training)\n")
            if (parameters_exp['model_to_use'].lower() in ["stm", "tgn", "graphsage", "gat", "gcn"]):
                # IMPORTANT: HERE WE MUST NOT USE mapped_edges_index AS THE MODEL HANDLES IT USING true_nodes_ids and edge_index
                node_embed = model_node_enc(x, true_nodes_ids, edge_index, edge_attr, timestamps)  
            elif (parameters_exp['model_to_use'].lower().lower() == "mlp"):
                node_embed = model_node_enc(x) 
            elif (parameters_exp['model_to_use'].lower().lower() == "simplegnn"):
                node_embed = model_node_enc(x, mapped_edges_index)
            else:
                node_embed = model_node_enc(x, mapped_edges_index, edge_attr)
        
            # Add the time dimension to the node embeddings
            # Only needed if doing auto-differentiation with respect to time
            if (parameters_exp['epidemio_informed']):
                if (parameters_exp['epidemio_informed_method'].lower() == 'autodiff_wrt_time_odes_residuals'):
                    if (model_node_enc.hetero_mode):
                        shared_timestamp = torch.nn.Parameter(torch.tensor(timestamps['Patient'][0]), requires_grad=True)
                        for node_type in timestamps.keys():
                            timestamps[node_type] = shared_timestamp.expand(node_embed[node_type].shape[0], 1)  # shape: [num_nodes, 1]
                            node_embed[node_type] = torch.cat([timestamps[node_type], node_embed[node_type]], axis=1)
                    else:
                        shared_timestamp = torch.nn.Parameter(torch.tensor(timestamps[0]), requires_grad=True)
                        timestamps = shared_timestamp.expand(node_embed.shape[0], 1)  # shape: [num_nodes, 1]
                        node_embed = torch.cat([timestamps, node_embed], axis=1)
                elif (parameters_exp['epidemio_informed_method'].lower() == 'odes_next_state_residuals'):
                    # IMPORTANT: no need to add time step value in the feature vector as we are not going to differentiate with respect to time!
                    pass
                else:
                    raise ValueError(f"\nEpdemio-informed approach {parameters_exp['epidemio_informed_method']} is not valid.\n")
        
        
            # Getting the prediction
            # Predictor of the states of the patients (nodes)
            seir_states_scores = model_compart_pred(node_embed)
            if (model_node_enc.hetero_mode):
                # We ONLY USE THE PATIENT NODES FOR PREDICTION
                node_embed = node_embed['Patient']
                seir_states_scores = seir_states_scores['Patient']
                y = y['Patient']
            # Getting the output probabilities
            if (parameters_exp['loss_function'].lower() == "evidentiallearningloss"):
                alphas = seir_states_scores + 1 # seir_states_scores is the evidence in this case
                preds_current_states_probs = alphas / torch.sum(alphas, dim=1, keepdim=True).unsqueeze(0)
            else:
                preds_current_states_probs = torch.nn.functional.softmax(seir_states_scores, dim=1).unsqueeze(0)
        
            
            #======================================================================#
            #======================================================================#
            # VERY IMPORTANT: EVEN WHEN FORECAST >= 2, WE ONLY CONSIDER THE NEXT DAY
            # VERY IMPORTANT: FOR THE COMPUTATIONS
            # Get the true local infection hazard
            I = gt_current_states_probs[:, :, 2]
            lambdas = (adj_mat @ I.unsqueeze(2)).squeeze(2)
            lambdas = (beta_gt*lambdas).detach().numpy().squeeze()
            local_infection_hazard_all_days['True'].append(lambdas)
        
            # Getting the predicted local infection hazard
            if (len(preds_current_states_probs.shape) == 4): # Forecast horizon >= 2
                print("\n\nWARNING: Even when forecast horizon >= 2, we only consider the next day for the computations of this metric !")
                # In this case, preds_current_states_probs is of shape (bs, num_nodes, num_states, forecast_horizon)
                I_pred = preds_current_states_probs[:, :, 2, 0]
            else: # Forecast horizon = 1
                I_pred = preds_current_states_probs[:, :, 2]

            lambdas_pred = (adj_mat @ I_pred.unsqueeze(2)).squeeze(2)
            learned_beta = torch.nn.functional.softplus(epi_params)[0]
            lambdas_pred = (learned_beta*lambdas_pred).detach().numpy().squeeze()
            local_infection_hazard_all_days['Pred'].append(lambdas_pred)
        
            # Compare both local infection forces
            # MSE
            if ('MSE' not in epi_params_metrics_over_time):
                epi_params_metrics_over_time['MSE'] = []
            mse = mean_squared_error(lambdas, lambdas_pred)
            epi_params_metrics_over_time['MSE'].append(mse)
            # Pearson Correlation Coefficient
            if ('PCC' not in epi_params_metrics_over_time):
                epi_params_metrics_over_time['PCC'] = []
            pcc = stats.pearsonr(lambdas, lambdas_pred)
            epi_params_metrics_over_time['PCC'].append(pcc)

    return local_infection_hazard_all_days, epi_params_metrics_over_time


#======================================================================#
#=================================MAIN=================================#
#======================================================================#
def main():
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_folder', required=True, help="Path to the folder containing the results of the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    
    #======================================================================#
    #===============================Load data===============================#
    #======================================================================#
    # Open the results file
    results_file = results_folder + "/metrics/final_results_all_repetitions_0.hdf5"


    #======================================================================#
    #=====================Parameters of the experiment=====================#
    #======================================================================#
    # Parameters of the experiment
    try:
        parameters_file = results_folder + f"/params_exp/params_0.pth"
        with open(parameters_file, 'rb') as pf:
            parameters_exp = pickle.load(pf)
    except:
        parameters_file = results_folder + f"/params_exp/params_0.yaml"
        with open(parameters_file, 'r') as file:
            parameters_exp = yaml.safe_load(file)
    #======================================================================#
    #=========Analysis of GLOBAL learned epidemiological parameters=========#
    #======================================================================#
    #======================================================================#
    # Number of epidemiological parameters
    if ('murcia' in results_folder.lower()):
        """
        # NEW VERSION
        n_epi_params = 5

        beta_gt = 1/(0.435 + 0.24)
        los_gt = 4.254
        dis_rate_gt = 1/los_gt
        alpha_gt = 1/2.5
        mu_gt = 0.027
        gamma_gt = 1 - mu_gt

        true_params = np.array(
                                [
                                    beta_gt, 
                                    dis_rate_gt, 
                                    alpha_gt, 
                                    gamma_gt, 
                                    mu_gt, 
                                ]
                            )
        """
        # Last version
        true_params = {}
        for preprocessed_data_path in parameters_exp['multiple_hdf5_dataset_filenames']:
            # Getting the dataset ID and path to the raw data
            split_path = preprocessed_data_path.split('/')
            DS_ID = None
            preprocessd_folder_ID_in_split_path = None
            for i_el in range(len(split_path)):
                el = split_path[i_el]
                if ('dataset' in el.lower()):
                    DS_ID = int(el.split('_')[-1])
                    break
                if (el == 'preprocessed'):
                    preprocessd_folder_ID_in_split_path = i_el
            if (DS_ID is not None):
                print(f"\n\n\n\n====================================================================================================")
                print(f"\n=========> Getting the epidemiological parameters of DS {DS_ID} \n")
            else:
                raise RuntimeError("Could not identify the dataset ID, so it is not possible to determine the GT epidemiological params")
            raw_data_path = '/'.join(split_path[:preprocessd_folder_ID_in_split_path]) + '/raw_data/'
            # Estimate the true parameters
            patients_data_path = raw_data_path + f'/patients_{DS_ID}.csv'
            locations_data_path = raw_data_path + f'/locations_{DS_ID}.csv'
            movement_data_path = raw_data_path + f'/movements_{DS_ID}.csv'
            patients_data_df = get_patients_data(patients_data_path)
            locations_info, main_parent_places = load_locations_data(locations_data_path=locations_data_path)
            movement_data_restructured,\
            places_list,\
            specific_places_list,\
            rooms_ward_mapping,\
            one_hot_enc_main_places,\
            inv_one_hot_enc_main_places,\
            one_hot_enc_specific_places,\
            inv_one_hot_enc_specific_places = load_movement_data(movement_data_path, locations_info)
            #RATES_IN_PER_DAY = False
            RATES_IN_PER_DAY = True
            epidemiological_params_for_csv = estimate_epidemic_params_from_data(
                                                                                    patients_data_df=patients_data_df,
                                                                                    movement_data=movement_data_restructured,
                                                                                    rates_in_per_day=RATES_IN_PER_DAY
                                                                                )      
            
            beta = epidemiological_params_for_csv['BETA']
            A = epidemiological_params_for_csv['A']
            A_S = epidemiological_params_for_csv['A_S']
            A_E = epidemiological_params_for_csv['A_E']
            A_I = epidemiological_params_for_csv['A_I']
            A_R = epidemiological_params_for_csv['A_R']
            A_NS = epidemiological_params_for_csv['A_NS']
            dis_rate = epidemiological_params_for_csv['DIS_RATE']
            alpha = epidemiological_params_for_csv['ALPHA']
            gamma = epidemiological_params_for_csv['GAMMA']
            mu = epidemiological_params_for_csv['MU']
            
            true_params[DS_ID] = np.array(
                                            [
                                                beta,
                                                A,
                                                A_S,
                                                A_E,
                                                A_I,
                                                A_R,
                                                A_NS,
                                                dis_rate,
                                                alpha,
                                                gamma,
                                                mu
                                            ]
                                        )
        
    else:
        n_epi_params = 3
        beta_gt = 0.0034391490028549392
        sigma_gt = 0.17119796091758707
        gamma_gt = 0.06853741496598639
        true_params = np.array([beta_gt, sigma_gt, gamma_gt])

    # Loading the models files
    models_dicts = {}
    for model_file in os.listdir(results_folder + "/model/"):
        models_dicts[model_file] = torch.load(results_folder + "/model/" + model_file, weights_only=False, map_location=torch.device('cpu'))

    # Getting the parameters fo each model
    cosine_similarities = []
    optimal_scales = []
    error_norms = []
    for model_name in models_dicts:
        if ('epi_params' in model_name.lower()):
            # Getting the dataset ID if present
            if ('dataset' in model_name.lower()):
                DS_ID = int(model_name.split('Dataset-')[-1].split('_.pth')[0])
            else:
                DS_ID = None

            # Getting the model and learned parameters
            tmp_model = models_dicts[model_name]['model']
            params = torch.nn.functional.softplus(tmp_model.epi_params).detach().numpy()

            # Getting the true params
            if (type(true_params) == dict):
                if (DS_ID is not None):
                    gt_params = true_params[DS_ID]
                else:
                    raise RuntimeError("Could not identify the dataset where the model was trained, so it is not possible to determine the GT epidemiological params")
            else:
                gt_params = true_params

            # Selecting onlY the parameters learned by the model based on the used epidemio-informed model as for the MURCIA DATASET they change
            if ('murcia' in results_folder.lower()):
                if (parameters_exp['epidemio_informed_method'].lower() == "odes_next_state_residuals"):
                    # Here we learn only 5 epidemiological parameters: beta, dis_rate, alpha, gamma, mu
                    gt_params = np.array([gt_params[0], gt_params[7], gt_params[8], gt_params[9], gt_params[10]])
                elif (parameters_exp['epidemio_informed_method'].lower() == "autodiff_wrt_time_odes_residuals"):
                    # Here we learn the 11 epidemiological parameters
                    pass 
                else:
                    raise ValueError(f"Computation of metrics for learned epidemiological parameters is not valid for epidemio informed models of type: {parameters_exp['epidemio_informed_method']}")

            print(f"\n===> True epidemiological parameters: {gt_params}\n")
            print(f"\t===> Learned epidemiological parameters: {params}\n")
            #breakpoint()

            res_eval_learned_params = evaluate_learned_params(params, gt_params)
            cosine_similarities.append(res_eval_learned_params['cosine_similarity'])
            optimal_scales.append(res_eval_learned_params['optimal_scale'])
            error_norms.append(res_eval_learned_params['error_norm'])
            print(f"\nTrue parameters: {gt_params}")
            print(f"Parameters of the model AFTER softplus for model {model_name}: {params}")
            print(f"\tCosine similarity with true parameters: {cosine_similarities[-1]}")
            print(f"\tOptimal scale with true parameters: {optimal_scales[-1]}")
            print(f"\tError norms with true parameters: {error_norms[-1]}")

    print(f"\n=========>Mean cosine similarity: {np.mean(cosine_similarities)} +- {np.std(cosine_similarities)}\n")
    print(f"\n=========>Mean optimal scale: {np.mean(optimal_scales)} +- {np.std(optimal_scales)}\n")
    print(f"\n=========>Mean error norm: {np.mean(error_norms)} +- {np.std(error_norms)}\n")


    #======================================================================#
    # Comparing the local infection hazard (learned and predicted vs GT)
    # Load the test data
    # Create instance of the experiment
    tmp_exp = InfectionRiskPred(parameters_exp)

    # Dataset loading
    #tmp_exp.createTorchDatasets(prefix_paths="../.")
    tmp_exp.createTorchDatasets(prefix_paths="")

    # Patients features correspondence
    if (parameters_exp['dataset_name'].lower() == 'murcia'):
        fn_patient_features_names_idx = "/".join(parameters_exp['hdf5_dataset_filename'].split("/")[:-1]) + "/patient_features_names_idx_0.pkl"
        with open(fn_patient_features_names_idx, mode='rb') as pf:
            patient_features_names_idx = pickle.load(pf)
    else:
        patient_features_names_idx = None

    # Loading the models for the prediction
    tmp_exp.modelCreation()
    untrained_models = tmp_exp.models

    # Get the test data
    test_ds = tmp_exp.test_ds
    del tmp_exp

    # Get the local infection hazard
    REP_ID = 0
    DS_ID = 0
    if (type(true_params) == dict):
        beta_gt = true_params[DS_ID][0]
    local_infection_hazard_all_days, epi_params_metrics_over_time = get_local_infection_hazard(
                                                                                                    dataset=test_ds,
                                                                                                    exp_folder=results_folder,
                                                                                                    patient_features_names_idx=patient_features_names_idx,
                                                                                                    dataset_type=parameters_exp['dataset_name'],
                                                                                                    beta_gt=beta_gt,
                                                                                                    untrained_models=untrained_models,
                                                                                                    rep_ID=REP_ID,
                                                                                                    DS_ID=DS_ID
                                                                                                )
    local_infection_hazard_all_days['True'] = np.concatenate(local_infection_hazard_all_days['True'])
    local_infection_hazard_all_days['Pred'] = np.concatenate(local_infection_hazard_all_days['Pred'])

    # Computing the MSE and PCC using ALL the local infection forces from ALL the days
    # MSE
    all_days_mse = mean_squared_error(local_infection_hazard_all_days['True'], local_infection_hazard_all_days['Pred'])
    print(f"\n=========>MSE between the true local infection forces of ALL patients and ALL days and the predicted ones: {all_days_mse}\n")
    # PCC
    all_days_pcc = stats.pearsonr(local_infection_hazard_all_days['True'], local_infection_hazard_all_days['Pred'])
    r, p_value = all_days_pcc.statistic, all_days_pcc.pvalue 
    print(f"\n=========>PCC between the true local infection forces of ALL patients and ALL days and the predicted ones: {all_days_pcc.statistic} (p-value {all_days_pcc.pvalue}, confidence interval {all_days_pcc.confidence_interval})\n")

    # Scatter plot
    plt.figure()
    plt.scatter(local_infection_hazard_all_days['True'], local_infection_hazard_all_days['Pred'], alpha=0.7, edgecolor="k")
    plt.title(f"Scatter Plot with Pearson r = {r:.2f} (p = {p_value:.3f})")
    plt.xlabel("True local infection hazard values")
    plt.ylabel("Pred local infection hazard values")

    # Add line of best fit for visualization
    m, b = np.polyfit(local_infection_hazard_all_days['True'], local_infection_hazard_all_days['Pred'], 1)
    plt.plot(local_infection_hazard_all_days['True'], m*local_infection_hazard_all_days['True'], + b, color="red", linewidth=2, label="Best fit line")
    plt.legend()
    plt.show()

    # Save values for plot outside this code
    local_inf_hazard = pd.DataFrame(
                                    {
                                        'TrueLocalInfectionHazard': local_infection_hazard_all_days['True'],
                                        'PredLocalInfectionHazard': local_infection_hazard_all_days['Pred']
                                    }
                                    )
    local_inf_hazard.to_csv(results_folder + '/local_inf_hazard.csv', index=False)


if (__name__=='__main__'):
    main()