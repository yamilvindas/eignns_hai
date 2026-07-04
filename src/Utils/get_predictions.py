import os
import re
import h5py
import yaml
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,\
                            matthews_corrcoef,\
                            roc_curve,\
                            roc_auc_score,\
                            balanced_accuracy_score,\
                            average_precision_score
import torch
from src.Experiments.InfectionRiskPredMurcia import InfectionRiskPred
from src.Utils.classification_metrics import get_classification_metrics,\
                                             compute_ece


def main():
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_folder', required=True, help="Path to the folder containing the results of the experiment", type=str)
    ap.add_argument('--use_all_patient_for_pred', help="Use if want to use all the patient for prediction (and not only nosocomial cases)", action='store_true')
    ap.add_argument('--dataset_folder_use', help="Path to the dataset folder to use for evaluation.", type=str)
    ap.add_argument('--mask_valid_samples', help="Mask samples for prediction (usually, only consider samples that are not currently in the Infected state).", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    use_all_patient_for_pred = args['use_all_patient_for_pred']
    dataset_folder_use = args['dataset_folder_use']
    mask_valid_samples = args['mask_valid_samples']

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

    # Modify data path
    if (dataset_folder_use is not None):
        parameters_exp['hdf5_dataset_filename'] = dataset_folder_use + '/' + parameters_exp['hdf5_dataset_filename'].split('/')[-1]
        for i in range(len(parameters_exp['multiple_hdf5_dataset_filenames'])):
            current_ds_path = parameters_exp['multiple_hdf5_dataset_filenames'][i]
            parameters_exp['multiple_hdf5_dataset_filenames'][i] = dataset_folder_use + '/' + current_ds_path.split('/')[-1]

    #======================================================================#
    #==================Creating instance of the experiment==================
    #======================================================================#
    # Create instance
    tmp_exp = InfectionRiskPred(parameters_exp)

    # Create untrained models
    tmp_exp.modelCreation()

    # Make the device the CPU or GPU
    tmp_exp.device = 'cpu'
    #tmp_exp.device = 'cuda:0'

    # Mask patients for prediction
    tmp_exp.x = mask_valid_samples

    #======================================================================#
    #==========================Loading the models==========================
    #======================================================================#
    models_dicts = {}
    for model_file in os.listdir(results_folder + "/model/"):
        rep_ID = int(model_file.split('rep-')[-1].split('_')[0])
        if (rep_ID not in models_dicts):
            models_dicts[rep_ID] = {}
        for submodel in tmp_exp.models:
            if (submodel in model_file):
                models_dicts[rep_ID][submodel] = torch.load(results_folder + "/model/" + model_file, weights_only=False, map_location=torch.device('cpu'))

    #======================================================================#
    #========================Getting the predictions========================
    #======================================================================#
    # Get embeddings
    sorted_keys = sorted(list(models_dicts.keys()))
    #data_splits_to_use = ['Train', 'Val', 'Test']
    data_splits_to_use = ['Test']
    preds_per_rep = {data_split_type: [] for data_split_type in data_splits_to_use}
    state_transition_metrics_per_rep = {data_split_type: [] for data_split_type in data_splits_to_use}
    for rep_ID in sorted_keys:
        print(f"\n\n\n========================================> Processing REPETITION {rep_ID} <========================================\n")
        # Create data
        if (len(tmp_exp.parameters_exp['multiple_hdf5_dataset_filenames']) == 1): 
            # In this case all the repetitions were trained and evaluated on the same datasets
            DS_ID = 0
        else:
            DS_ID = rep_ID

        tmp_exp.hdf5_dataset_filename = tmp_exp.parameters_exp['multiple_hdf5_dataset_filenames'][DS_ID]
        tmp_exp.current_dataset_ID = DS_ID
        if (use_all_patient_for_pred):
            if ('OnlyNosocForPred' in tmp_exp.hdf5_dataset_filename):
                # This pattern looks for 'OnlyNosocForPred-', 
                # then captures everything until the next underscore
                match = re.search(r"OnlyNosocForPred-(True|False|[^_]+)", tmp_exp.hdf5_dataset_filename)
                if match:
                    raw_val = match.group(1)
                    only_nosocomial_for_pred = raw_val.lower() == 'true'
                if (only_nosocomial_for_pred):
                    tmp_exp.hdf5_dataset_filename = re.sub(r"(OnlyNosocForPred-)[^_]+", r"\1False", tmp_exp.hdf5_dataset_filename)
                else:
                    # Nothing to do as we already take into account ALL patients
                    pass
            else:
                # Get base name
                start = "_".join(tmp_exp.hdf5_dataset_filename.split('_')[:-1])
                end = tmp_exp.hdf5_dataset_filename.split('_')[-1]
                tmp_exp.hdf5_dataset_filename = start + '_OnlyNosocForPred-False_' + end
        else:
            if ('OnlyNosocForPred' in tmp_exp.hdf5_dataset_filename):
                # To use if the HDF5 datasets are renamed with OnlyNosocForPred-True when only using nosocomial cases for pred
                #tmp_exp.hdf5_dataset_filename = re.sub(r"(OnlyNosocForPred-)[^_]+", r"\1True", tmp_exp.hdf5_dataset_filename)
                # To use in the other cases
                tmp_exp.hdf5_dataset_filename = re.sub(r"_OnlyNosocForPred-[^_]+", "", tmp_exp.hdf5_dataset_filename)

        tmp_exp.h5_file = h5py.File(tmp_exp.hdf5_dataset_filename, 'r')

        # Initialize everything for forward pass
        tmp_exp.createTorchDatasets()
        tmp_exp.dataloadersCreation()
        tmp_exp.createOptimizer()
        tmp_exp.createLossFunction()
        tmp_exp.addClassWeightsLoss()

        # Count number of samples that can be used for predictions in ALL days
        n_counts = 0
        for batch in tqdm(tmp_exp.test_loader):
            snapshot = batch
            # Get inputs and labels
            if (tmp_exp.models['model_node_enc'].hetero_mode):
                x = {key: torch.tensor(snapshot[key].x).to(tmp_exp.device).float() for key in snapshot.node_types}
                y = {key: torch.tensor(snapshot[key].y).to(tmp_exp.device).long() for key in snapshot.node_types}
                if (tmp_exp.parameters_exp["predict_inf_risk"]):
                    y_inf_risk = {key: torch.tensor(snapshot[key]["inf_risk_targets_dicts"]).to(tmp_exp.device).long() for key in snapshot.node_types}
                edge_index = {key: torch.tensor(snapshot[key].edge_index).to(tmp_exp.device).long() for key in snapshot.edge_types}
                edge_attr = {key: torch.tensor(snapshot[key].edge_attr).to(tmp_exp.device).float() for key in snapshot.edge_types}
                timestamps = {key: torch.tensor(snapshot[key].timestamps).to(tmp_exp.device).float() for key in snapshot.node_types}
                true_nodes_ids = {key: torch.tensor(snapshot[key].ids).to(tmp_exp.device) for key in snapshot.node_types} 
                if (tmp_exp.parameters_exp['dataset_name'].lower() == 'hug'):
                    tag_use_for_pred = {key: torch.tensor(snapshot[key].tag_use_node_for_pred_dicts).to(torch.device(tmp_exp.device)) for key in snapshot.node_types} 
                    n_counts += int(tag_use_for_pred['Patient'].sum().cpu().detach().numpy())
            else:
                x = torch.tensor(snapshot.x).to(tmp_exp.device).float()
                y = torch.tensor(snapshot.y).to(tmp_exp.device).long()
                if (tmp_exp.parameters_exp["predict_inf_risk"]):
                    y_inf_risk = torch.tensor(snapshot.inf_risk_targets).to(tmp_exp.device).long()
                edge_index = torch.tensor(snapshot.edge_index).to(tmp_exp.device).long()
                edge_attr = torch.tensor(snapshot.edge_attr).to(tmp_exp.device).float()
                timestamps = torch.tensor(snapshot.timestamps).to(tmp_exp.device).float()
                true_nodes_ids = torch.tensor(snapshot.ids).to(tmp_exp.device)

        print(f"\n\n==========> Number of samples that can be used for prediction in ALL days: {n_counts} \n\n")

        # Load models weights
        tmp_exp.modelCreation() # Reinitialize models weights
        for submodel in tmp_exp.models:
            tmp_exp.models[submodel].load_state_dict(models_dicts[rep_ID][submodel]['model_state_dict'])
            #tmp_exp.models[submodel] = tmp_exp.models[submodel].to(torch.device('cpu'))

        # Predictions
        data_loaders = {'Train': tmp_exp.train_loader, 'Val': tmp_exp.val_loader, 'Test':tmp_exp.test_loader}
        #data_loaders = {'Test':tmp_exp.test_loader}
        for data_loader_type in data_splits_to_use:
            print(f"\n\n\n================> Processing {data_loader_type} dataset <================\n")
            # Getting the predictions
            losses, preds = tmp_exp.evalCurrentModel(data_loaders[data_loader_type], -1)
            preds_per_rep[data_loader_type].append(preds)
















            # Predicting infection risk using directly the predicted epidemiological states
            true_risk_from_states = []
            pred_risk_from_states = []
            for i in range(len(preds)):
                day_true_risk_from_states = []
                day_pred_risk_from_states = []
                for tmp_node_i in range(preds[i]['true_seir_states'].shape[0]):
                    day_true_risk_from_states.append(1 in preds[i]['true_seir_states'][tmp_node_i])
                    day_pred_risk_from_states.append(1 in preds[i]['pred_seir_states_probs'].argmax(dim=1)[tmp_node_i])
                true_risk_from_states.append(torch.tensor(day_true_risk_from_states))
                pred_risk_from_states.append(torch.tensor(day_pred_risk_from_states))
            # breakpoint()
















            # Computing change of state performance
            if (tmp_exp.models['model_node_enc'].hetero_mode):
                all_true_inf_risk_class = torch.cat([preds[i]['true_inf_risk_class'] for i in range(len(preds))]).detach().numpy()
                all_inf_risk_pred_probs = torch.cat([preds[i]['seir_states_probs_inf_risk'] for i in range(len(preds))]).detach().numpy()[:, 1]
                all_current_epi_state = torch.cat([preds[i]['current_epi_state'] for i in range(len(preds))]).detach().numpy()
            else:
                all_true_inf_risk_class = torch.cat([preds[i]['true_seir_states'] for i in range(len(preds))]).detach().numpy()
                all_inf_risk_pred_probs = torch.cat([preds[i]['pred_seir_states_probs'] for i in range(len(preds))]).detach().numpy()[:, 1]
                all_current_epi_state = torch.cat([preds[i]['current_epi_state'] for i in range(len(preds))]).detach().numpy()


            tmp_a = torch.cat(true_risk_from_states)
            tmp_b = torch.cat(pred_risk_from_states)
            tmp_c = []
            for tmp_i in range(len(preds)):
                tmp_mask = (preds[i]['current_epi_state'].argmax(axis=1) != 0)
                tmp_c.append(balanced_accuracy_score(true_risk_from_states[i].to(torch.int)[tmp_mask], pred_risk_from_states[i].to(torch.int)[tmp_mask], adjusted=True))
            # breakpoint()



            # Convert one-hot back to categorical for easy filtering
            # 0: Susceptible, 1: Infected, 2: Recovered
            states = all_current_epi_state.argmax(axis=1)
            results_state_transition_current_data_split = {}
            # Forecast horizon for SocioPattern (computed dynamically)
            if (parameters_exp['dataset_name'].lower() == 'hug'):
                state_names = {0: 'Susceptible', 1: 'Infected', 2: 'Recovered'}
            elif (parameters_exp['dataset_name'].lower() == 'sociopatterns'):
                state_names = {0: 'Susceptible', 1: 'Exposed', 2: 'Infected', 3: 'Recovered'}
            elif (parameters_exp['dataset_name'].lower() == 'murcia'):
                state_names = {0: 'Susceptible', 1: 'Exposed', 2: 'Infected', 3: 'Recovered', 4: 'Deceased', 5:'NonSusceptible'}

            # Performance if the state is S or R
            mask = (states != 1)
            state_y_true = all_true_inf_risk_class[mask]
            state_y_probs = all_inf_risk_pred_probs[mask]
            state_y_preds = (state_y_probs > 0.5).astype(int)
            auc = roc_auc_score(state_y_true, state_y_probs)
            b_acc = balanced_accuracy_score(state_y_true, state_y_preds, adjusted=True)
            # breakpoint()

            # Performance per state
            for state_val, name in state_names.items():
                mask = (states == state_val)
                if mask.sum() > 0:
                    # Filter data for this state only
                    state_y_true = all_true_inf_risk_class[mask]
                    state_y_probs = all_inf_risk_pred_probs[mask]
                    state_y_preds = (state_y_probs > 0.5).astype(int)
                    
                    # Calculate metrics if both classes are present in the subset
                    if len(np.unique(state_y_true)) > 1:

                        if (tmp_exp.parameters_exp['dataset_name'].lower() in ['sociopatterns', 'hug']):
                            auc = roc_auc_score(state_y_true, state_y_probs)
                            b_acc = balanced_accuracy_score(state_y_true, state_y_preds, adjusted=True)
                        elif (tmp_exp.parameters_exp['dataset_name'].lower() == 'murcia'):
                            auc = roc_auc_score(state_y_true, state_y_probs, multi_class='ovo', average="macro")
                            b_acc = balanced_accuracy_score(state_y_true, state_y_preds, adjusted=True)
                    else:
                        auc = float('nan') # Occurs if, e.g., all Infected stay Positive
                        b_acc = float('nan') # Occurs if, e.g., all Infected stay Positive
                    results_state_transition_current_data_split[name] = {
                                                                            'AUC': auc,
                                                                            'BalancedAccuracy': b_acc,
                                                                            'Count': mask.sum()
                                                                         }                                                      
            state_transition_metrics_per_rep[data_loader_type].append(results_state_transition_current_data_split)

            #breakpoint()

    for data_loader_type in state_transition_metrics_per_rep:
        print(f"\n\n=========={data_loader_type.upper()} DATASET==========")
        auc_per_sir_state_per_rep = {}
        b_acc_per_sir_state_per_rep = {}
        for rep_ID in range(len(state_transition_metrics_per_rep[data_loader_type])):
            for current_epi_state in state_transition_metrics_per_rep[data_loader_type][rep_ID]:
                if (current_epi_state not in auc_per_sir_state_per_rep):
                    auc_per_sir_state_per_rep[current_epi_state] = []
                    b_acc_per_sir_state_per_rep[current_epi_state] = []
                auc_per_sir_state_per_rep[current_epi_state].append(state_transition_metrics_per_rep[data_loader_type][rep_ID][current_epi_state]['AUC'])
                b_acc_per_sir_state_per_rep[current_epi_state].append(state_transition_metrics_per_rep[data_loader_type][rep_ID][current_epi_state]['BalancedAccuracy'])
        for current_epi_state in auc_per_sir_state_per_rep:
            print(f"\t===> AUC for patient in initial state {current_epi_state.upper()}: {round(np.mean(auc_per_sir_state_per_rep[current_epi_state])*100, 3)} \pm {round(np.std(auc_per_sir_state_per_rep[current_epi_state])*100, 3)}")
            print(f"\t===> Balanced Accuracy for patient in initial state {current_epi_state.upper()}: {round(np.mean(b_acc_per_sir_state_per_rep[current_epi_state])*100, 3)} \pm {round(np.std(b_acc_per_sir_state_per_rep[current_epi_state])*100, 3)}")
        print("\n\n")

                        
    #breakpoint()









    #======================================================================#
    #============================Compute Metrics============================
    #======================================================================#
    metrics_per_rep = {data_split_type: [] for data_split_type in data_splits_to_use}
    for rep_ID in tqdm(range(len(sorted_keys))):
        for data_split_type in list(preds_per_rep.keys()):
            targets_all_days = []
            preds_all_days = []
            preds_all_days_probs = []
            targets_inf_risk_all_days = []
            preds_inf_risk_all_days = []
            preds_inf_risk_all_days_probs = []
            n_days = len(preds_per_rep[data_split_type][rep_ID])
            list_days = sorted([i for i in range(n_days)])
            for day in list_days:
                # Getting targets and predictions
                targets_all_days.append(preds_per_rep[data_split_type][rep_ID][day]['true_seir_states'].cpu().detach().numpy())
                preds_all_days.append(np.argmax(preds_per_rep[data_split_type][rep_ID][day]['pred_seir_states_probs'].cpu().detach().numpy(), axis=1))
                preds_all_days_probs.append(preds_per_rep[data_split_type][rep_ID][day]['pred_seir_states_probs'].cpu().detach().numpy())
                targets_inf_risk_all_days.append(preds_per_rep[data_split_type][rep_ID][day]['true_inf_risk_class'].cpu().detach().numpy())
                preds_inf_risk_all_days.append(np.argmax(preds_per_rep[data_split_type][rep_ID][day]['seir_states_probs_inf_risk'].cpu().detach().numpy(), axis=1))
                preds_inf_risk_all_days_probs.append(preds_per_rep[data_split_type][rep_ID][day]['seir_states_probs_inf_risk'].cpu().detach().numpy())

            targets_all_days = np.concatenate(targets_all_days, axis=0)
            preds_all_days = np.concatenate(preds_all_days, axis=0)
            preds_all_days_probs = np.concatenate(preds_all_days_probs, axis=0)
            targets_inf_risk_all_days = np.concatenate(targets_inf_risk_all_days, axis=0)
            preds_inf_risk_all_days = np.concatenate(preds_inf_risk_all_days, axis=0)
            preds_inf_risk_all_days_probs = np.concatenate(preds_inf_risk_all_days_probs, axis=0)

            print(f"\n\n=========>Number of predictions for ALL days for repetition {rep_ID}: {len(targets_all_days)}\n")

            # Reshaping targets_all_days, preds_all_days, and preds_all_days_probs in case of having a forecast horizon > 1
            if (len(preds_all_days_probs.shape) > 2):
                targets_all_days = targets_all_days.reshape(-1)
                preds_all_days = preds_all_days.reshape(-1)
                preds_all_days_probs = preds_all_days_probs.transpose(0, 2, 1).reshape(-1, preds_all_days_probs.shape[1])

            # For HUG COVID dataset, we only keep the valid labels
            if (tmp_exp.parameters_exp['dataset_name'].lower() == 'hug'):
                mask_valid_labels = targets_all_days != -1
                if (tmp_exp.parameters_exp['forecast_horizon'] == 1):
                    mask_valid_labels = mask_valid_labels[:, 0]
                    targets_all_days = targets_all_days[mask_valid_labels].squeeze()
                else:
                    targets_all_days = targets_all_days[mask_valid_labels]
                preds_all_days_probs = preds_all_days_probs[mask_valid_labels]
                preds_all_days = preds_all_days[mask_valid_labels]
            
            # Getting the metrics for the epoch
            #print(f"\n\n=========> {data_split_type.upper()} DATA SPLIT FOR EPOCH {epoch} <=========\n")
            # MCC, F1-Score and Balanced Accuracy
            N_UNIQUE_CLASSES_INF_RISK = [i for i in range(preds_inf_risk_all_days_probs.shape[1])]
            sensitivity_inf_risk, specificity_inf_risk, mcc_inf_risk, f1_score_val_inf_risk, balanced_acc_inf_risk = get_classification_metrics(
                                                                                                    targets_inf_risk_all_days,
                                                                                                    preds_inf_risk_all_days,
                                                                                                    data_split_type,
                                                                                                    N_UNIQUE_CLASSES_INF_RISK,
                                                                                                    verbose=False,
                                                                                                    print_classification_report=False
                                                                                                )
            # AUC
            auc_inf_risk = roc_auc_score(targets_inf_risk_all_days, preds_inf_risk_all_days_probs[:, 1], average="macro")*100
            # ECE
            # Get the expected calibration error ECE FOR THE INFECTION RISK
            N_BINS = 10 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
            #N_BINS = 15 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
            ece_inf_risk = compute_ece(
                                                probs=preds_inf_risk_all_days_probs,
                                                labels=targets_inf_risk_all_days,
                                                n_bins=N_BINS
                                        )
            # Final metrics
            current_metrics = {
                                'Sensitivity': sensitivity_inf_risk,
                                'Specificity': specificity_inf_risk,
                                'MCC': mcc_inf_risk,
                                'F1Score': f1_score_val_inf_risk,
                                'BalancedAccuracy': balanced_acc_inf_risk,
                                'AUC':auc_inf_risk,
                                'ECE':ece_inf_risk
                            }
            metrics_per_rep[data_split_type].append(current_metrics)

    # Compute mean metrics
    sensitivity_test = [metrics_per_rep['Test'][rep_ID]['Sensitivity'] for rep_ID in range(len(metrics_per_rep['Test']))]
    specificity_test = [metrics_per_rep['Test'][rep_ID]['Specificity'] for rep_ID in range(len(metrics_per_rep['Test']))]
    balanced_acc_test = [metrics_per_rep['Test'][rep_ID]['BalancedAccuracy'] for rep_ID in range(len(metrics_per_rep['Test']))]
    auc_test = [metrics_per_rep['Test'][rep_ID]['AUC'] for rep_ID in range(len(metrics_per_rep['Test']))]
    ece_test = [metrics_per_rep['Test'][rep_ID]['ECE'] for rep_ID in range(len(metrics_per_rep['Test']))]

        
    print("\n\n========================================> Infection Risk Metrics <========================================\n")
    print(f"\n\t=========> Sensitivity: {round(np.mean(sensitivity_test), 2)} \pm {round(np.std(sensitivity_test), 2)}\n")
    print(f"\n\t=========> Specificity: {round(np.mean(specificity_test), 2)} \pm {round(np.std(specificity_test), 2)}\n")
    print(f"\n\t=========> Balanced Accuracy: {round(np.mean(balanced_acc_test), 2)} \pm {round(np.std(balanced_acc_test), 2)}\n")
    print(f"\n\t=========> AUC: {round(np.mean(auc_test), 2)} \pm {round(np.std(auc_test), 2)}\n")
    print(f"\n\t=========> ECE: {round(np.mean(ece_test), 2)} \pm {round(np.std(ece_test), 2)}\n")
    print(f"\n\t=========> LaTeX: ${round(np.mean(sensitivity_test), 2)} \pm {round(np.std(sensitivity_test), 2)}$ & ${round(np.mean(specificity_test), 2)} \pm {round(np.std(specificity_test), 2)}$ & ${round(np.mean(balanced_acc_test), 2)} \pm {round(np.std(balanced_acc_test), 2)}$ & ${round(np.mean(auc_test), 2)} \pm {round(np.std(auc_test), 2)}$ & ${round(np.mean(ece_test), 2)} \pm {round(np.std(ece_test), 2)}$\n")


if __name__=='__main__':
   main() 