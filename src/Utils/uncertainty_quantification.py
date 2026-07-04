"""
    Plot the metrics and results of an experiment
"""
import os
import h5py
import yaml
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef,\
                            f1_score,\
                            balanced_accuracy_score,\
                            roc_auc_score



def get_evidential_uncertainties(h5_results_file, epoch_to_use, data_split_to_use, base_name_main_group='Rep-'):
    # Getting the average epistemic, aleatoric and total uncertainty for the different repetitions
    n_repetitions = len(h5_results_file)
    if ("aleatoric_uncert" in h5_results_file[f"{base_name_main_group}0"]["Preds"][data_split_to_use]["Epoch-0"]["Day-0"]):
        n_days_split = len(h5_results_file[f"{base_name_main_group}0"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"])
        n_individuals = len(h5_results_file[f"{base_name_main_group}0"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"]["Day-0"]["aleatoric_uncert"])
        uncertainties_dict = {
                                'Epistemic': [ [ [None for _ in range(n_repetitions)] for _ in range(n_individuals)] for _ in range(n_days_split) ],
                                'Aleatoric': [ [ [None for _ in range(n_repetitions)] for _ in range(n_individuals)] for _ in range(n_days_split) ],
                                'Total': [ [ [None for _ in range(n_repetitions)] for _ in range(n_individuals)] for _ in range(n_days_split) ],
                                'Targets': [ [ [None for _ in range(n_repetitions)] for _ in range(n_individuals)] for _ in range(n_days_split) ],
                                'Preds': [ [ [None for _ in range(n_repetitions)] for _ in range(n_individuals)] for _ in range(n_days_split) ]
                            }    
        for rep_ID in range(n_repetitions):
            for day in range(len(h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"])):
                # Getting the uncertainties
                for individual_ID in range(len(h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['aleatoric_uncert'])):
                    uncertainties_dict["Epistemic"][day][individual_ID][rep_ID] = h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['epistemic_uncert'][individual_ID]
                    uncertainties_dict["Aleatoric"][day][individual_ID][rep_ID] = h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['aleatoric_uncert'][individual_ID]
                    uncertainties_dict["Total"][day][individual_ID][rep_ID] = h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['total_uncert'][individual_ID]
                    uncertainties_dict["Targets"][day][individual_ID][rep_ID] = h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['true_seir_states'][individual_ID]
                    uncertainties_dict["Preds"][day][individual_ID][rep_ID] = np.argmax(h5_results_file[f"{base_name_main_group}{rep_ID}"]["Preds"][data_split_to_use][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'][individual_ID])

    # Transform into numpy arrays
    uncertainties_dict["Epistemic"] = np.array(uncertainties_dict["Epistemic"])
    uncertainties_dict["Aleatoric"] = np.array(uncertainties_dict["Aleatoric"])
    uncertainties_dict["Total"] = np.array(uncertainties_dict["Total"])
    uncertainties_dict["Targets"] = np.array(uncertainties_dict["Targets"])
    uncertainties_dict["Preds"] = np.array(uncertainties_dict["Preds"])

    return uncertainties_dict


def get_uncertainties_per_rep_all_days(results_h5_file, epoch_to_use, base_name_main_group='Rep-'):
    """
        Get the uncertainties per repetition.

        Parameters:
        -----------
        results_h5_file: h5py._hl.files.File
            HDF5 file containing the results of the experiment.
        epoch_to_use: int
            Epoch to use to select the data.

        Returns:
        --------
        total_uncertainties_all_days_epoch_to_use: dict
            Dictionary containing the total uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        alea_uncertainty_all_days_epoch_to_use: dict
            Dictionary containing the aleatoric uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        epi_uncertainty_all_days_epoch_to_use: dict
            Dictionary containing the epistemic uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
    """
    data_splits = list(results_h5_file[f"{base_name_main_group}0"]["Preds"].keys())
    n_repetitions = len(results_h5_file)
    total_uncertainties_all_days_epoch_to_use = {rep_id: {data_split:None for data_split in data_splits} for rep_id in range(n_repetitions)}
    alea_uncertainty_all_days_epoch_to_use = {rep_id: {data_split:None for data_split in data_splits} for rep_id in range(n_repetitions)}
    epi_uncertainty_all_days_epoch_to_use = {rep_id: {data_split:None for data_split in data_splits} for rep_id in range(n_repetitions)}
    for rep_id in range(n_repetitions):
        for data_split in data_splits:
            total_uncertainty_all_days = []
            epi_uncertainty_all_days = []
            alea_uncertainty_all_days = []
            preds_all_days = []
            n_days = len(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"].keys())
            for day in range(n_days):
                # Getting targets and predictions
                total_uncertainty_all_days.append(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['total_uncert'])
                alea_uncertainty_all_days.append(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['aleatoric_uncert'])
                epi_uncertainty_all_days.append(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['epistemic_uncert'])
                preds_all_days.append(np.argmax(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'], axis=1))

            # Transform into arrays
            total_uncertainty_all_days = np.concatenate(total_uncertainty_all_days, axis=0)
            alea_uncertainty_all_days = np.concatenate(alea_uncertainty_all_days, axis=0)
            epi_uncertainty_all_days = np.concatenate(epi_uncertainty_all_days, axis=0)
            preds_all_days = np.concatenate(preds_all_days, axis=0)

            # Selecting the uncertainties of the PREDICTED CLASS$
            total_uncertainty_all_days = total_uncertainty_all_days[np.arange(preds_all_days.shape[0]), preds_all_days]
            alea_uncertainty_all_days = alea_uncertainty_all_days[np.arange(preds_all_days.shape[0]), preds_all_days]
            epi_uncertainty_all_days = epi_uncertainty_all_days[np.arange(preds_all_days.shape[0]), preds_all_days]
            
            
            # Adding them to output variables
            total_uncertainties_all_days_epoch_to_use[rep_id][data_split] = total_uncertainty_all_days
            alea_uncertainty_all_days_epoch_to_use[rep_id][data_split] = alea_uncertainty_all_days
            epi_uncertainty_all_days_epoch_to_use[rep_id][data_split] = epi_uncertainty_all_days



    return total_uncertainties_all_days_epoch_to_use,\
           alea_uncertainty_all_days_epoch_to_use,\
           epi_uncertainty_all_days_epoch_to_use


def plot_metric_vs_uncertainty(
                                uncertainty_all_days_per_rep,
                                targets_all_days_per_rep,
                                preds_all_days_per_rep,
                                preds_probs_all_days_per_rep=None,
                                metric_to_use='BalancedAccuracy',
                                rep_id=0,
                                data_split='Test'
                             ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT.
        
        Plots a perfomance metric against the uncertainty of the predictions

        Parameters:
        -----------
        uncertainty_all_days_per_rep: dict
            Dictionary containing the uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_all_days_per_rep: dict
            Dictionary containing the targets of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_all_days_per_rep: dict
            Dictionary containing the predictions of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_probs_all_days_per_rep: dict
            Dictionary containing the predictions probabilities of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information). ONLY NECESSARY
            when metric_to_use is AUC.
        metric_to_use: str
            Metric to use for the plot.
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Getting the useful data
    uncertainties = uncertainty_all_days_per_rep[rep_id][data_split]
    true_labels = targets_all_days_per_rep[rep_id][data_split]
    predictions = preds_all_days_per_rep[rep_id][data_split]
    predictions_probs = preds_probs_all_days_per_rep[rep_id][data_split]
    all_classes = np.unique(true_labels)
    
    # Define thresholds over uncertainty
    thresholds = np.linspace(uncertainties.min(), uncertainties.max(), 50)
    
    metric_values = []
    num_kept = []
    
    for t in thresholds:
        # Keep only predictions below the threshold
        mask = uncertainties <= t
        num_kept.append(np.sum(mask))
        
        if np.sum(mask) == 0:
            metric_values.append(np.nan)  # avoid empty slice
            continue
        
        filtered_true = true_labels[mask]
        filtered_pred = predictions[mask]
        filtered_pred_probs = predictions_probs[mask]

        if (metric_to_use.lower() == 'mcc'):
            metric = matthews_corrcoef(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'balancedaccuracy'):
            metric = balanced_accuracy_score(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'f1score'):
            metric = f1_score(filtered_true, filtered_pred, average="micro") 
        elif (metric_to_use.lower() == 'auc'):
            if (filtered_pred_probs.shape[1] == 2): # Binary classification
                metric = roc_auc_score(filtered_true, filtered_pred_probs[:, 1]) 
            else:
                metric = roc_auc_score(filtered_true, filtered_pred_probs, multi_class='ovo', average="macro", labels=all_classes)
        else:
            raise NotImplementedError()
        metric_values.append(metric)

    # Comput the metric for all the data
    if (metric_to_use.lower() == 'mcc'):
        metric_all = matthews_corrcoef(true_labels, predictions)
    elif (metric_to_use.lower() == 'balancedaccuracy'):
        metric_all = balanced_accuracy_score(true_labels, predictions)
    elif (metric_to_use.lower() == 'f1score'):
        metric_all = f1_score(true_labels, predictions, average="micro") 
    elif (metric_to_use.lower() == 'auc'):
        if (predictions_probs.shape[1] == 2): # Binary classification
            metric_all = roc_auc_score(true_labels, predictions_probs[:, 1])
        else:
            metric_all = roc_auc_score(true_labels, predictions_probs, multi_class='ovo', average="macro", labels=all_classes)
    else:
        raise NotImplementedError()

    
    # Plot metric
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    color = 'tab:blue'
    ax1.set_xlabel("Uncertainty threshold")
    ax1.set_ylabel(f"{metric_to_use.upper()}", color=color)
    ax1.plot(thresholds, metric_values, marker='o', color=color, label=metric_to_use.upper())
    ax1.axhline(metric_all, color='gray', linestyle=':', label=f'{metric_to_use.upper()} (all data) = {metric_all:.3f}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Plot number of predictions kept on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("N° predictions kept", color=color)
    ax2.plot(thresholds, num_kept, marker='x', linestyle='--', color=color, label='Num kept')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"{metric_to_use.upper()} and N° Predictions Kept vs Uncertainty Threshold")
    plt.show()


def plot_calibration_uncertainty_correctness(
                                                uncertainty_all_days_per_rep,
                                                targets_all_days_per_rep,
                                                preds_all_days_per_rep,
                                                rep_id,
                                                data_split
                                           ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT  
        
        Plots the uncertainty vs the correctness.

        Parameters:
        -----------
        uncertainty_all_days_per_rep: dict
            Dictionary containing the uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_all_days_per_rep: dict
            Dictionary containing the targets of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_all_days_per_rep: dict
            Dictionary containing the predictions of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Getting the data
    uncertainties = uncertainty_all_days_per_rep[rep_id][data_split]
    true_labels = targets_all_days_per_rep[rep_id][data_split]
    predictions = preds_all_days_per_rep[rep_id][data_split]
    correct = (predictions == true_labels).astype(int)

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(uncertainties, correct, alpha=0.3)
    plt.xlabel("Uncertainty")
    plt.ylabel("Correct prediction")
    plt.title("Uncertainty vs correctness (1=True, 0=False)")
    plt.grid(True)
    plt.show()

    # Bin the uncertainties
    num_bins = 20
    bins = np.linspace(uncertainties.min(), uncertainties.max(), num_bins + 1)
    bin_indices = np.digitize(uncertainties, bins) - 1  # indices from 0 to num_bins-1
    
    # Compute average correctness per bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    prob_correct = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        in_bin = bin_indices == i
        counts[i] = in_bin.sum()
        if counts[i] > 0:
            prob_correct[i] = correct[in_bin].mean()
        else:
            prob_correct[i] = np.nan  # no data in bin
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, prob_correct, marker='o')
    plt.xlabel("Uncertainty")
    plt.ylabel("Observed prob. of correctness")
    plt.title("Calibration: Uncertainty vs Prob. of Correctness")
    plt.grid(True)
    plt.show()
    
    # Optional: show counts
    plt.figure(figsize=(8,3))
    plt.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.9)
    plt.xlabel("Uncertainty")
    plt.ylabel("N° of predictions per bin")
    plt.title("N° of predictions per uncertainty bin")
    plt.show()



def plot_calibrated_uncertainty_correctness(
                                                uncertainty_all_days_per_rep,
                                                targets_all_days_per_rep,
                                                preds_all_days_per_rep,
                                                preds_probs_all_days_per_rep=None,
                                                metric_to_use='BalancedAccuracy',
                                                rep_id=0,
                                                data_split='Test'
                                           ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT  
        
        Plots the CALIBRATED uncertainty vs probability
        of the correctness THRESHOLD.

        Parameters:
        -----------
        uncertainty_all_days_per_rep: dict
            Dictionary containing the uncertainty of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_all_days_per_rep: dict
            Dictionary containing the targets of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_all_days_per_rep: dict
            Dictionary containing the predictions of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_probs_all_days_per_rep: dict
            Dictionary containing the predictions probabilities of ALL the 
            days. The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information). ONLY NECESSARY
            when metric_to_use is AUC.
        metric_to_use: str
            Metric to use for the plot.
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Step 1: Compute correctness (1 if correct, 0 if wrong)
    uncertainties = uncertainty_all_days_per_rep[rep_id][data_split]
    true_labels = targets_all_days_per_rep[rep_id][data_split]
    predictions = preds_all_days_per_rep[rep_id][data_split]
    predictions_probs = preds_probs_all_days_per_rep[rep_id][data_split]
    all_classes = np.unique(true_labels)
    correct = (predictions == true_labels).astype(int)
    
    # Step 2: Fit isotonic regression: uncertainty -> probability of correctness
    # We invert uncertainty because higher uncertainty should correspond to lower correctness
    iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=False)
    iso_reg.fit(uncertainties, correct)
    
    # Step 3: Get calibrated probabilities
    calibrated_probs = iso_reg.predict(uncertainties)
    
    # Optional: Plot calibration curve
    plt.figure(figsize=(8,5))
    plt.scatter(uncertainties, correct, alpha=0.2, label='Original data')
    plt.plot(np.sort(uncertainties), iso_reg.predict(np.sort(uncertainties)),
             color='red', label='Isotonic calibration')
    plt.xlabel("Uncertainty")
    plt.ylabel("Observed probability of correctness")
    plt.title("Calibration: Uncertainty vs Correctness")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Step 4: Filter predictions based on calibrated probability threshold to maximize metric
    thresholds = np.linspace(0, 1, 50)
    metric_values = []
    num_kept = []
    
    for t in thresholds:
        mask = calibrated_probs >= t  # keep predictions with high probability of being correct
        num_kept.append(np.sum(mask))
        
        if np.sum(mask) == 0:
            metric_values.append(np.nan)
            continue
        
        filtered_true = true_labels[mask]
        filtered_pred = predictions[mask]
        filtered_pred_probs = predictions_probs[mask]

        if (metric_to_use.lower() == 'mcc'):
            metric = matthews_corrcoef(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'balancedaccuracy'):
            metric = balanced_accuracy_score(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'f1score'):
            metric = f1_score(filtered_true, filtered_pred, average="micro") 
        elif (metric_to_use.lower() == 'auc'):
            if (filtered_pred_probs.shape[1] == 2): # Binary classification
                metric = roc_auc_score(filtered_true, filtered_pred_probs[:, 1]) 
            else:
                metric = roc_auc_score(filtered_true, filtered_pred_probs, multi_class='ovo', average="macro", labels=all_classes)
        else:
            raise NotImplementedError()
        metric_values.append(metric)
    
    # Step 5: Plot metric vs number of predictions kept
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # Step 6: Compute metric using all data (baseline)
    if (metric_to_use.lower() == 'mcc'):
        metric_all = matthews_corrcoef(true_labels, predictions)
    elif (metric_to_use.lower() == 'balancedaccuracy'):
        metric_all = balanced_accuracy_score(true_labels, predictions)
    elif (metric_to_use.lower() == 'f1score'):
        metric_all = f1_score(true_labels, predictions, average="micro") 
    elif (metric_to_use.lower() == 'auc'):
        if (predictions_probs.shape[1] == 2): # Binary classification
            metric_all = roc_auc_score(true_labels, predictions_probs[:, 1])
        else:
            metric_all = roc_auc_score(true_labels, predictions_probs, multi_class='ovo', average="macro", labels=all_classes)
    else:
        raise NotImplementedError()
    
    color = 'tab:blue'
    ax1.set_xlabel("Porb. of correctness threshold")
    ax1.set_ylabel(f"{metric_to_use.upper()}", color=color)
    ax1.plot(thresholds, metric_values, marker='o', color=color, label=metric_to_use.upper())
    ax1.axhline(metric_all, color='gray', linestyle=':', label=f'{metric_to_use.upper()} (all data) = {metric_all:.3f}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Optional: plot number of predictions kept on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(thresholds, num_kept, marker='x', linestyle='--', color=color, label='N° kept')
    ax2.set_ylabel("N° of preds kept", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"{metric_to_use.upper()} and N° of Preds Kept vs Calibrated Prob. Threshold")
    plt.show()


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
    results_h5_file = h5py.File(results_file, 'r')

    # Number of unique classes (necessary for performances of random classifier)
    if ('murcia' in results_folder.lower()):
        N_UNIQUE_CLASSES = [tmp_i for tmp_i in range(6)]
    else:
        N_UNIQUE_CLASSES = [tmp_i for tmp_i in range(4)]

    # Opening the parameters used for this experiment
    try:
        parameters_file = results_folder + f"/params_exp/params_0.pth"
        with open(parameters_file, 'rb') as pf:
            params_exp = pickle.load(pf)
    except:
        parameters_file = results_folder + f"/params_exp/params_0.yaml"
        with open(parameters_file, 'r') as file:
            params_exp = yaml.safe_load(file)


    # Mapping
    if (params_exp['dataset_name'].lower() == 'sociopatterns'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    elif (params_exp['dataset_name'].lower() == 'murcia'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}

    #======================================================================#
    #===========================Use of multiple DS===========================#
    #======================================================================#
    # Getting the metrics dicts for the different losses
    if ('Rep-0' in results_h5_file):
        base_name_main_group = 'Rep-'
        multiple_datasets = False
    else:
        base_name_main_group = 'Rep-0_Dataset-'
        multiple_datasets = True


    #======================================================================#
    #========================Get performance metrics========================#
    #======================================================================#
    # Getting the predictions for each epoch (concatenate days)
    n_repetitions = len(results_h5_file)
    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[f"{base_name_main_group}0"]["Preds"]["Train"].keys())])
    targets_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_probs_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"].keys())} for rep_id in range(n_repetitions)}
    for rep_id in range(n_repetitions):
        for data_split in list(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"].keys()):
            if (len(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split].keys()) > 0):
                for epoch in epochs_list:
                    targets_all_days = []
                    preds_all_days = []
                    preds_all_days_probs = []
                    n_days = len(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch}"].keys())
                    for day in range(n_days):
                        # Getting targets and predictions
                        targets_all_days.append(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['true_seir_states'])
                        preds_all_days.append(np.argmax(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'], axis=1))
                        preds_all_days_probs.append(results_h5_file[f"{base_name_main_group}{rep_id}"]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'])
                    targets_all_days = np.concatenate(targets_all_days, axis=0)
                    preds_all_days = np.concatenate(preds_all_days, axis=0)
                    preds_all_days_probs = np.concatenate(preds_all_days_probs, axis=0)

                    # Last epochs targets and preds
                    if (epoch == max(epochs_list)):
                        targets_all_days_last_epoch[rep_id][data_split] = targets_all_days
                        preds_all_days_last_epoch[rep_id][data_split] = preds_all_days
                        preds_probs_all_days_last_epoch[rep_id][data_split] = preds_all_days_probs
                    


    #======================================================================#
    #======================Uncertainty quantification======================#
    #======================================================================#
    if (params_exp['loss_function'].lower() == 'evidentiallearningloss'):
        #======================================================================#
        #==================Uncertainty PLOTS (NO CALIBRATION)==================#
        #======================================================================#
        # Getting the average epistemic, aleatoric and total uncertainty for the different repetitions
        #data_split_to_use = 'Train' 
        #data_split_to_use = 'Val' 
        data_split_to_use = 'Test'
        epoch_to_use = max(epochs_list)
        uncertainties_dict = get_evidential_uncertainties(
                                                            h5_results_file=results_h5_file,
                                                            epoch_to_use=epoch_to_use,
                                                            data_split_to_use=data_split_to_use,
                                                            base_name_main_group=base_name_main_group
                                                        )

        # Analyzing some uncertainties
        # Min and max possible values
        n_classes = len(N_UNIQUE_CLASSES)
        #print("\n=========> EPISTEMIC uncertainty ranges between 0 and 1")
        #print(f"\t=========> ALEATORIC uncertainty ranges between 0 and {np.log(n_classes)}")
        #print(f"\t=========> TOTAL uncertainty ranges between 0 and {np.log(n_classes)}")


        # For a given day
        #day_to_use = 0
        day_to_use = -1
        # Get uncertainties of ONLY the predicted class
        tmp_n_indiv = uncertainties_dict['Preds'][day_to_use].shape[0]
        tmp_n_reps = uncertainties_dict['Preds'][day_to_use].shape[1]
        epis_uncert_pred_class = uncertainties_dict['Epistemic'][day_to_use][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day_to_use]]
        alea_uncert_pred_class = uncertainties_dict['Aleatoric'][day_to_use][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day_to_use]]
        tot_uncert_pred_class = uncertainties_dict['Total'][day_to_use][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day_to_use]]
        # Epistemic
        print(f"\n===>MEAN EPISTEMIC uncertainties for day {day_to_use}: {np.mean(epis_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        #print(f"\t===>STD EPISTEMIC uncertainties for day {day_to_use}: {np.std(epis_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        # Aleatoric
        print(f"\n===>MEAN ALEATORIC uncertainties for day {day_to_use}: {np.mean(alea_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        #print(f"\t===>STD ALEATORIC uncertainties for day {day_to_use}: {np.std(alea_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        # Total
        print(f"\n===>MEAN TOTAL uncertainties for day {day_to_use}: {np.mean(tot_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        #print(f"\t===>STD TOTAL uncertainties for day {day_to_use}: {np.std(tot_uncert_pred_class, axis=1)}") # Statistic over all samples and repetitions
        print("\n\n")
    

        # Comparing EPISTEMIC uncertainty for different days
        days_compare = [0, 5, -1]
        plt.figure()
        for day in days_compare:
            # Selecting the uncertainties of the PREDICTED CLASS
            tmp_n_indiv = uncertainties_dict['Preds'][day].shape[0]
            tmp_n_reps = uncertainties_dict['Preds'][day].shape[1]
            epis_uncert_pred_class = uncertainties_dict['Epistemic'][day][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day]]

            # Plot
            indiv_list = list(range(len(uncertainties_dict["Epistemic"][day])))
            epistemic_uncert_mean = np.mean(epis_uncert_pred_class, axis=1)
            epistemic_uncert_std = np.std(epis_uncert_pred_class, axis=1)
            plt.scatter(indiv_list, epistemic_uncert_mean, label=f"Day-{day}")
            #plt.plot(indiv_list, epistemic_uncert_mean, label=f"Day-{day}")
            #plt.errorbar(indiv_list, epistemic_uncert_mean, epistemic_uncert_std, label=f"Day-{day}")
        plt.legend()
        plt.xlabel("Individuals")
        plt.ylabel("Uncertainty")
        plt.title("EPISTEMIC Uncertainty")
        plt.show()

        # Comparing ALEATORIC uncertainty for different days
        days_compare = [0, 5, -1]
        plt.figure()
        for day in days_compare:
            # Selecting the uncertainties of the PREDICTED CLASS
            tmp_n_indiv = uncertainties_dict['Preds'][day].shape[0]
            tmp_n_reps = uncertainties_dict['Preds'][day].shape[1]
            alea_uncert_pred_class = uncertainties_dict['Aleatoric'][day][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day]]

            # Plot
            indiv_list = list(range(len(uncertainties_dict["Aleatoric"][day])))
            aleatoric_uncert_mean = np.mean(alea_uncert_pred_class, axis=1)
            aleatoric_uncert_std = np.std(alea_uncert_pred_class, axis=1)
            plt.scatter(indiv_list, aleatoric_uncert_mean, label=f"Day-{day}")
            #plt.plot(indiv_list, aleatoric_uncert_mean, label=f"Day-{day}")
            #plt.errorbar(indiv_list, aleatoric_uncert_mean, aleatoric_uncert_std, label=f"Day-{day}")
        plt.legend()
        plt.xlabel("Individuals")
        plt.ylabel("Uncertainty")
        plt.title("ALEATORIC Uncertainty")
        plt.show()

        # Comparing TOTAL uncertainty for different days
        days_compare = [0, 5, -1]
        plt.figure()
        for day in days_compare:
            # Selecting the uncertainties of the PREDICTED CLASS
            tmp_n_indiv = uncertainties_dict['Preds'][day].shape[0]
            tmp_n_reps = uncertainties_dict['Preds'][day].shape[1]
            tot_uncert_pred_class = uncertainties_dict['Total'][day][np.arange(tmp_n_indiv)[:, None], np.arange(tmp_n_reps), uncertainties_dict['Preds'][day]]

            # Plot
            indiv_list = list(range(len(uncertainties_dict["Total"][day])))
            total_uncert_mean = np.mean(tot_uncert_pred_class, axis=1)
            total_uncert_std = np.std(tot_uncert_pred_class, axis=1)
            plt.scatter(indiv_list, total_uncert_mean, label=f"Day-{day}")
            #plt.plot(indiv_list, total_uncert_mean, label=f"Day-{day}")
            #plt.errorbar(indiv_list, total_uncert_mean, total_uncert_std, label=f"Day-{day}")
        plt.legend()
        plt.xlabel("Individuals")
        plt.ylabel("Uncertainty")
        plt.title("TOTAL Uncertainty")
        plt.show()

        # Get the average uncertainties per class
        # Day to use
        #day_to_use = 0
        day_to_use = -1
        #day_to_use = 10
        # Get per class uncertainty
        # Epistemic
        per_class_epis_uncert_mean = np.mean(uncertainties_dict['Epistemic'][day_to_use], axis=(0, 1))
        per_class_epis_uncert_std = np.std(uncertainties_dict['Epistemic'][day_to_use], axis=(0, 1))
        # Aleatoric
        per_class_alea_uncert_mean = np.mean(uncertainties_dict['Aleatoric'][day_to_use], axis=(0, 1))
        per_class_alea_uncert_std = np.std(uncertainties_dict['Aleatoric'][day_to_use], axis=(0, 1))
        # Total
        per_class_total_uncert_mean = np.mean(uncertainties_dict['Total'][day_to_use], axis=(0, 1))
        per_class_total_uncert_std = np.std(uncertainties_dict['Total'][day_to_use], axis=(0, 1))
        for target_label in range(per_class_epis_uncert_mean.shape[0]):
            print(f"\n\n=========> CLASS {target_label} <=========\n")
            # Epistemic
            print(f"\n===>MEAN EPISTEMIC uncertainties for day {day_to_use}: {per_class_epis_uncert_mean[target_label]}") # Statistic over all samples and repetitions
            #print(f"\t===>STD EPISTEMIC uncertainties for day {day_to_use}: {per_class_epis_uncert_std[target_label]}") # Statistic over all samples and repetitions
            # Aleatoric
            print(f"\n===>MEAN ALEATORIC uncertainties for day {day_to_use}: {per_class_alea_uncert_mean[target_label]}") # Statistic over all samples and repetitions
            #print(f"\t===>STD ALEATORIC uncertainties for day {day_to_use}: {per_class_alea_uncert_std[target_label]}") # Statistic over all samples and repetitions
            # Total
            print(f"\n===>MEAN TOTAL uncertainties for day {day_to_use}: {per_class_total_uncert_mean[target_label]}") # Statistic over all samples and repetitions
            #print(f"\t===>STD TOTAL uncertainties for day {day_to_use}: {per_class_total_uncert_std[target_label]}") # Statistic over all samples and repetitions
        

        # Uncertainty evolution of a patient over time
        #node_to_use = 0
        #n_nodes = len(uncertainties_dict["Epistemic"][0])
        n_nodes = 3
        #n_nodes = 10
        for node_to_use in range(n_nodes):
            n_days = len(uncertainties_dict["Epistemic"])
            uncertainties_per_day_current_node = {
                                                    "Epistemic": [-1 for _ in range(n_days)],
                                                    "Aleatoric": [-1 for _ in range(n_days)],
                                                    "Total": [-1 for _ in range(n_days)]
                                                }
            for day in range(n_days):
                tmp_n_reps = uncertainties_dict['Preds'][day].shape[1]
                pred_class = uncertainties_dict["Preds"][day][node_to_use][:]
                uncertainties_per_day_current_node["Epistemic"][day] = uncertainties_dict["Epistemic"][day][node_to_use][np.arange(tmp_n_reps), pred_class]
                uncertainties_per_day_current_node["Aleatoric"][day] = uncertainties_dict["Aleatoric"][day][node_to_use][np.arange(tmp_n_reps), pred_class]
                uncertainties_per_day_current_node["Total"][day] = uncertainties_dict["Total"][day][node_to_use][np.arange(tmp_n_reps), pred_class]

            plt.figure()
            for uncert_type in uncertainties_per_day_current_node:
                mean_uncert = [np.mean(uncertainties_per_day_current_node[uncert_type][day]) for day in range(n_days)]
                std_uncert = [np.std(uncertainties_per_day_current_node[uncert_type][day]) for day in range(n_days)]
                #plt.scatter(list(range(n_days)), mean_uncert, label=uncert_type)
                #plt.plot(list(range(n_days)), mean_uncert, label=uncert_type)
                plt.errorbar(list(range(n_days)), mean_uncert, std_uncert, label=uncert_type)
            plt.legend()
            plt.xlabel("Time (days)")
            plt.ylabel("Uncertainty")
            plt.title(f"Uncertainties for node {node_to_use}")
            plt.show()


        #======================================================================#
        #==================Uncertainty PLOTS (CALIBRATION)==================#
        #======================================================================#
        # Reordering the uncertainties to have one per sample as for targets and predictions of all the days in the last epoch
        #data_split = "Train"
        #data_split = "Val"
        data_split = "Test"
        epoch_to_use = max([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"][data_split].keys())])
        total_uncertainties_all_days_epoch_to_use,\
        alea_uncertainty_all_days_epoch_to_use,\
        epi_uncertainty_all_days_epoch_to_use = get_uncertainties_per_rep_all_days(
                                                                                    results_h5_file=results_h5_file,
                                                                                    epoch_to_use=epoch_to_use
                                                                                )
        
        # Parameters for ALL THE PLOTS
        rep_id = 0
        #metric_to_use = "MCC"
        #metric_to_use = "F1Score"
        metric_to_use = "BalancedAccuracy"
        #metric_to_use = "AUC"
        #uncertainty_all_days_per_rep = total_uncertainties_all_days_epoch_to_use
        uncertainty_all_days_per_rep = alea_uncertainty_all_days_epoch_to_use
        #uncertainty_all_days_per_rep = epi_uncertainty_all_days_epoch_to_use
        targets_all_days_per_rep = targets_all_days_last_epoch
        preds_all_days_per_rep = preds_all_days_last_epoch
        preds_probs_all_days_per_rep = preds_probs_all_days_last_epoch

        # Plot metric vs uncertainty
        plot_metric_vs_uncertainty(
                                    uncertainty_all_days_per_rep=uncertainty_all_days_per_rep,
                                    targets_all_days_per_rep=targets_all_days_per_rep,
                                    preds_all_days_per_rep=preds_all_days_per_rep,
                                    preds_probs_all_days_per_rep=preds_probs_all_days_per_rep,
                                    metric_to_use=metric_to_use,
                                    rep_id=rep_id,
                                    data_split=data_split
                                )
        
        # Uncertainty vs Correctness
        plot_calibration_uncertainty_correctness(
                                                    uncertainty_all_days_per_rep,
                                                    targets_all_days_per_rep,
                                                    preds_all_days_per_rep,
                                                    rep_id,
                                                    data_split
                                            )
        
        # Plot calibrated uncertainty using the probability of correctness threshold
        plot_calibrated_uncertainty_correctness(
                                                uncertainty_all_days_per_rep=uncertainty_all_days_per_rep,
                                                targets_all_days_per_rep=targets_all_days_per_rep,
                                                preds_all_days_per_rep=preds_all_days_per_rep,
                                                preds_probs_all_days_per_rep=preds_probs_all_days_per_rep,
                                                metric_to_use=metric_to_use,
                                                rep_id=rep_id,
                                                data_split=data_split
                                            )

    else:
        print("\nUncertainty quantification can only be done for evidential learning experiments")


if (__name__=='__main__'):
    main()