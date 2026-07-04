"""
    Plot the metrics and results of an experiment
"""
import h5py
import yaml
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,\
                            f1_score,\
                            balanced_accuracy_score,\
                            classification_report,\
                            roc_auc_score,\
                            average_precision_score
from imblearn.metrics import sensitivity_score, specificity_score
import torch

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Global variables
PLOT_FIGURES = True
#PLOT_FIGURES = False

def plot_metric_epochs(metric_dict, metric_name="Loss"):
    """
        Plots a metric of the results of a model over the epochs

        Parameters:
        -----------
        metric_dict: dict
            Dictionary where the keys are the data splits and the values are
            the lists with the values of the different losses over the epochs
            and repetitions for that data split.
            The structure is the following:
            data_split -> epochs -> repetitions
    """
    # Creating the figure
    fig = plt.figure()
    # Iterating over the different data splits
    for data_split in metric_dict:
        # Getting the number of epochs
        if (type(metric_dict[data_split]) == dict):
            epochs = sorted(list(metric_dict[data_split].keys()))
        else:
            epochs = list(range(len(metric_dict[data_split])))

        # Getting the mean and std value per epoch
        mean_per_epoch = []
        std_per_epoch = []
        for values_per_rep in metric_dict[data_split]:
            if (type(metric_dict[data_split]) == dict):
                if (metric_dict[data_split][values_per_rep][0] is not None):
                    mean_per_epoch.append(np.mean(metric_dict[data_split][values_per_rep]))
                    std_per_epoch.append(np.std(metric_dict[data_split][values_per_rep])) 
            else:
                mean_per_epoch.append(np.mean(values_per_rep))
                std_per_epoch.append(np.std(values_per_rep))
        if (len(mean_per_epoch) > 0):
            plt.errorbar(epochs, mean_per_epoch, std_per_epoch, label=f"{data_split} {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    if (PLOT_FIGURES):
        plt.show()

def plot_pred_epidemic_evolution(h5_results_file, epoch_to_use, states_mapping, data_split='Test'):
    # Getting the inverse state mapping
    inv_states_mapping = {v: k for k, v in states_mapping.items()}
    
    # Counting the number of individuals per state in each day
    n_repetitions = len(h5_results_file)
    try:
        n_days = len(h5_results_file["Rep-0"]["Preds"][data_split][f"Epoch-{epoch_to_use}"])
    except:
        n_days = len(h5_results_file["Rep-0_Dataset-0"]["Preds"][data_split][f"Epoch-{epoch_to_use}"])

    n_ind_per_state_per_day_true_mean = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_true_std = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_pred_mean = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_pred_std = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}

    # Getting the mean and std values per day
    for day in range(n_days):
        vals_in_day_true = {tmp_state: [0 for _ in range(n_repetitions)] for tmp_state in states_mapping}
        vals_in_day_pred = {tmp_state: [0 for _ in range(n_repetitions)] for tmp_state in states_mapping}
        for rep_id in range(n_repetitions):
            # True values
            try:
                true_states = h5_results_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['true_seir_states'][:]
            except:
                true_states = h5_results_file[f"Rep-0_Dataset-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['true_seir_states'][:]
            true_values, true_counts = np.unique(true_states, return_counts=True)
            true_value_counts = {v.item(): c.item() for v, c in zip(true_values, true_counts)}
            for idx_state in true_value_counts:
                vals_in_day_true[inv_states_mapping[idx_state]][rep_id] = true_value_counts[idx_state]
        
            # Preds
            try:
                pred_states = np.argmax(h5_results_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'][:, :], axis=1)
            except:
                pred_states = np.argmax(h5_results_file[f"Rep-0_Dataset-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'][:, :], axis=1)
            pred_values, pred_counts = np.unique(pred_states, return_counts=True)
            pred_value_counts = {v.item(): c.item() for v, c in zip(pred_values, pred_counts)}
            for idx_state in pred_value_counts:
                vals_in_day_pred[inv_states_mapping[idx_state]][rep_id] = pred_value_counts[idx_state]
        # Get mean and std for the day
        for state in vals_in_day_true:
            n_ind_per_state_per_day_true_mean[state][day] = np.mean(vals_in_day_true[state])
            n_ind_per_state_per_day_true_std[state][day] = np.std(vals_in_day_true[state])
            n_ind_per_state_per_day_pred_mean[state][day] = np.mean(vals_in_day_pred[state])
            n_ind_per_state_per_day_pred_std[state][day] = np.std(vals_in_day_pred[state])

    # Plot one per figure
    fig = plt.figure()
    colors = {"S": "b", "E": "orange", "I": "green", "R": "r", "D": "purple", "NS": "brown"}
    days_list = list(range(n_days))
    for state in n_ind_per_state_per_day_true_mean:
        plt.plot(days_list, n_ind_per_state_per_day_true_mean[state], c=colors[state], linestyle='-', label=f"True {state}")
        plt.plot(days_list, n_ind_per_state_per_day_pred_mean[state], c=colors[state], linestyle='--', label=f"Predicted {state}")
        plt.legend()
        if (PLOT_FIGURES):
            plt.show()

    # Plot all in the same figure
    fig = plt.figure()
    colors = {"S": "b", "E": "orange", "I": "green", "R": "r", "D": "purple", "NS": "brown"}
    days_list = list(range(n_days))
    for state in n_ind_per_state_per_day_true_mean:
        plt.plot(days_list, n_ind_per_state_per_day_true_mean[state], c=colors[state], linestyle='-', label=f"True {state}")
        plt.plot(days_list, n_ind_per_state_per_day_pred_mean[state], c=colors[state], linestyle='--', label=f"Predicted {state}")
    plt.legend()
    if (PLOT_FIGURES):
        plt.show()

def get_classification_metrics(targets, preds, data_split='Test', n_unique_classes=None, verbose=False, print_classification_report=False):
    # Reshaping targets and predictions in the case that we have a forecast horizon larger than 1 (in that case, 
    # the shape of targets and preds is (num_patients, forecast_horizon) instead of (num_patients))
    if (len(targets.shape) > 1):
        targets = targets.reshape(-1) 
        preds = preds.reshape(-1) 

    # Getting predictions for random classifier
    if (n_unique_classes is None):
        n_unique_classes = np.unique(targets)
    random_pred = torch.randint(low=0, high=max(n_unique_classes), size=( len(preds), 1)).detach().cpu().numpy()

    # Sensitivity
    if (len(n_unique_classes) == 2):
        sensitivity = sensitivity_score(targets, preds)
        sensitivity_random = sensitivity_score(targets, random_pred)
    else:
        sensitivity = sensitivity_score(targets, preds, average='weighted')
        sensitivity_random = sensitivity_score(targets, random_pred, average='weighted')
    if (verbose):
        print(f"\n{data_split} sensitivity: {sensitivity*100}%")
        print(f"\t{data_split} sensitivity random classifier: {sensitivity_random*100}%")
    # Specificty
    if (len(n_unique_classes) == 2):
        specificity = specificity_score(targets, preds)
        specificity_random = specificity_score(targets, random_pred)
    else:
        specificity = specificity_score(targets, preds, average='weighted')
        specificity_random = specificity_score(targets, random_pred, average='weighted')
    if (verbose):
        print(f"\n{data_split} specificity accuracy: {specificity*100}%")
        print(f"\t{data_split} specificity random classifier: {specificity_random*100}%")
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(targets, preds, adjusted=True)
    balanced_acc_random = balanced_accuracy_score(targets, random_pred, adjusted=True)
    if (verbose):
        print(f"\n{data_split} balanced accuracy: {balanced_acc*100}%")
        print(f"\t{data_split} balanced accuracy random classifier: {balanced_acc_random*100}%")
    # MCC
    mcc = matthews_corrcoef(targets, preds)
    mcc_random = matthews_corrcoef(targets, random_pred)
    if (verbose):
        print(f"\n{data_split} MCC: {mcc*100}%")
        print(f"\t{data_split} MCC random classifier: {mcc_random*100}%")
    # F1 Score
    if (len(n_unique_classes) == 2):
        f1_score_val = f1_score(targets, preds, average="binary")
        f1_score_val_random = f1_score(targets, random_pred, average="binary")
    else:
        f1_score_val = f1_score(targets, preds, average="micro")
        f1_score_val_random = f1_score(targets, random_pred, average="micro")
    if (verbose):
        print(f"\n{data_split} F1 Score: {f1_score_val*100}%")
        print(f"\t{data_split} F1 Score random classifier: {f1_score_val_random*100}%")
    
    # Performance per class
    if (print_classification_report):
        if (len(n_unique_classes) == 4):
            target_names = ["S", "E", "I", "R"]
        if (len(n_unique_classes) == 3):
            target_names = ["S", "I", "R"]
        elif (len(n_unique_classes) == 6):
            target_names = ["S", "E", "I", "R", "D", "NS"]
        print("\n\n{} classification report: {}".format(data_split, classification_report(targets, preds, target_names=target_names, labels=n_unique_classes) ))
        print("\n\t{} classification report random classifier: {}".format(data_split, classification_report(targets, random_pred, target_names=target_names, labels=n_unique_classes)))

    return sensitivity*100, specificity*100, mcc*100, f1_score_val*100, balanced_acc*100


def get_classification_metrics_per_days(results_h5_file, n_unique_classes, data_split_to_use='Test', is_hug_dataset=False, folder_save_results=None):
    # Base group name
    if ('Rep-0' in list(results_h5_file.keys())):
        base_group_name = 'Rep'
    else:
        base_group_name = 'Rep-0_Dataset'

    # Getting basic information to plot the metrics
    n_repetitions = len(results_h5_file)
    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[f"{base_group_name}-0"]["Preds"]["Train"].keys())])


    # Evaluation metrics PER DAYS
    last_epoch = max(epochs_list)
    metrics_per_days = {data_split: {} for data_split in list(results_h5_file[f"{base_group_name}-0"]["Preds"].keys())}
    n_samples_per_class_per_days = {data_split: {} for data_split in list(results_h5_file[f"{base_group_name}-0"]["Preds"].keys())}
    for rep_id in tqdm(range(n_repetitions)):
        # print(f"\n\n=========> Handling repetition {rep_id} <=========")
        for data_split in list(results_h5_file[f"{base_group_name}-{rep_id}"]["Preds"].keys()):
            # IMPORTANT: THE NUMBER OF DAYS DEPENDS ON THE DATASET SPLIT
            n_days = len(results_h5_file[f"{base_group_name}-{rep_id}"]["Preds"][data_split][f"Epoch-{last_epoch}"].keys())
            for day in range(n_days):
                # Metrics dict for the current day 
                if (day not in metrics_per_days[data_split]):
                    metrics_per_days[data_split][day] = {
                                                            "Sensitivity": [None for _ in range(n_repetitions)],
                                                            "Specificity": [None for _ in range(n_repetitions)],
                                                            "MCC": [None for _ in range(n_repetitions)],
                                                            "F1Score": [None for _ in range(n_repetitions)],
                                                            "BalancedAccuracy": [None for _ in range(n_repetitions)],
                                                            "AUC": [None for _ in range(n_repetitions)],
                                                            "PerClassAUC": [None for _ in range(n_repetitions)]
                                                        }

                # Number of samples per class dict for the current day 
                # IMPORTANT: BETWEEN REPETITIONS IS NOT THE SAME AS THE DATASET CHANGES !!!
                if (day not in n_samples_per_class_per_days[data_split]):
                    n_samples_per_class_per_days[data_split][day] = {tmp_class: [0 for _ in range(n_repetitions)] for tmp_class in n_unique_classes}

                # Getting targets and predictions
                targets_current_day = results_h5_file[f"{base_group_name}-{rep_id}"]["Preds"][data_split][f"Epoch-{last_epoch}"][f"Day-{day}"]['true_seir_states'][:]
                preds_current_day = np.argmax(results_h5_file[f"{base_group_name}-{rep_id}"]["Preds"][data_split][f"Epoch-{last_epoch}"][f"Day-{day}"]['pred_seir_states_probs'][:], axis=1)
                preds_probs_current_day = results_h5_file[f"{base_group_name}-{rep_id}"]["Preds"][data_split][f"Epoch-{last_epoch}"][f"Day-{day}"]['pred_seir_states_probs'][:]

                # Number of samples per class
                unique_targets, counts = np.unique(targets_current_day, return_counts=True)
                counts_per_class = dict(zip(unique_targets, counts))
                for tmp_class in counts_per_class:
                    if (tmp_class not in n_samples_per_class_per_days[data_split][day]):
                        n_samples_per_class_per_days[data_split][day][tmp_class] = [0 for _ in range(n_repetitions)]
                    n_samples_per_class_per_days[data_split][day][tmp_class][rep_id] = counts_per_class[tmp_class]


                # For HUG COVID dataset, we only keep the valid labels
                if (is_hug_dataset):
                    mask_valid_labels = targets_current_day != -1
                    if (len(mask_valid_labels.shape) == 2) and (mask_valid_labels.shape[1] == 1):
                        mask_valid_labels = mask_valid_labels[:, 0]
                        targets_current_day = targets_current_day[mask_valid_labels].squeeze()
                    else:
                        targets_current_day = targets_current_day[mask_valid_labels]
                    preds_probs_current_day = preds_probs_current_day[mask_valid_labels]
                    preds_current_day = preds_current_day[mask_valid_labels]
                

                try:
                    sensitivity, specificity, mcc, f1_score_val, balanced_acc = get_classification_metrics(
                                                                                                            targets_current_day,
                                                                                                            preds_current_day,
                                                                                                            data_split,
                                                                                                            n_unique_classes,
                                                                                                            #verbose=True,
                                                                                                            verbose=False,
                                                                                                            #print_classification_report=True
                                                                                                            print_classification_report=False
                                                                                                        )
                except:
                    sensitivity, specificity, mcc, f1_score_val, balanced_acc = np.nan, np.nan, np.nan, np.nan, np.nan 

                # AUC
                # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                try:
                    auc = roc_auc_score(targets_current_day, preds_probs_current_day, multi_class='ovo', average="macro", labels=n_unique_classes)
                    random_preds_probs = np.random.dirichlet(alpha=np.ones(max(n_unique_classes)+1), size=len(targets_current_day))
                    auc_random = roc_auc_score(targets_current_day, random_preds_probs, multi_class='ovo', average="macro", labels=n_unique_classes)
                    # Per-class PR AUCs
                    per_class_pr_auc = []
                    per_class_pr_auc_random = []
                    for c in n_unique_classes:
                        y_true_c = (targets_current_day == c).astype(int)
                        y_score_c = preds_probs_current_day[:, c]
                        y_score_random_c = random_preds_probs[:, c]
                        per_class_pr_auc.append(average_precision_score(y_true_c, y_score_c))
                        per_class_pr_auc_random.append(average_precision_score(y_true_c, y_score_random_c))
                except:
                    auc = np.nan
                    auc_random = np.nan
                    # Per-class PR AUCs
                    per_class_pr_auc = []
                    per_class_pr_auc_random = []
                    for c in n_unique_classes:
                        per_class_pr_auc.append(np.nan)
                        per_class_pr_auc_random.append(np.nan)
                
                metrics_per_days[data_split][day]["Sensitivity"][rep_id] = sensitivity 
                metrics_per_days[data_split][day]["Specificity"][rep_id] = specificity 
                metrics_per_days[data_split][day]["MCC"][rep_id] = mcc 
                metrics_per_days[data_split][day]["F1Score"][rep_id] = f1_score_val 
                metrics_per_days[data_split][day]["BalancedAccuracy"][rep_id] = balanced_acc 
                metrics_per_days[data_split][day]["AUC"][rep_id] = auc 
                metrics_per_days[data_split][day]["PerClassAUC"][rep_id] = per_class_pr_auc 

    # Plot metrics per days
    sorted_list_days = sorted(list(metrics_per_days[data_split_to_use].keys()))
    list_metrics = list(metrics_per_days[data_split_to_use][0].keys())
    mean_metrics_per_day = {
                                metric: {
                                            'Mean': [None for tmp_day in sorted_list_days],
                                            'Std': [None for tmp_day in sorted_list_days]
                                        } for metric in list_metrics if (metric != 'PerClassAUC')
                        }
    for tmp_class in n_unique_classes:
        mean_metrics_per_day[f'PerClassAUC_{tmp_class}'] = {
                                                                'Mean': [None for tmp_day in sorted_list_days],
                                                                'Std': [None for tmp_day in sorted_list_days]
                                                            }
    mean_n_samples_per_class_per_day = {
                                            target_class: {
                                                                'Mean': [None for tmp_day in sorted_list_days],
                                                                'Std': [None for tmp_day in sorted_list_days]
                                                            } for target_class in n_unique_classes
                                    }
    for day in tqdm(sorted_list_days):
        # Metrics
        for metric in list_metrics:
            # Compute metrics only if there are a minimum of samples for AT LEAST TWO CLASSES (i.e. all the samples are not in a single class)
            compute_mean_metrics = True
            for rep_id in range(len(n_samples_per_class_per_days[data_split_to_use][day][tmp_class])):
                n_total_individuals = np.array(list(n_samples_per_class_per_days[data_split_to_use][day].values()))[:, rep_id].sum()
                if (n_samples_per_class_per_days[data_split_to_use][day][tmp_class][rep_id] == n_total_individuals):
                    compute_mean_metrics = False
                    break
            # Compute metrics if possible
            if (compute_mean_metrics):
                if (metric == 'PerClassAUC'):
                    for tmp_class in n_unique_classes:
                        class_auc = [metrics_per_days[data_split_to_use][day]['PerClassAUC'][rep_id][tmp_class] for rep_id in range(n_repetitions)]
                        mean_metrics_per_day[metric+f'_{tmp_class}']['Mean'][day] = np.mean(class_auc)
                        mean_metrics_per_day[metric+f'_{tmp_class}']['Std'][day] = np.std(class_auc)
                else:
                    mean_metrics_per_day[metric]['Mean'][day] = np.mean(metrics_per_days[data_split_to_use][day][metric])
                    mean_metrics_per_day[metric]['Std'][day] = np.std(metrics_per_days[data_split_to_use][day][metric])
            else:
                if (metric == 'PerClassAUC'):
                    for tmp_class in n_unique_classes:
                        mean_metrics_per_day[metric+f'_{tmp_class}']['Mean'][day] = np.nan
                        mean_metrics_per_day[metric+f'_{tmp_class}']['Std'][day] = np.nan
                else:
                    mean_metrics_per_day[metric]['Mean'][day] = np.nan
                    mean_metrics_per_day[metric]['Std'][day] = np.nan

        # Number of samples per day
        # IMPORTANT: BETWEEN REPETITIONS IS NOT THE SAME AS THE DATASET CHANGES !!!
        for tmp_class in n_samples_per_class_per_days[data_split_to_use][day]:
            if (tmp_class != -1):
                mean_n_samples_per_class_per_day[tmp_class]['Mean'][day] = np.mean(n_samples_per_class_per_days[data_split_to_use][day][tmp_class])
                mean_n_samples_per_class_per_day[tmp_class]['Std'][day] = np.std(n_samples_per_class_per_days[data_split_to_use][day][tmp_class])
            
                
    # Plot
    for metric in mean_metrics_per_day:
        # Create the base plot
        fig, ax1 = plt.subplots()
        
        # Plot the metrics
        color = 'tab:purple'
        ax1.set_xlabel('Day')
        ax1.set_ylabel(metric, color=color)
        #ax1.plot(sorted_list_days, mean_metrics_per_day[metric]['Mean'], color=color, label=metric)
        ax1.errorbar(sorted_list_days, mean_metrics_per_day[metric]['Mean'], mean_metrics_per_day[metric]['Std'], color=color, label=metric)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.legend()
        
        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()
        
        # Plot the number of samples per class
        ax2.set_ylabel('Number of samples')
        for tmp_class in mean_n_samples_per_class_per_day:
            #ax2.plot(sorted_list_days, mean_n_samples_per_class_per_day[tmp_class]['Mean'], linestyle='--', label=tmp_class)
            ax2.errorbar(sorted_list_days, mean_n_samples_per_class_per_day[tmp_class]['Mean'], mean_n_samples_per_class_per_day[tmp_class]['Std'], linestyle='--', label=tmp_class)
            ax2.tick_params(axis='y')
        
        # Add a title and improve layout
        plt.title(metric.upper())
        fig.tight_layout()

        plt.legend()
        if (PLOT_FIGURES):
            plt.show()


    # Mean performance over all the days
    # To do this we FIRST compute the metrics for each repetition for 
    # each day, THEN we average for all the days to get the mean metrics
    # over the days FOR EACH REPETITION.
    mean_metrics_per_rep = { metric: [None for rep in range(n_repetitions)] for metric in list_metrics if (metric != 'PerClassAUC') }
    for tmp_class in n_unique_classes:
        mean_metrics_per_rep[f'PerClassAUC_{tmp_class}'] = [None for rep in range(n_repetitions)]
    for rep_id in tqdm(range(n_repetitions)):
        for metric in list_metrics:
            if (metric == 'PerClassAUC'):
                for tmp_class in n_unique_classes:
                    # The following lines DOES NOT take into account the case where all the individuals are in a single state
                    class_auc = []
                    for day in sorted_list_days:
                        # If all the patients are in one state, then we ignore the value as it will give Nan or 0 values
                        n_total_individuals = np.array(list(n_samples_per_class_per_days[data_split_to_use][day].values()))[:, rep_id].sum()
                        ignore_value = False
                        for tmp_state in n_samples_per_class_per_days[data_split_to_use][day]:
                            if (n_samples_per_class_per_days[data_split_to_use][day][tmp_state][rep_id] == n_total_individuals):
                                ignore_value = True
                                break
                        if (not ignore_value):
                            class_auc.append(metrics_per_days[data_split_to_use][day]['PerClassAUC'][rep_id][tmp_class])
                    class_auc = np.array(class_auc)
                    class_auc = class_auc[np.isfinite(class_auc)]
                    mean_metrics_per_rep[metric+f'_{tmp_class}'][rep_id] = np.nanmean(class_auc)
            else:
                # The following lines DOES NOT take into account the case where all the individuals are in a single state
                tmp_metric_vals = []
                for day in sorted_list_days:
                    # If all the patients are in one state, then we ignore the value as it will give Nan or 0 values
                    n_total_individuals = np.array(list(n_samples_per_class_per_days[data_split_to_use][day].values()))[:, rep_id].sum()
                    ignore_value = False
                    for tmp_state in n_samples_per_class_per_days[data_split_to_use][day]:
                        if (n_samples_per_class_per_days[data_split_to_use][day][tmp_state][rep_id] == n_total_individuals):
                            ignore_value = True
                            break
                    if (not ignore_value):
                        tmp_metric_vals.append(metrics_per_days[data_split_to_use][day][metric][rep_id])
                tmp_metric_vals = np.array(tmp_metric_vals)
                mean_metrics_per_rep[metric][rep_id] = np.nanmean(tmp_metric_vals)

    # Print the mean metrics (average first over the days, then over the repetitions)
    lines_to_write = []
    lines_to_write.append("\n\n\n=========> Metrics computed by average FIRST by days, THEN by repetitions")
    for metric in mean_metrics_per_rep:
        tmp_metric_vals = np.array(mean_metrics_per_rep[metric])
        tmp_metric_vals = tmp_metric_vals[np.isfinite(tmp_metric_vals)]
        if ('PerClassAUC' in metric) or ('AUC' in metric):
            lines_to_write.append(f"\n\t{data_split_to_use.upper()} {metric}: {np.nanmean(tmp_metric_vals)*100} \xB1 {np.nanstd(tmp_metric_vals)*100}")
        else:
            lines_to_write.append(f"\n\t{data_split_to_use.upper()} {metric}: {np.nanmean(tmp_metric_vals)} \xB1 {np.nanstd(tmp_metric_vals)}")
    for str_to_print in lines_to_write:
        print(str_to_print)

    if (folder_save_results is not None):
        file_name = folder_save_results + "/metrics_average_per_day.txt"
        try:
            with open(file_name, 'w') as f:
                f.writelines(lines_to_write)
            print(f"Successfully wrote to {file_name}")
        except IOError as e:
            print(f"An error occurred: {e}")
                



def compute_ece(probs, labels, n_bins=10):
    """
        IMPORTANT: Generated using ChatGPT.
        Computes the expected calibratione error (ECE).

        Parameters:
        -----------
        probs: np.array or torch.tensor
            [N, num_classes] softmax probabilities
        probs: np.array or torch.tensor
            [N] true labels
    """
    # Get the "confindence" of the model on the predictions (i.e. max prob per sample, allowing to choose the predicted class)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    # Computing the ECE
    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins+1)
    for i in range(n_bins):
        # Select samples in bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * abs(acc_in_bin - avg_conf_in_bin)
    
    return ece

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
    ap.add_argument('--print_classification_report', help="Use it if want to print the classification report per epochs", action='store_true')
    ap.add_argument('--print_overall_metrics', help="Print the overall metrics concatenating the predictions of all days and then computing the metrics", action='store_true')
    ap.add_argument('--print_average_metrics_over_days', help="Print the average metrics over the days (compute the metrics for each day and then average)", action='store_true')
    ap.add_argument('-v', '--verbose', help="Use it to get details about the performances per epoch", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    print_classification_report = args['print_classification_report']
    print_overall_metrics = args['print_overall_metrics']
    print_average_metrics_over_days = args['print_average_metrics_over_days']
    verbose = args['verbose']
    
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

    #======================================================================#
    #===============================Plot loss===============================#
    #======================================================================#
    # Getting the metrics dicts for the different losses
    if ('Rep-0' in results_h5_file):
        base_name_main_group = 'Rep-'
        multiple_datasets = False
    else:
        base_name_main_group = 'Rep-0_Dataset-'
        multiple_datasets = True
    loss_names = list(results_h5_file[base_name_main_group+"0"]['Loss']['Train'].keys())
    n_repetitions = len(results_h5_file)

    # Creating the loss dict
    losses_dict = {}
    for loss_name in loss_names:
        losses_dict[loss_name] = {}
        for data_split in ["Train", "Val", "Test"]:
            if (data_split in results_h5_file[base_name_main_group+"0"]['Loss']):
                if (len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split].keys()) > 0):
                    n_epochs = len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split][loss_name])
                    losses_dict[loss_name][data_split] = [[None for _ in range(n_repetitions)] for _ in range(n_epochs)]
    # Filling the loss dict
    for loss_name in loss_names:
        for rep_id in range(n_repetitions):
            for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]['Loss'].keys()):
                if (len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split].keys()) > 0):
                    n_epochs = len(results_h5_file[base_name_main_group+str(rep_id)]['Loss'][data_split][loss_name])
                    for epoch in range(n_epochs):
                        losses_dict[loss_name][data_split][epoch][rep_id] = results_h5_file[base_name_main_group+str(rep_id)]['Loss'][data_split][loss_name][epoch]

    # Plotting the different losses
    for loss_type in loss_names:
        plot_metric_epochs(metric_dict=losses_dict[loss_type], metric_name=loss_type)

    #======================================================================#
    #========================Get performance metrics========================#
    #======================================================================#
    # Getting the predictions for each epoch (concatenate days)
    if (print_overall_metrics):
        n_repetitions = len(results_h5_file)
        epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"]["Train"].keys())])
        metrics_per_data_split = {
                                    "Sensitivity": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "Specificity": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "MCC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "F1Score": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "BalancedAccuracy": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "AUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                    "PerClassAUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                }
        if (multiple_datasets):
            base_group_name = "Rep"
        targets_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        preds_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        preds_probs_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        targets_inf_risk_all_days_last_epoch = None
        preds_inf_risk_all_days_last_epoch = None
        preds_probs_inf_risk_all_days_last_epoch = None
        if ("true_inf_risk_class" in results_h5_file[base_name_main_group+"0"]["Preds"]["Train"][f"Epoch-{epoch}"][f"Day-0"]):
            targets_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
            preds_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
            preds_probs_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
            metrics_per_data_split["SensitivityInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["SpecificityInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["MCCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["F1ScoreInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["BalancedAccuracyInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["AUCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
            metrics_per_data_split["PerClassAUCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
        for rep_id in range(n_repetitions):
            for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys()):
                if (len(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split].keys()) > 0):
                    for epoch in epochs_list:
                        targets_all_days = []
                        preds_all_days = []
                        preds_all_days_probs = []
                        targets_inf_risk_all_days = []
                        preds_inf_risk_all_days = []
                        preds_inf_risk_all_days_probs = []
                        n_days = len(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"].keys())
                        for day in range(n_days):
                            # Getting targets and predictions
                            targets_all_days.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['true_seir_states'])
                            preds_all_days.append(np.argmax(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'], axis=1))
                            preds_all_days_probs.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'])
                            if (targets_inf_risk_all_days_last_epoch is not None):
                                targets_inf_risk_all_days.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['true_inf_risk_class'])
                                preds_inf_risk_all_days.append(np.argmax(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['seir_states_probs_inf_risk'], axis=1))
                                preds_inf_risk_all_days_probs.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['seir_states_probs_inf_risk'])
                        targets_all_days = np.concatenate(targets_all_days, axis=0)
                        preds_all_days = np.concatenate(preds_all_days, axis=0)
                        preds_all_days_probs = np.concatenate(preds_all_days_probs, axis=0)
                        if (targets_inf_risk_all_days_last_epoch is not None):
                            targets_inf_risk_all_days = np.concatenate(targets_inf_risk_all_days, axis=0)
                            preds_inf_risk_all_days = np.concatenate(preds_inf_risk_all_days, axis=0)
                            preds_inf_risk_all_days_probs = np.concatenate(preds_inf_risk_all_days_probs, axis=0)

                        # Reshaping targets_all_days, preds_all_days, and preds_all_days_probs in case of having a forecast horizon > 1
                        if (len(preds_all_days_probs.shape) > 2):                       
                            # Reshaping
                            targets_all_days = targets_all_days.reshape(-1)
                            preds_all_days = preds_all_days.reshape(-1)
                            preds_all_days_probs = preds_all_days_probs.transpose(0, 2, 1).reshape(-1, preds_all_days_probs.shape[1])

                        # Last epochs targets and preds
                        if (epoch == max(epochs_list)):
                            # States preds
                            targets_all_days_last_epoch[rep_id][data_split] = targets_all_days
                            preds_all_days_last_epoch[rep_id][data_split] = preds_all_days
                            preds_probs_all_days_last_epoch[rep_id][data_split] = preds_all_days_probs
                            # Infection risks preds
                            if (targets_inf_risk_all_days_last_epoch is not None):
                                targets_inf_risk_all_days_last_epoch[rep_id][data_split] = targets_inf_risk_all_days
                                preds_inf_risk_all_days_last_epoch[rep_id][data_split] = preds_inf_risk_all_days
                                preds_probs_inf_risk_all_days_last_epoch[rep_id][data_split] = preds_inf_risk_all_days_probs
                        
                        # Getting the metrics for the epoch
                        print(f"\n\n=========> {data_split.upper()} DATA SPLIT FOR EPOCH {epoch} <=========\n")
                        # STATES predictions metrics
                        # MCC, F1-Score and Balanced Accuracy
                        N_UNIQUE_CLASSES = [i for i in range(preds_all_days_probs.shape[1])]
                        sensitivity, specificity, mcc, f1_score_val, balanced_acc = get_classification_metrics(
                                                                                                                targets_all_days,
                                                                                                                preds_all_days,
                                                                                                                data_split,
                                                                                                                N_UNIQUE_CLASSES,
                                                                                                                verbose=verbose,
                                                                                                                print_classification_report=print_classification_report
                                                                                                            )
                        # AUC
                        if (preds_all_days_probs.shape[1] == 2): # Binary classification
                            auc = roc_auc_score(targets_all_days, preds_all_days_probs[:, 1], average="macro")
                            random_preds_probs = np.random.dirichlet(alpha=np.ones(2), size=len(targets_all_days))
                            auc_random = roc_auc_score(targets_all_days, random_preds_probs[:, 1], average="macro")
                        else:
                            # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                            # For HUG COVID dataset, we only keep the valid labels
                            if (params_exp['dataset_name'].lower() == 'hug'):
                                mask_valid_labels = targets_all_days != -1
                                if (params_exp['forecast_horizon'] == 1):
                                    mask_valid_labels = mask_valid_labels[:, 0]
                                    targets_all_days = targets_all_days[mask_valid_labels].squeeze()
                                else:
                                    targets_all_days = targets_all_days[mask_valid_labels]
                                preds_all_days_probs = preds_all_days_probs[mask_valid_labels]
                            auc = roc_auc_score(targets_all_days, preds_all_days_probs, multi_class='ovo', average="macro", labels=N_UNIQUE_CLASSES)
                            random_preds_probs = np.random.dirichlet(alpha=np.ones(max(N_UNIQUE_CLASSES)+1), size=len(targets_all_days))
                            auc_random = roc_auc_score(targets_all_days, random_preds_probs, multi_class='ovo', average="macro", labels=N_UNIQUE_CLASSES)
                        if (verbose):
                            print(f"\n{data_split} AUC: {auc}")
                            print(f"\t{data_split} AUC random classifier: {auc_random}")
                        # Per-class PR AUCs
                        if (preds_all_days_probs.shape[1] > 2):
                            per_class_pr_auc = []
                            per_class_pr_auc_random = []
                            for c in N_UNIQUE_CLASSES:
                                y_true_c = (targets_all_days == c).astype(int)
                                y_score_c = preds_all_days_probs[:, c]
                                y_score_random_c = random_preds_probs[:, c]
                                per_class_pr_auc.append(average_precision_score(y_true_c, y_score_c))
                                per_class_pr_auc_random.append(average_precision_score(y_true_c, y_score_random_c))
                            if (verbose):
                                print(f"\n{data_split} Per class AUC: {per_class_pr_auc}")
                                print(f"\t{data_split} Per class AUC random classifier: {per_class_pr_auc_random}")
                        
                        metrics_per_data_split["Sensitivity"][data_split][epoch][rep_id] = sensitivity 
                        metrics_per_data_split["Specificity"][data_split][epoch][rep_id] = specificity
                        metrics_per_data_split["MCC"][data_split][epoch][rep_id] = mcc 
                        metrics_per_data_split["F1Score"][data_split][epoch][rep_id] = f1_score_val 
                        metrics_per_data_split["BalancedAccuracy"][data_split][epoch][rep_id] = balanced_acc 
                        metrics_per_data_split["AUC"][data_split][epoch][rep_id] = auc 
                        if (preds_all_days_probs.shape[1] > 2):
                            metrics_per_data_split["PerClassAUC"][data_split][epoch][rep_id] = per_class_pr_auc 
                        # INFECTION RISK prediction metrics
                        if (targets_inf_risk_all_days_last_epoch is not None):
                            # MCC, F1-Score and Balanced Accuracy
                            N_UNIQUE_CLASSES_INF_RISK = [i for i in range(preds_inf_risk_all_days_probs.shape[1])]
                            sensitivity_inf_risk, specificity_inf_risk, mcc_inf_risk, f1_score_val_inf_risk, balanced_acc_inf_risk = get_classification_metrics(
                                                                                            targets_inf_risk_all_days,
                                                                                            preds_inf_risk_all_days,
                                                                                            data_split,
                                                                                            N_UNIQUE_CLASSES_INF_RISK,
                                                                                            verbose=verbose,
                                                                                            print_classification_report=print_classification_report
                                                                                        )
                            # AUC
                            if (preds_inf_risk_all_days_probs.shape[1] == 2): # Binary classification
                                auc_inf_risk = roc_auc_score(targets_inf_risk_all_days, preds_inf_risk_all_days_probs[:, 1], average="macro")
                                random_preds_probs_inf_risk = np.random.dirichlet(alpha=np.ones(2), size=len(targets_inf_risk_all_days))
                                auc_random_inf_risk = roc_auc_score(targets_inf_risk_all_days, random_preds_probs_inf_risk[:, 1], average="macro")
                            else:
                                # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                                auc_inf_risk = roc_auc_score(targets_inf_risk_all_days, preds_inf_risk_all_days_probs, multi_class='ovo', average="macro")
                                random_preds_probs_inf_risk = np.random.dirichlet(alpha=np.ones(max(N_UNIQUE_CLASSES_INF_RISK)+1), size=len(targets_inf_risk_all_days))
                                auc_random_inf_risk = roc_auc_score(targets_inf_risk_all_days, random_preds_probs_inf_risk, multi_class='ovo', average="macro")
                            if (verbose):
                                print(f"\n{data_split} AUC: {auc}")
                                print(f"\t{data_split} AUC random classifier: {auc_random}")
                            # Per-class PR AUCs
                            if (preds_inf_risk_all_days_probs.shape[1] > 2):
                                per_class_pr_auc_inf_risk = []
                                per_class_pr_auc_random_inf_risk = []
                                for c in N_UNIQUE_CLASSES_INF_RISK:
                                    y_true_c_inf_risk = (targets_inf_risk_all_days == c).astype(int)
                                    y_score_c_inf_risk = preds_inf_risk_all_days_probs[:, c]
                                    y_score_random_c_inf_risk = random_preds_probs_inf_risk[:, c]
                                    per_class_pr_auc_inf_risk.append(average_precision_score(y_true_c_inf_risk, y_score_c_inf_risk))
                                    per_class_pr_auc_random_inf_risk.append(average_precision_score(y_true_c_inf_risk, y_score_random_c_inf_risk))
                                if (verbose):
                                    print(f"\n{data_split} Per class AUC: {per_class_pr_auc_inf_risk}")
                                    print(f"\t{data_split} Per class AUC random classifier: {per_class_pr_auc_random_inf_risk}")
                            metrics_per_data_split["SensitivityInfectionRisk"][data_split][epoch][rep_id] = sensitivity_inf_risk
                            metrics_per_data_split["SpecificityInfectionRisk"][data_split][epoch][rep_id] = specificity_inf_risk
                            metrics_per_data_split["MCCInfectionRisk"][data_split][epoch][rep_id] = mcc_inf_risk
                            metrics_per_data_split["F1ScoreInfectionRisk"][data_split][epoch][rep_id] = f1_score_val_inf_risk
                            metrics_per_data_split["BalancedAccuracyInfectionRisk"][data_split][epoch][rep_id] = balanced_acc_inf_risk
                            metrics_per_data_split["AUCInfectionRisk"][data_split][epoch][rep_id] = auc_inf_risk
                            if (preds_inf_risk_all_days_probs.shape[1] > 2):
                                metrics_per_data_split["PerClassAUCInfectionRisk"][data_split][epoch][rep_id] = per_class_pr_auc_inf_risk

        # Plotting the different metrics over the epohs
        for metric_type in metrics_per_data_split:
            if (metric_type != 'PerClassAUC'):
                plot_metric_epochs(metric_dict=metrics_per_data_split[metric_type], metric_name=metric_type)

        # Plot the TEST metrics for the last epoc
        # STATES PREDICTIONS
        if (params_exp['forecast_horizon'] > 1):
            print("\n===> VERY IMPORTANT: as the forecast horizon is greater than 1, the metrics are computed by concatenating all the predictions for all individuals, days, and forecast window! So metrics are computed over more values than for a forecast horizon of 1 (n_individuals*n_days*forecast_hor_length values instead of n_individuals*n_days values\n) \n")
        last_epoch = sorted(list(metrics_per_data_split["MCC"]["Test"].keys()))[-1]
        last_sensitivity_test_mean = np.mean(metrics_per_data_split["Sensitivity"]["Test"][last_epoch])
        last_sensitivity_test_std = np.std(metrics_per_data_split["Sensitivity"]["Test"][last_epoch])
        last_specificity_test_mean = np.mean(metrics_per_data_split["Specificity"]["Test"][last_epoch])
        last_specificity_test_std = np.std(metrics_per_data_split["Specificity"]["Test"][last_epoch])
        last_mcc_test_mean = np.mean(metrics_per_data_split["MCC"]["Test"][last_epoch])
        last_mcc_test_std = np.std(metrics_per_data_split["MCC"]["Test"][last_epoch])
        last_f1_score_test_mean = np.mean(metrics_per_data_split["F1Score"]["Test"][last_epoch])
        last_f1_score_test_std = np.std(metrics_per_data_split["F1Score"]["Test"][last_epoch])
        last_balanced_acc_test_mean = np.mean(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
        last_balanced_acc_test_std = np.std(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
        last_auc_test_mean = np.mean(metrics_per_data_split["AUC"]["Test"][last_epoch])
        last_auc_test_std = np.std(metrics_per_data_split["AUC"]["Test"][last_epoch])
        if (preds_all_days_probs.shape[1] > 2):
            last_per_class_auc_test_mean = np.mean(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
            last_per_class_auc_test_std = np.std(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
        print("\n=========> TEST Sensitivity in the last epoch: {} +- {}%".format(last_sensitivity_test_mean, last_sensitivity_test_std))
        print("\tTEST Specificity in the last epoch: {} +- {}%".format(last_specificity_test_mean, last_specificity_test_std))
        print("\tTEST MCC in the last epoch: {} +- {}%".format(last_mcc_test_mean, last_mcc_test_std))
        print("\tTEST F1 Score in the last epoch: {} +- {}%".format(last_f1_score_test_mean, last_f1_score_test_std))
        print("\tTEST Balanced Accuracy in the last epoch: {} +- {}%".format(last_balanced_acc_test_mean, last_balanced_acc_test_std))
        print("\tTEST AUC in the last epoch: {} +- {}%".format(last_auc_test_mean*100, last_auc_test_std*100))
        if (preds_all_days_probs.shape[1] > 2):
            for tmp_class in range(len(last_per_class_auc_test_mean)):
                print("\tTEST PER CLASS AUC in the last epoch for class {}: {} +- {}%".format(tmp_class, last_per_class_auc_test_mean[tmp_class]*100, last_per_class_auc_test_std[tmp_class]*100))
        # INFECTION RISK predictions
        if (targets_inf_risk_all_days_last_epoch is not None):
            if (params_exp['forecast_horizon'] > 1):
                print("\n===> VERY IMPORTANT: as the forecast horizon is greater than 1, the metrics are computed by concatenating all the predictions for all individuals, days, and forecast window! So metrics are computed over more values than for a forecast horizon of 1 (n_individuals*n_days*forecast_hor_length values instead of n_individuals*n_days values\n) \n")
            last_sensitivity_inf_risk_test_mean = np.mean(metrics_per_data_split["SensitivityInfectionRisk"]["Test"][last_epoch])
            last_sensitivity_inf_risk_test_std = np.std(metrics_per_data_split["SensitivityInfectionRisk"]["Test"][last_epoch])
            last_specificity_inf_risk_test_mean = np.mean(metrics_per_data_split["SpecificityInfectionRisk"]["Test"][last_epoch])
            last_specificity_inf_risk_test_std = np.std(metrics_per_data_split["SpecificityInfectionRisk"]["Test"][last_epoch])
            last_mcc_inf_risk_test_mean = np.mean(metrics_per_data_split["MCCInfectionRisk"]["Test"][last_epoch])
            last_mcc_inf_risk_test_std = np.std(metrics_per_data_split["MCCInfectionRisk"]["Test"][last_epoch])
            last_f1_score_inf_risk_test_mean = np.mean(metrics_per_data_split["F1ScoreInfectionRisk"]["Test"][last_epoch])
            last_f1_score_inf_risk_test_std = np.std(metrics_per_data_split["F1ScoreInfectionRisk"]["Test"][last_epoch])
            last_balanced_acc_inf_risk_test_mean = np.mean(metrics_per_data_split["BalancedAccuracyInfectionRisk"]["Test"][last_epoch])
            last_balanced_acc_inf_risk_test_std = np.std(metrics_per_data_split["BalancedAccuracyInfectionRisk"]["Test"][last_epoch])
            last_auc_inf_risk_test_mean = np.mean(metrics_per_data_split["AUCInfectionRisk"]["Test"][last_epoch])
            last_auc_inf_risk_test_std = np.std(metrics_per_data_split["AUCInfectionRisk"]["Test"][last_epoch])
            if (preds_inf_risk_all_days_probs.shape[1] > 2):
                last_per_class_auc_inf_risk_test_mean = np.mean(metrics_per_data_split["PerClassAUCInfectionRisk"]["Test"][last_epoch], axis=0)
                last_per_class_auc_inf_risk_test_std = np.std(metrics_per_data_split["PerClassAUCInfectionRisk"]["Test"][last_epoch], axis=0)
            print("\n=========> TEST Sensitivity INFECTION RISK in the last epoch: {} +- {}%".format(last_sensitivity_inf_risk_test_mean, last_sensitivity_inf_risk_test_std))
            print("\tTEST Specificity INFECTION RISK in the last epoch: {} +- {}%".format(last_specificity_inf_risk_test_mean, last_specificity_inf_risk_test_std))
            print("\tTEST MCC INFECTION RISK in the last epoch: {} +- {}%".format(last_mcc_inf_risk_test_mean, last_mcc_inf_risk_test_std))
            print("\tTEST F1 Score INFECTION RISK in the last epoch: {} +- {}%".format(last_f1_score_inf_risk_test_mean, last_f1_score_inf_risk_test_std))
            print("\tTEST Balanced Accuracy INFECTION RISK in the last epoch: {} +- {}%".format(last_balanced_acc_inf_risk_test_mean, last_balanced_acc_inf_risk_test_std))
            print("\tTEST AUC INFECTION RISK in the last epoch: {} +- {}%".format(last_auc_inf_risk_test_mean*100, last_auc_inf_risk_test_std*100))
            if (preds_inf_risk_all_days_probs.shape[1] > 2):
                for tmp_class in range(len(last_per_class_auc_inf_risk_test_mean)):
                    print("\tTEST PER CLASS AUC in the last epoch for class {}: {} +- {}%".format(tmp_class, last_per_class_auc_inf_risk_test_mean[tmp_class]*100, last_per_class_auc_inf_risk_test_std[tmp_class]*100))

    # Get classification performances per class
    if (print_average_metrics_over_days):
        if (params_exp['forecast_horizon'] == 1):
            is_hug_dataset = params_exp['dataset_name'].lower() == 'hug'
            get_classification_metrics_per_days(results_h5_file, N_UNIQUE_CLASSES, data_split_to_use='Test', is_hug_dataset=is_hug_dataset, folder_save_results=results_folder)
        else:
            print(f"\nMetrics averaged first by day and then by repetition is not implemented for forecast horizon greater than 2")

    # IMPORTANT: WHEN FORECAST HORIZON IS GREATER THAN 1, WE HAVE TARGETS PER DAY OF SHAPE (N_INDIVIDUAL, FORECAST_HORIZON) (INSTEAD OF (N_INDIVIDUALS))
    # IMPORTANT: AND PREDICTED PROBS PER DAY PF SJAÈE (N_INDIVIDUAL, N_CLASSES, FORECAST_HORIZON) (INSTEAD OF SHAPE (N_INDIVIDUAL, N_CLASSES)).
    # IMPORTANT: HOWEVER, IN THAT CASE, we cannot compute the metrics for each sample and forecast window, as all sample can be of the same class (GT) so
    # IMPORTANT: classic metrics such as accuracy, F1-score, precision, recall, AUC, MCC are meaningless in that case.


    #======================================================================#
    #=====================Plot SEIR evolution over time=====================#
    #======================================================================#
    # Mapping
    if (params_exp['dataset_name'].lower() == 'sociopatterns'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    elif (params_exp['dataset_name'].lower() == 'murcia'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}

    # Data to use
    #epoch_to_use = 0
    #data_split = "Train"
    #data_split = "Val"
    data_split = "Test"
    last_epoch = max([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"][data_split].keys())])
    epoch_to_use = last_epoch

    # Plot
    plot_pred_epidemic_evolution(
                                    h5_results_file=results_h5_file,
                                    epoch_to_use=epoch_to_use,
                                    states_mapping=MAPPING,
                                    data_split=data_split
                                )
        


    #======================================================================#
    #=====================Calibration error=====================#
    #======================================================================#
    # Get the expected calibration error ECE
    #data_split_to_use = 'Val'
    data_split_to_use = 'Test'
    ece_per_rep = []
    N_BINS = 10 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    #N_BINS = 15 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    for rep_ID in preds_probs_all_days_last_epoch:
        ece_rep = compute_ece(
                                    probs=preds_probs_all_days_last_epoch[rep_ID][data_split_to_use],
                                    labels=targets_all_days_last_epoch[rep_ID][data_split_to_use],
                                    n_bins=N_BINS
                            )
        ece_per_rep.append(ece_rep)

    # Getting the mean ECE
    mean_ece = np.mean(ece_per_rep)
    std_ece = np.std(ece_per_rep)
    print(f"\nThe {data_split_to_use} ECE in the LAST epoch is: {mean_ece} +- {std_ece}\n")
    if (mean_ece < 0.05):
        print(f"\n\t=========>The model is relatively well calibrated\n")
    elif (mean_ece >= 0.05) and (mean_ece <= 0.1):
        print(f"\n\t=========>The model has acceptable calibrated BUT it has noticeable miscalibration\n")
    else:
        print(f"\n\t=========>The model has a significant calibration issue (common for deep networks, especially overconfident ones)\n")
        


if (__name__=='__main__'):
    main()