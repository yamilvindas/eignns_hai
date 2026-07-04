import os
import h5py
import yaml
import pickle
import math
import argparse
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, matthews_corrcoef, 
    balanced_accuracy_score, recall_score, precision_score,
    accuracy_score, roc_curve
)
from sklearn.metrics import auc as general_auc # Computes the area under a curve (not necessarily ROC)
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from imblearn.metrics import sensitivity_score, specificity_score

# Suppress undefined metric warnings (common in early epochs or rare classes)
warnings.filterwarnings('ignore') 

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
        Created with the help of Gemini (validated).
        Computes ECE for multi-class classification.
    """
    pred_y = np.argmax(y_prob, axis=-1)
    confidences = np.max(y_prob, axis=-1)
    accuracy = (pred_y == y_true)
    
    ece = 0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Filter predictions in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracy[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def plot_analyze_multiclass_failure(y_true, y_pred, class_names):
    # 1. Calculate the Raw and Adjusted Balanced Accuracy
    raw_bal_acc = balanced_accuracy_score(y_true, y_pred)
    n_classes = len(class_names)
    adj_bal_acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    
    print(f"Raw Balanced Accuracy: {raw_bal_acc:.2f}")
    print(f"Adjusted Balanced Accuracy: {adj_bal_acc:.2f}")

    # 2. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Look for high values OUTSIDE the diagonal')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def calculate_metrics(y_true_flat, y_probs_flat, y_current_flat, transitions_preds=False, analyze_multiclass_failure=False):
    """
        Created with the help of Gemini (validated).
        Computes global metrics AND transition-specific metrics to detect Identity Mapping.
    """
    # Getting predictions
    y_pred_flat = np.argmax(y_probs_flat, axis=1)
    n_classes = y_probs_flat.shape[1]

    # Getting random predictions
    y_random_flat = np.random.randint(0, n_classes, size=y_pred_flat.shape)

    # Global Metrics ---
    # Adjusted Balanced Accuracy
    adj_bal_acc = balanced_accuracy_score(y_true_flat, y_pred_flat, adjusted=True)
    adj_bal_acc_random = balanced_accuracy_score(y_true_flat, y_random_flat, adjusted=True)

    # Sensitivity and specificity macro-averaged to handla class imbalance
    if (n_classes == 2):
        sensitivity = sensitivity_score(y_true_flat, y_pred_flat)
        sensitivity_random = sensitivity_score(y_true_flat, y_random_flat)
        specificity = specificity_score(y_true_flat, y_pred_flat)
        specificity_random = specificity_score(y_true_flat, y_random_flat)
    else:
        sensitivity = sensitivity_score(y_true_flat, y_pred_flat, average='macro')
        sensitivity_random = sensitivity_score(y_true_flat, y_random_flat, average='macro')
        specificity = specificity_score(y_true_flat, y_pred_flat, average='macro')
        specificity_random = specificity_score(y_true_flat, y_random_flat, average='macro')

    # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true_flat, y_probs_flat[:, 1])
        else:
            auc = roc_auc_score(y_true_flat, y_probs_flat, multi_class='ovr', average='macro')
    except:
        auc = 0.5 # Fallback if only one class present

    # AUPRC (Average Precision)
    # Average Precision is the standard proxy for AUPRC in sklearn
    try:
        if n_classes == 2:
            auprc = average_precision_score(y_true_flat, y_probs_flat[:, 1])
        else:
            # Macro-average AUPRC (Average Precision) across all classes
            auprc = average_precision_score(y_true_flat_one_hot, y_probs_flat, average='macro')
    except:
        auprc = 0.0

    # MCC
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    mcc_random = matthews_corrcoef(y_true_flat, y_random_flat)
    
    # ECE
    ece = expected_calibration_error(y_true_flat, y_probs_flat)

    # Metrics
    if (transitions_preds):
        suffix = 'Transitions'
    else:
        suffix = ''
    metrics = {
                f'Sensitivity{suffix}': sensitivity,
                f'Specificity{suffix}': specificity,
                f'BalancedAccuracy{suffix}': adj_bal_acc,
                f'AUC{suffix}': auc,
                f'MCC{suffix}': mcc,
                f'ECE{suffix}': ece,
                f'AUCPR{suffix}': auprc,

                f'SensitivityRandom{suffix}': sensitivity_random,
                f'SpecificityRandom{suffix}': specificity_random,
                f'BalancedAccuracyRandom{suffix}': adj_bal_acc_random,
                f'MCCRandom{suffix}': mcc_random,
            }
    
    # AUCROC and AUPR per class
    # Variables for ROC
    fpr = dict()
    tpr = dict()
    per_class_aucroc = dict()
    # Variables for PR curves
    per_class_auprc = dict()
    precision = dict()
    recall = dict()
    baseline = dict()
    # One hot encoding of the true labels
    fixed_classes = [[i for i in range(n_classes)]] # OneHotEncoder asks for a list of array
    encoder = OneHotEncoder(sparse_output=False, categories=fixed_classes)
    y_true_flat_one_hot = encoder.fit_transform(y_true_flat.reshape(-1, 1)) # Necessary to get the per_class AUC
    # Getting values
    for i in range(n_classes):
        # AUC ROC
        fpr[i], tpr[i], _ = roc_curve(y_true_flat_one_hot[:, i], y_probs_flat[:, i])
        per_class_aucroc[i] = general_auc(fpr[i], tpr[i])

        # AUC PR
        # This is specifically useful for the 'Infected' class (usually class 2)
        precision[i], recall[i], _ = precision_recall_curve(y_true_flat_one_hot[:, i], y_probs_flat[:, i])
        baseline[i] = np.sum(y_true_flat_one_hot[:, i]) / len(y_true_flat_one_hot[:, i])
        per_class_auprc[i] = average_precision_score(y_true_flat_one_hot[:, i], y_probs_flat[:, i])
    metrics[f'PerClassAUC{suffix}'] = per_class_aucroc
    metrics[f'PerClassFPR{suffix}'] = fpr
    metrics[f'PerClassTPR{suffix}'] = tpr
    metrics[f'PerClassAUCPR{suffix}'] = per_class_auprc
    metrics[f'PerClassPrecision{suffix}'] = precision
    metrics[f'PerClassRecall{suffix}'] = recall
    metrics[f'PerClassPRBaseline{suffix}'] = baseline

    # Identity Mapping / Transition Metrics ---
    if (not transitions_preds):
        # Identify masks
        # Transition: Ground Truth is DIFFERENT from Current State (at t=0)
        # Stability: Ground Truth is SAME as Current State
        mask_transition = (y_true_flat != y_current_flat)
        mask_stability = (y_true_flat == y_current_flat)


        # Transition Sensitivity (Did we catch the change?)
        if (np.sum(mask_transition) > 0):
            metrics['Transition_Sensitivity'] = sensitivity_score(
                                                                y_true_flat[mask_transition], 
                                                                y_pred_flat[mask_transition], 
                                                                average='macro'
                                                            )
            metrics['Transition_Sensitivity_Random'] = sensitivity_score(
                                                                            y_true_flat[mask_transition], 
                                                                            y_random_flat[mask_transition], 
                                                                            average='macro'
                                                                        )
            # Accuracy on transitions (When we predicted change, was it right?)
            # Note: This is harder to define macro-wise, simplified to accuracy on transition set
            #metrics['Transition_Accuracy'] = accuracy_score(y_true_flat[mask_transition], y_pred_flat[mask_transition])
            metrics['Transition_Balanced_Accuracy'] = balanced_accuracy_score(y_true_flat[mask_transition], y_pred_flat[mask_transition], adjusted=True)
            metrics['Transition_Balanced_Accuracy_Random'] = balanced_accuracy_score(y_true_flat[mask_transition], y_random_flat[mask_transition], adjusted=True)


            # Analyze multi-class failure
            if (analyze_multiclass_failure):
                unique_y_true_masked = {str(val) for val in np.unique(y_true_flat[mask_transition])}
                unique_y_pred_masked = {str(val) for val in np.unique(y_pred_flat[mask_transition])}
                unique_y_vals_masked = unique_y_true_masked | unique_y_pred_masked
                plot_analyze_multiclass_failure(
                                            y_true=y_true_flat[mask_transition],
                                            y_pred=y_pred_flat[mask_transition],
                                            class_names=unique_y_vals_masked
                                        )
                # if (metrics['Transition_Balanced_Accuracy'] < 0):
                #     breakpoint()


        # Stability Sensitivity (Did we correctly predict 'no change'?)
        if (np.sum(mask_stability) > 0):
            metrics['Stability_Recall'] = sensitivity_score(
                                                                y_true_flat[mask_stability], 
                                                                y_pred_flat[mask_stability], 
                                                                average='macro'
                                                            )
            metrics['Stability_Recall_Random'] = sensitivity_score(
                                                                y_true_flat[mask_stability], 
                                                                y_random_flat[mask_stability], 
                                                                average='macro'
                                                            )

    return metrics


def get_metrics_per_rep(h5_path, parameters_exp={}, analyze_multiclass_failure=False):
    """
        Created with the help of Gemini.
        Computes the metrics of an experiment using the HDF5 results file.
    """
    # Storage for aggregation
    # structure: metrics_storage[metric_name][split][epoch] = [list of values from reps]
    metrics_storage = {} 
    # structure: loss_storage[loss_type][split] = [array of arrays (epochs)]
    loss_storage = {}
    n_classes = None
    with h5py.File(h5_path, 'r') as f:
        rep_ids = list(f.keys())
        print(f"Found {len(rep_ids)} repetitions: {rep_ids}")

        for rep_id in rep_ids:
            print(f"Processing {rep_id}...")
            
            # --- 1. Process LOSS ---
            if 'Loss' in f[rep_id]:
                loss_grp = f[rep_id]['Loss']
                for split in loss_grp.keys():
                    for loss_type in loss_grp[split].keys():
                        loss_vals = np.array(loss_grp[split][loss_type][:])
                        
                        if loss_type not in loss_storage: loss_storage[loss_type] = {}
                        if split not in loss_storage[loss_type]: loss_storage[loss_type][split] = []
                        
                        loss_storage[loss_type][split].append(loss_vals)

            # --- 2. Process PREDICTIONS ---
            if ('Preds' in f[rep_id]):
                preds_grp = f[rep_id]['Preds']
                for split in preds_grp.keys():
                    # Sort epochs numerically (Epoch-1, Epoch-2...)
                    epoch_keys = [k for k in preds_grp[split].keys() if k.startswith('Epoch-')]
                    # simple helper to sort 'Epoch-10' correctly vs 'Epoch-2'
                    if ('continual_training' not in parameters_exp) or (not parameters_exp['continual_training']):
                        epoch_keys.sort(key=lambda x: int(x.split('-')[1]))

                    for epoch_key in epoch_keys:
                        if (len(preds_grp[split][epoch_key]) != 0):
                            if ('continual_training' not in parameters_exp) or (not parameters_exp['continual_training']):
                                epoch_num = int(epoch_key.split('-')[1])
                            else:
                                epoch_num = 'Last'

                            # Accumulate data for this epoch across ALL days
                            all_true = []
                            all_probs = []
                            all_current = []
                            if (parameters_exp["predict_state_transitions"]):
                                all_true_transitions = []
                                all_pred_transitions_probs = []
                                
                            day_grp = preds_grp[split][epoch_key]
                            for day_key in day_grp.keys():
                                d_data = day_grp[day_key]
                                
                                # Load raw data
                                # Shapes: 
                                # true: [N, W]
                                # prob: [N, C, W] or [N, C]
                                # curr: [N, C] (One-hot)
                                
                                curr = np.array(d_data['current_epi_state'])
                                true = np.array(d_data['true_epi_states'])
                                prob = np.array(d_data['pred_epi_states_probs'])
                                if (parameters_exp["predict_state_transitions"]):
                                    true_transitions = np.array(d_data["true_transitions"])
                                    pred_transitions_probs = np.array(d_data["pred_transitions_probs"])

                                # Pre-processing to align shapes (Flattening Window)
                                N_nodes = true.shape[0]
                                
                                # 1. Handle Current State (One-hot -> Index)
                                if curr.ndim > 1:
                                    curr_idx = np.argmax(curr, axis=1) # [N]
                                else:
                                    curr_idx = curr # [N]
                                
                                # Handre
                                
                                # 2. Handle Window (W)
                                # If W > 1, we must repeat 'current' state W times to match 'true'
                                # so we can compare current vs true for transition logic
                                if (true.ndim > 1):
                                    W = true.shape[1]
                                    # Flatten True: [N, W] -> [N*W]
                                    true_flat = true.flatten()
                                    
                                    # Flatten Probs: [N, C, W] -> [N, W, C] -> [N*W, C]
                                    if prob.ndim == 3: # [N, C, W]
                                        prob_perm = np.transpose(prob, (0, 2, 1)) # [N, W, C]
                                        prob_flat = prob_perm.reshape(-1, prob.shape[1])
                                    else:
                                        # If probs are static [N, C], repeat them? 
                                        # Usually probs are [N, C, W]. Assuming 3D.
                                        prob_flat = prob.reshape(N_nodes, -1) 
                                    
                                    # Repeat Current: [N] -> [N, W] -> [N*W]
                                    # The idea is to have [curr_state_t_0, ..., curr_state_t_0, curr_state_t_1, ..., curr_state_t_1, ...] where the current state is repeated over the forecast window to be able to know if there is a state transition within the forecast window using true.
                                    curr_repeated = np.repeat(curr_idx[:, np.newaxis], W, axis=1).flatten()
                                    
                                else:
                                    true_flat = true
                                    prob_flat = prob
                                    curr_repeated = curr_idx

                                all_true.append(true_flat)
                                all_probs.append(prob_flat)
                                all_current.append(curr_repeated)
                                if (parameters_exp["predict_state_transitions"]):
                                    all_true_transitions.append(true_transitions)
                                    all_pred_transitions_probs.append(pred_transitions_probs)

                            # Concatenate all days for this epoch
                            y_true_epoch = np.concatenate(all_true)
                            y_probs_epoch = np.concatenate(all_probs)
                            y_curr_epoch = np.concatenate(all_current)
                            if (parameters_exp["predict_state_transitions"]):
                                y_true_transitions_epoch = np.concatenate(all_true_transitions)
                                y_transitions_probs_epoch = np.concatenate(all_pred_transitions_probs)

                            # Number of classes 
                            if (n_classes is None):
                                n_classes = y_probs_epoch.shape[1]
                            
                            # Calculate Metrics
                            # Store main state classification metrics
                            epoch_metrics = calculate_metrics(y_true_epoch, y_probs_epoch, y_curr_epoch, analyze_multiclass_failure)
                            for m_name, m_val in epoch_metrics.items():
                                if m_name not in metrics_storage: metrics_storage[m_name] = {}
                                if split not in metrics_storage[m_name]: metrics_storage[m_name][split] = {}
                                if epoch_num not in metrics_storage[m_name][split]: 
                                    metrics_storage[m_name][split][epoch_num] = []

                                metrics_storage[m_name][split][epoch_num].append(m_val)

                            # Store state transitions classification metrics
                            if (parameters_exp["predict_state_transitions"]):
                                epoch_transitions_metrics = calculate_metrics(y_true_transitions_epoch, y_transitions_probs_epoch, y_curr_epoch, transitions_preds=True, analyze_multiclass_failure=analyze_multiclass_failure)
                                for m_name, m_val in epoch_transitions_metrics.items():
                                    m_name
                                    if m_name not in metrics_storage: metrics_storage[m_name] = {}
                                    if split not in metrics_storage[m_name]: metrics_storage[m_name][split] = {}
                                    if epoch_num not in metrics_storage[m_name][split]: 
                                        metrics_storage[m_name][split][epoch_num] = []

                                    metrics_storage[m_name][split][epoch_num].append(m_val)
    
    return loss_storage, metrics_storage, n_classes

def plot_metric_curve(metric_name, data_dict, save_path, show_plot=True):
    """
        Created with the help of Gemini (validated).
        Plots mean +/- std for a given metric across epochs.
    """
    # Get the data splits
    splits = data_dict.keys()
    colors = {'Train': 'blue', 'Val': 'orange', 'Test': 'green'}

    # Create figure
    metrics_statistics = {}
    for split in splits:
        epochs = sorted(data_dict[split].keys())
        means = []
        stds = []
        for ep in epochs:
            values = data_dict[split][ep]
            means.append(np.mean(values))
            stds.append(np.std(values))
        means = np.array(means)
        stds = np.array(stds)
        xs = np.array(epochs)
        color = colors.get(split, 'gray')
        plt.plot(xs, means, label=f"{split} Mean", color=color)
        plt.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
        
        # Print final stats
        if (metric_name != 'ECE') and (metric_name != 'ECETransitions'):
            metric_mean = means[-1]*100
            metric_std = stds[-1]*100
        else:
            metric_mean = means[-1]
            metric_std = stds[-1]
        
        print(f"[{metric_name}] {split} Last Epoch ({epochs[-1]}): {metric_mean:.4f} ± {metric_std:.4f}")
        metrics_statistics[split] = {'Mean': metric_mean, 'Std': metric_std}

    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric_name}.png"))
    if (show_plot): # Show AFTER saving image, to avoid saving empty images
        plt.show()
    plt.close()

    return metrics_statistics



def threshold_sweep(h5_file_path, rep_id, split='Test', epoch_id=-1, print_report=False):
    """
        Created with the help of Gemini (validated).
        Allows to see the influence of the used threshold for prediction, on the transition
        and stability performance.
    """
    # Thresholds to use
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    if (print_report):
        print(f"--- Threshold Sweep for {rep_id} | {split} | {epoch_id} ---")
        print(f"{'Threshold':<12} | {'Stability Rec':<15} | {'Trans Precision':<15} | {'Trans Recall':<15}")
        print("-" * 65)

    # Getting th results per treshold
    sweep_results = []
    with h5py.File(h5_file_path, 'r') as f:
        # Selecting a given repetition and epoch
        reps_list = [rep_key for rep_key in f]
        epochs_list = sorted([key for key in f[reps_list[rep_id]]['Preds'][split]])
        day_grp = f[reps_list[rep_id]]['Preds'][split][epochs_list[epoch_id]]
        
        # Variables to compute metrics
        all_true = []
        all_probs = []
        all_current = []

        # Iterating over the days (concatenating results)
        for day in day_grp.keys():
            # Get current day data
            true = np.array(day_grp[day]['true_epi_states']).flatten()
            probs = np.array(day_grp[day]['pred_epi_states_probs'])
            curr = np.array(day_grp[day]['current_epi_state'])
            
            # Index current state
            curr_idx = np.argmax(curr, axis=1) if curr.ndim > 1 else curr
            
            # Handle Window flattening
            if (probs.ndim == 3): # [N, C, W] -> [N*W, C]
                W = probs.shape[2]
                probs = np.transpose(probs, (0, 2, 1)).reshape(-1, probs.shape[1])
                curr_repeated = np.repeat(curr_idx, W)
            else:
                curr_repeated = curr_idx

            all_true.append(true)
            all_probs.append(probs)
            all_current.append(curr_repeated)

        y_true = np.concatenate(all_true)
        y_probs = np.concatenate(all_probs)
        y_curr = np.concatenate(all_current)

        for t in thresholds:
            # Get Argmax Predictions
            y_pred_raw = np.argmax(y_probs, axis=1)
            max_probs = np.max(y_probs, axis=1)
            
            # Apply Threshold Logic:
            # If confidence < threshold AND it's a predicted transition, force to Current State
            final_preds = y_pred_raw.copy()
            mask_change = (y_pred_raw != y_curr)
            mask_uncertain = (max_probs < t)
            final_preds[mask_change & mask_uncertain] = y_curr[mask_change & mask_uncertain]

            # 3. Compute Metrics
            mask_trans_gt = (y_true != y_curr)
            mask_stable_gt = (y_true == y_curr)
            
            # Precision on transitions: (True and Correct Transitions / Predicted Transitions)
            pred_changes = (final_preds != y_curr) # Keep only transitions
            if (pred_changes.any()):
                trans_prec = np.mean(y_true[pred_changes] == final_preds[pred_changes]) # Keep only correct transitions
            else:
                print(f"\n\n=========> No predicted changes for threshold t = {t}\n")
                trans_prec = 0.0

            #trans_rec = recall_score(y_true[mask_trans_gt], final_preds[mask_trans_gt], average='macro', zero_division=0)
            trans_rec = sensitivity_score(y_true[mask_trans_gt], final_preds[mask_trans_gt], average='macro')
            #stab_rec = recall_score(y_true[mask_stable_gt], final_preds[mask_stable_gt], average='macro', zero_division=0)
            stab_rec = sensitivity_score(y_true[mask_stable_gt], final_preds[mask_stable_gt], average='macro')

            sweep_results.append(
                                    {
                                        'Threshold': t,
                                        'Trans_Precision': trans_prec,
                                        'Trans_Recall': trans_rec,
                                        'Stability_Rec': stab_rec
                                    }
                                )

            if (print_report):
                print(f"{t:<12.2f} | {stab_rec:<15.4f} | {trans_prec:<15.4f} | {trans_rec:<15.4f}")
    
    return sweep_results


def plot_calibration_for_transitions(h5_file_path, rep_id, save_path, split='Test', epoch_id=-1, show_plot=True):
    """
        Created with the help of Gemini (validated).
        Computes the calibration plot for the transitions (capacity of determine correct
        transitions).
    """
    # Variables to compute metrics
    all_true_binary = []
    all_probs_target = []

    with h5py.File(h5_file_path, 'r') as f:
        # Selecting a given repetition and epoch
        reps_list = [rep_key for rep_key in f]
        epochs_list = sorted([key for key in f[reps_list[rep_id]]['Preds']['Test']])
        day_grp = f[reps_list[rep_id]]['Preds'][split][epochs_list[epoch_id]]
        
        # Iterating over the days (concatenating results)
        for day in day_grp.keys():
            # Get current day data
            true = np.array(day_grp[day]['true_epi_states']).flatten()
            probs = np.array(day_grp[day]['pred_epi_states_probs'])
            curr = np.array(day_grp[day]['current_epi_state'])
            curr_idx = np.argmax(curr, axis=1) if curr.ndim > 1 else curr
            
            # Identify Transitions: We only care about predicting a CHANGE
            # Binary target: 1 if state at t+W != state at t=0, else 0
            if (probs.ndim == 3):
                W = probs.shape[2]
                curr_repeated = np.repeat(curr_idx, W)
                probs_reshaped = np.transpose(probs, (0, 2, 1)).reshape(-1, probs.shape[1])
            else:
                curr_repeated = curr_idx
                probs_reshaped = probs

            # Binary Label: Did a transition happen?
            y_true_binary = (true != curr_repeated).astype(int)
            
            # Transition Probability: 1 - P(Current_State)
            # This is the total probability the model assigns to "Changing"
            prob_transition = 1.0 - probs_reshaped[np.arange(len(curr_repeated)), curr_repeated]

            # Variables to compute metrics
            all_true_binary.append(y_true_binary)
            all_probs_target.append(prob_transition)

    y_true = np.concatenate(all_true_binary)
    y_prob = np.concatenate(all_probs_target)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="GNN Risk Model")

    plt.ylabel("Observed Fraction of Transitions")
    plt.xlabel("Mean Predicted Probability of Transition")
    plt.title(f"Calibration Plot (Reliability Diagram) - {split}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"CalibrationPlot_TransitionFromStatesPred.png"))
    if (show_plot):
        plt.show()



def evaluate_infection_onset(h5_file_path, rep_id, model_type='SIR', split='Test', epoch_id=-1, threshold=0.7):
    """
        Created with the help of Gemini (validated).
        Evaluates the model's ability to predict the next stage of infection for 
        at-risk patients (S or E).
    """
    # State Mapping (Standardizing across datasets)
    # SIR: 0:S, 1:I, 2:R
    # SEIR/SEIRD: 0:S, 1:E, 2:I, 3:R, ...
    S_STATE = 0
    if (model_type == 'SIR'):
        I_STATE = 1
        E_STATE = None # No Exposed state in SIR model
    else:
        E_STATE = 1
        I_STATE = 2 # Usually I is 2 in SEIR and SEIRD-NS

    # Results dictionary
    results = {
                    'S_to_Infected_Sensitivity': 0.0,
                    'S_to_Infected_Specificity': 0.0,
                    'S_to_Infected_BalAcc': 0.0,
                    'S_to_Infected_Precision': 0.0,
                    'S_to_Infected_AUC': 0.5,
                    'S_to_Infected_FPR': 0.0,
                    'S_to_Infected_TPR': 0.0,
                    'S_to_Infected_AUPRC': 0.0,
                    'S_to_Infected_ECE': 0.0
                }
    if (model_type != 'SIR'):
        results.update({
                            'E_to_I_Sensitivity': 0.0,
                            'E_to_I_Specificity': 0.0,
                            'E_to_I_BalAcc': 0.0,
                            'E_to_I_Precision': 0.0,
                            'E_to_I_AUC': 0.5,
                            'E_to_I_FPR': 0.0,
                            'E_to_I_TPR': 0.0,
                            'E_to_I_AUPRC': 0.0,
                            'E_to_I_ECE': 0.0
                        })

    # Computing the metrics
    with h5py.File(h5_file_path, 'r') as f:
        # Selecting a given repetition and epoch
        reps_list = [rep_key for rep_key in f]
        epochs_list = sorted([key for key in f[reps_list[rep_id]]['Preds']['Test']])
        day_grp = f[reps_list[rep_id]]['Preds'][split][epochs_list[epoch_id]]
        
        # Variables to compute metrics
        all_true = []
        all_probs = []
        all_current = []

        # Iterating over the days (concatenating results)
        for day in day_grp.keys():
            all_true.append(np.array(day_grp[day]['true_epi_states']))
            all_probs.append(np.array(day_grp[day]['pred_epi_states_probs']))
            curr = np.array(day_grp[day]['current_epi_state'])
            all_current.append(np.argmax(curr, axis=1) if curr.ndim > 1 else curr)

        # Concatenate across days
        y_true = np.concatenate(all_true)    # [N, H]
        y_probs = np.concatenate(all_probs)  # [N, C, H]
        y_curr = np.concatenate(all_current) # [N]

        # Put predicted probs in the form [N, H, C]
        y_probs = np.transpose(y_probs, (0, 2, 1))

    # S -> Infected Analysis (Capturing Transmission) ---
    # Patients starting at S
    s_mask = (y_curr == S_STATE)
    if (s_mask.any()):
        y_true_s = y_true[s_mask]
        y_prob_s = y_probs[s_mask]
        
        # Ground Truth: Did they ever move to E or I in the window?
        # For SIR, next is 1. For SEIR, next is 1 or 2.
        if (model_type != 'SIR'):
            target_states = [1, 2]
        else:
            target_states = [1]

        # np.isin test if the elements in y_true_s are in target_states, returning a boolean array of the same shape as y_true_s, indicating if each element is in target_states
        gt_became_inf = np.any(np.isin(y_true_s, target_states), axis=1).astype(int)

        # RISK SCORE: The highest probability assigned to any target state at any time in H
        # This represents the model's "conviction" that a transition will occur
        risk_score_s = np.max(np.max(y_prob_s[:, :, target_states], axis=2), axis=1)
        
        # Prediction: Did model predict any of these with confidence > threshold?
        #pred_became_inf = np.any(np.max(y_prob_s[:, :, target_states], axis=2) > threshold, axis=1)
        pred_became_inf = pred_became_inf = (risk_score_s > threshold).astype(int)
        
        # Standard metrics
        #results['S_to_Infected_Sensitivity'] = sensitivity_score(gt_became_inf, pred_became_inf, average='macro')
        results['S_to_Infected_Sensitivity'] = sensitivity_score(gt_became_inf, pred_became_inf, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
        #results['S_to_Infected_Specificity'] = specificity_score(gt_became_inf, pred_became_inf, average='macro') 
        results['S_to_Infected_Specificity'] = specificity_score(gt_became_inf, pred_became_inf, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
        results['S_to_Infected_BalAcc'] = balanced_accuracy_score(gt_became_inf, pred_became_inf, adjusted=True)
         # Here we focus on detecting the positive class i.e. transitions to I state
        #results['S_to_Infected_Precision'] = precision_score(gt_became_inf, pred_became_inf, average='macro')
        results['S_to_Infected_Precision'] = precision_score(gt_became_inf, pred_became_inf, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
        results['S_to_Infected_ECE'] = expected_calibration_error(gt_became_inf, risk_score_s)
        
        # AUC ROC AND PR
        if len(np.unique(gt_became_inf)) > 1:
            results['S_to_Infected_AUC'] = roc_auc_score(gt_became_inf, risk_score_s)
            fpr_s_to_i, tpr_s_to_i, _ = roc_curve(gt_became_inf, risk_score_s)
            results['S_to_Infected_FPR'] = fpr_s_to_i
            results['S_to_Infected_TPR'] = tpr_s_to_i
            results['S_to_Infected_AUPRC'] = average_precision_score(gt_became_inf, risk_score_s)

    # E -> I Analysis (Capturing Progression) ---
    if (model_type != 'SIR'):
        e_mask = (y_curr == E_STATE)
        if (e_mask.any()):
            y_true_e = y_true[e_mask]
            y_prob_e = y_probs[e_mask]
            
            # Ground Truth: Did they progress to I (state 2)?
            gt_became_i = np.any(y_true_e == I_STATE, axis=1)

            # RISK SCORE: Max probability of state I across the horizon H
            risk_score_e = np.max(y_prob_e[:, :, I_STATE], axis=1)
            
            # Prediction: Did model predict state 2 > threshold?
            #pred_became_i = np.any(y_prob_e[:, :, I_STATE] > threshold, axis=1)
            pred_became_i = (risk_score_e > threshold).astype(int)
            
            # Standard metrics
            #results['E_to_I_Sensitivity'] = sensitivity_score(gt_became_i, pred_became_i, average='macro')
            results['E_to_I_Sensitivity'] = sensitivity_score(gt_became_i, pred_became_i, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
            #results['E_to_I_Specificity'] = specificity_score(gt_became_i, pred_became_i, average='macro')
            results['E_to_I_Specificity'] = specificity_score(gt_became_i, pred_became_i, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
            results['E_to_I_BalAcc'] = balanced_accuracy_score(gt_became_i, pred_became_i, adjusted=True)
            #results['E_to_I_Precision'] = precision_score(gt_became_i, pred_became_i, average='macro')
            results['E_to_I_Precision'] = precision_score(gt_became_i, pred_became_i, average='binary') # Here we focus on detecting the positive class i.e. transitions to I state
            results['E_to_I_ECE'] = expected_calibration_error(gt_became_i, risk_score_e)

            # AUC ROC AND PR
            if len(np.unique(gt_became_i)) > 1:
                results['E_to_I_AUC'] = roc_auc_score(gt_became_i, risk_score_e)
                fpr_e_to_i, tpr_e_to_i, _ = roc_curve(gt_became_i, risk_score_e)
                results['E_to_I_FPR'] = fpr_e_to_i
                results['E_to_I_TPR'] = tpr_e_to_i
                results['E_to_I_AUPRC'] = average_precision_score(gt_became_i, risk_score_e)

    return results



def find_optimal_threshold(sweep_results):
    """
        Created with the help of Gemini (validated).
        Finds the optimal thresholds using two methods:
        - Youden’s J Statistic: Maximize (Sensitivity + Specificity - 1).
        - F2-Score: Prioritizing catching infections
        
        sweep_results: list of dicts or rows from your sweep 
        (Threshold, Stability_Rec, Trans_Prec, Trans_Rec)
    """
    # Metrics to determine the optimal threshold
    best_j = -1 # Youden’s J Statistic: Maximize (Sensitivity + Specificity - 1)
    best_f2 = -1 # F2-Score: Prioritizing catching infections
    optimal_t_j = 0
    optimal_t_f2 = 0

    # Iterate over the threshold sweep results
    for res in sweep_results:
        t = res['Threshold']
        prec = res['Trans_Precision']
        rec = res['Trans_Recall']
        spec = res['Stability_Rec'] # Using Stability Recall as a proxy for Specificity

        # Youden's J: Maximize (Sensitivity + Specificity - 1)
        j_stat = rec + spec - 1
        if (j_stat > best_j):
            best_j = j_stat
            optimal_t_j = t

        # 2. F2-Score: Weighted so Recall is 2x as important as Precision
        if ((prec + rec) > 0):
            f2 = (5 * prec * rec) / (4 * prec + rec)
            if (f2 > best_f2):
                best_f2 = f2
                optimal_t_f2 = t

    return {
                "Optimal_J_Threshold": optimal_t_j, 
                "Optimal_F2_Threshold": optimal_t_f2
            }




def plot_pred_epidemic_evolution(h5_results_file, epoch_to_use, states_mapping, use_next_day_only=False, data_split='Test', plot_error_bars=False, fn_prefix="./"):
    # Getting the inverse state mapping
    inv_states_mapping = {v: k for k, v in states_mapping.items()}
    
    # Counting the number of individuals per state in each day
    n_repetitions = len(h5_results_file)
    try:
        n_days = len(h5_results_file["Rep-0"]["Preds"][data_split][f"Epoch-{epoch_to_use}"])
        list_days = sorted([int(day_str.split('-')[-1]) for day_str in h5_results_file['Rep-0']['Preds'][data_split][f'Epoch-{epoch_to_use}']])
    except:
        n_days = len(h5_results_file["Rep-0_Dataset-0"]["Preds"][data_split][f"Epoch-{epoch_to_use}"])
        list_days = sorted([int(day_str.split('-')[-1]) for day_str in h5_results_file['Rep-0_Dataset-0']['Preds'][data_split][f'Epoch-{epoch_to_use}']])

    n_ind_per_state_per_day_true_mean = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_true_std = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_pred_mean = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    n_ind_per_state_per_day_pred_std = {tmp_state: [0 for _ in range(n_days)] for tmp_state in states_mapping}
    true_count_per_day_per_rep = {tmp_state: np.zeros((n_days, n_repetitions)) for tmp_state in states_mapping}
    pred_count_per_day_per_rep = {tmp_state: np.zeros((n_days, n_repetitions)) for tmp_state in states_mapping}

    # Getting the mean and std values per day
    for day in list_days:
        vals_in_day_true = {tmp_state: [0 for _ in range(n_repetitions)] for tmp_state in states_mapping}
        vals_in_day_pred = {tmp_state: [0 for _ in range(n_repetitions)] for tmp_state in states_mapping}
        for rep_id in range(n_repetitions):
            # True values
            try:
                true_states = h5_results_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['true_seir_states'][:]
            except:
                true_states = h5_results_file[f"Rep-0_Dataset-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['true_seir_states'][:]
            # Mask for HUG dataset
            if (-1 in true_states):
                mask_valid_labels = true_states != -1
                true_states = true_states[mask_valid_labels]
            if (use_next_day_only):
                true_states = np.expand_dims(true_states[:, 0], axis=1)
            # Counts
            true_values, true_counts = np.unique(true_states, return_counts=True)
            true_value_counts = {v.item(): c.item() for v, c in zip(true_values, true_counts)}
            for idx_state in true_value_counts:
                vals_in_day_true[inv_states_mapping[idx_state]][rep_id] = true_value_counts[idx_state]
                true_count_per_day_per_rep[inv_states_mapping[idx_state]][day, rep_id] = true_value_counts[idx_state]
        
            # Preds
            try:
                pred_states = np.argmax(h5_results_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'][:, :], axis=1)
            except:
                pred_states = np.argmax(h5_results_file[f"Rep-0_Dataset-{rep_id}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"][f"Day-{day}"]['pred_seir_states_probs'][:, :], axis=1)
            # Mask for HUG dataset
            if (-1 in true_states):
                mask_valid_labels = true_states != -1
                pred_states = pred_states[mask_valid_labels]
            if (use_next_day_only):
                pred_states = np.expand_dims(pred_states[:, 0], axis=1)
            # Counts
            pred_values, pred_counts = np.unique(pred_states, return_counts=True)
            pred_value_counts = {v.item(): c.item() for v, c in zip(pred_values, pred_counts)}
            for idx_state in pred_value_counts:
                vals_in_day_pred[inv_states_mapping[idx_state]][rep_id] = pred_value_counts[idx_state]
                pred_count_per_day_per_rep[inv_states_mapping[idx_state]][day, rep_id] = pred_value_counts[idx_state]
            
        # Get mean and std for the day
        for state in vals_in_day_true:
            n_ind_per_state_per_day_true_mean[state][day - min(list_days)] = np.mean(vals_in_day_true[state])
            n_ind_per_state_per_day_true_std[state][day - min(list_days)] = np.std(vals_in_day_true[state])
            n_ind_per_state_per_day_pred_mean[state][day - min(list_days)] = np.mean(vals_in_day_pred[state])
            n_ind_per_state_per_day_pred_std[state][day - min(list_days)] = np.std(vals_in_day_pred[state])

    # Plot one per figure
    colors = {"S": "b", "E": "orange", "I": "green", "R": "r", "D": "purple", "NS": "brown"}
    days_list = list(range(n_days))
    for state in n_ind_per_state_per_day_true_mean:
        fig = plt.figure()
        plt.plot(days_list, n_ind_per_state_per_day_true_mean[state], c=colors[state], linestyle='-', label=f"True {state}", linewidth=4)
        #plt.errorbar(days_list, n_ind_per_state_per_day_true_mean[state], n_ind_per_state_per_day_true_std[state], c=colors[state], linestyle='-', label=f"True {state}")
        plt.plot(days_list, n_ind_per_state_per_day_pred_mean[state], c=colors[state], linestyle='--', label=f"Predicted {state}", linewidth=4)
        #plt.errorbar(days_list, n_ind_per_state_per_day_pred_mean[state], n_ind_per_state_per_day_pred_std[state], c=colors[state], linestyle='--', label=f"Predicted {state}")
        plt.xlabel("Day")
        plt.ylabel("N° of individuals")
        plt.legend()
        if (PLOT_FIGURES):
            plt.show()
        # Save figures
        fig.savefig(f"{fn_prefix}{state.upper()}_True_vs_Pred_WholeModel.pdf", bbox_inches="tight")
        fig.savefig(f"{fn_prefix}{state.upper()}_True_vs_Pred_WholeModel.png", dpi=300, bbox_inches="tight")

    # Plot all in the same figure
    fig = plt.figure()
    colors = {"S": "b", "E": "orange", "I": "green", "R": "r", "D": "purple", "NS": "brown"}
    days_list = list(range(n_days))
    mean_metrics_per_state = {"MSE": {}, "PCC": {}, "CCC": {}}
    for state in n_ind_per_state_per_day_true_mean:
        mse_per_rep = []
        pcc_per_rep = []
        ccc_per_rep = []
        for rep_ID in range(pred_count_per_day_per_rep[state].shape[1]):
            # MSE
            mse = mean_squared_error(true_count_per_day_per_rep[state][:, rep_ID], pred_count_per_day_per_rep[state][:, rep_ID])
            mse_per_rep.append(mse)
            # Pearson Correlation Coefficient
            pcc = stats.pearsonr(true_count_per_day_per_rep[state][:, rep_ID], pred_count_per_day_per_rep[state][:, rep_ID])
            if (np.isnan(pcc.statistic)):
                pcc = 0
                #pcc = -1
            else:
                pcc = pcc.statistic
            pcc_per_rep.append(pcc)
            # Compute Concordance Correlation Coefficient (CCC)
            ccc_function = ConcordanceCorrCoef()
            ccc = ccc_function(torch.from_numpy(true_count_per_day_per_rep[state][:, rep_ID]), torch.from_numpy(pred_count_per_day_per_rep[state][:, rep_ID]))
            if (np.isnan(ccc)):
                ccc = 0
                #ccc = -1
            ccc_per_rep.append(float(ccc))

        # Compute the mean and std of the metrics
        mse_state_mean = np.mean(mse_per_rep)
        mse_state_std = np.std(mse_per_rep)
        pcc_state_mean = np.mean(pcc_per_rep)
        pcc_state_std = np.std(pcc_per_rep)
        ccc_state_mean = np.mean(ccc_per_rep)
        ccc_state_std = np.std(ccc_per_rep)
        if (state not in mean_metrics_per_state["MSE"]):
            mean_metrics_per_state["MSE"][state] = {"Mean": None, "Std": None}
            mean_metrics_per_state["PCC"][state] = {"Mean": None, "Std": None}
            mean_metrics_per_state["CCC"][state] = {"Mean": None, "Std": None}
        mean_metrics_per_state['MSE'][state]['Mean'] = mse_state_mean
        mean_metrics_per_state['MSE'][state]['Std'] = mse_state_std
        mean_metrics_per_state['PCC'][state]['Mean'] = pcc_state_mean
        mean_metrics_per_state['PCC'][state]['Std'] = pcc_state_std
        mean_metrics_per_state['CCC'][state]['Mean'] = ccc_state_mean
        mean_metrics_per_state['CCC'][state]['Std'] = ccc_state_std
        
        # Plot evolution
        if (plot_error_bars):
            plt.errorbar(days_list, n_ind_per_state_per_day_true_mean[state], n_ind_per_state_per_day_true_std[state], c=colors[state], linestyle='-', label=f"True {state}")
            plt.errorbar(days_list, n_ind_per_state_per_day_pred_mean[state], n_ind_per_state_per_day_pred_std[state], c=colors[state], linestyle='--', label=f"Predicted {state}")
        else:
            plt.plot(days_list, n_ind_per_state_per_day_true_mean[state], c=colors[state], linestyle='-', label=f"True {state}", linewidth=4)
            plt.plot(days_list, n_ind_per_state_per_day_pred_mean[state], c=colors[state], linestyle='--', label=f"Predicted {state}", linewidth=4)
        plt.xlabel("Days")
        plt.ylabel("Number of individuals")
    plt.legend()
    if (PLOT_FIGURES):
        plt.show()

    # Print mean metrics
    for metric_type in mean_metrics_per_state:
        print(f"\n\n\n========================================>{metric_type.upper()}<========================================")
        val_for_latex = {'S': '-', 'E': '-', 'I': '-', 'R': '-', 'D': '-', 'NS': '-'}
        for state in mean_metrics_per_state[metric_type]:
            print(f"\n\t======> The {metric_type.upper()} between the true and predicted total counts for state {state} is {round(mean_metrics_per_state[metric_type][state]['Mean'], 2)} \pm {round(mean_metrics_per_state[metric_type][state]['Std'], 2)}\n\n")
            val_for_latex[state] = f"${round(mean_metrics_per_state[metric_type][state]['Mean'], 2)} \pm {round(mean_metrics_per_state[metric_type][state]['Std'], 2)}$"

        # For LaTeX
        print(f"LATEX: {val_for_latex['S']} & {val_for_latex['E']} & {val_for_latex['I']} & {val_for_latex['R']} & {val_for_latex['D']} & {val_for_latex['NS']}\\\\ \n\n")



def main():
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_folder', required=True, help="Path to the folder containing the results of the experiment", type=str)
    ap.add_argument('--plot_loss', help="Use if want to plot the loss curves", action="store_true")
    ap.add_argument('--plot_metrics_evolution', help="Use if want to plot the metrics evolutions over the epochs", action="store_true")
    ap.add_argument('--force_metrics_computation', help="Use it if want to force the metrics comuptation even if they were already previously computed", action='store_true')
    ap.add_argument('--analyze_multiclass_failure', help="Use if want to compute and plot confusion matrices for debug", action="store_true")
    ap.add_argument('--debug', help="Use if want to the debuffer at some specific points", action="store_true")
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    force_metrics_computation = args['force_metrics_computation']
    analyze_multiclass_failure = args['analyze_multiclass_failure']
    plot_loss = args['plot_loss']
    plot_metrics_evolution = args['plot_metrics_evolution']
    debug = args['debug']

    #======================================================================#
    #=====================Parameters of the experiment=====================#
    #======================================================================#
    # Parameters of the experiment
    try:
        parameters_file = results_folder + f"/params_exp/params_0.pth"
        with open(parameters_file, 'rb') as pf:
            params_exp = pickle.load(pf)
    except:
        parameters_file = results_folder + f"/params_exp/params_0.yaml"
        with open(parameters_file, 'r') as file:
            params_exp = yaml.safe_load(file)

    #======================================================================#
    #============================Compute metrics============================#
    #======================================================================#
    # Path handling
    h5_path = os.path.join(results_folder, 'metrics', 'final_results_all_repetitions_0.hdf5')
    if not os.path.exists(h5_path):
        print(f"Error: File not found at {h5_path}")
        return

    # Folders for the results (to use to get the metrics if exist, or to create if not)
    # Dir for plots
    output_dir = os.path.join(results_folder, 'analysis_plots')
    create_output_dir_files = False
    if (os.path.exists(output_dir)):
        if (not os.listdir(output_dir)): # Empty dir
            create_output_dir_files = True
    else:
        create_output_dir_files = True
    # Dir for metrics
    perf_dir = os.path.join(results_folder, 'perf_dir')
    create_perf_dir_files = False
    if (os.path.exists(perf_dir)):
        if (not os.listdir(perf_dir)): # Empty dir
            create_perf_dir_files = True
    else:
        create_perf_dir_files = True
    

    if (create_output_dir_files) or (create_perf_dir_files) or (force_metrics_computation):
        #======================================================================#
        # Create folders to store the results
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(perf_dir, exist_ok=True)

        #======================================================================#
        # Load data
        print(f"Loading results from {h5_path}...")
        
        # Get main metrics
        loss_storage, metrics_storage, n_classes = get_metrics_per_rep(h5_path, params_exp, analyze_multiclass_failure)

        #======================================================================#
        # Plot losses
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY (Last Epoch Average)")
        print("="*50)
        
        if (plot_loss):
            for loss_type, split_data in loss_storage.items():
                plt.figure(figsize=(8, 6))
                for split, values_list in split_data.items():
                    # values_list is list of arrays (one per rep). 
                    # Find min length to average (in case of different epoch counts)
                    min_len = min(len(v) for v in values_list)
                    truncated_values = [v[:min_len] for v in values_list]
                    
                    means = np.mean(truncated_values, axis=0)
                    stds = np.std(truncated_values, axis=0)
                    epochs = range(1, len(means) + 1)
                    
                    plt.plot(epochs, means, label=f"{split} Mean")
                    plt.fill_between(epochs, means - stds, means + stds, alpha=0.2)
                    
                    print(f"[{loss_type}] {split} Final: {means[-1]:.4f} ± {stds[-1]:.4f}")

                plt.title(f"{loss_type} over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"Loss_{loss_type}.png")) # Show AFTER saving image, to avoid saving empty images
                plt.show()
                plt.close()


        #======================================================================#
        # Plot Main Metrics
        # Random model
        print("\n\n\n========================================>RANDOM MODEL<========================================")
        for metric_name, split_data in metrics_storage.items():
            if ("random" in metric_name.lower()):
                print("\n\n")
                if ('perclass' not in metric_name.lower()):
                    plot_metric_curve(metric_name, split_data, output_dir, show_plot=plot_metrics_evolution)
        # Trained model
        print("\n\n\n========================================>TRAINED MODEL<========================================")
        sens = {'Mean': None, 'Std': None}
        spec = {'Mean': None, 'Std': None}
        bal_acc = {'Mean': None, 'Std': None}
        auc = {'Mean': None, 'Std': None}
        ece = {'Mean': None, 'Std': None}
        if (params_exp['predict_state_transitions']):
            sens_transitions = {'Mean': None, 'Std': None}
            spec_transitions = {'Mean': None, 'Std': None}
            bal_acc_transitions = {'Mean': None, 'Std': None}
            auc_transitions = {'Mean': None, 'Std': None}
            ece_transitions = {'Mean': None, 'Std': None}

        for metric_name, split_data in metrics_storage.items():
            if ("random" not in metric_name.lower()):
                print("\n\n")
                if ('perclass' not in metric_name.lower()):
                    metrics_statistics = plot_metric_curve(metric_name, split_data, output_dir, show_plot=plot_metrics_evolution)
                    if (metric_name == 'Sensitivity'):
                        sens = metrics_statistics['Test']
                    if (metric_name == 'Specificity'):
                        spec = metrics_statistics['Test']
                    if (metric_name == 'BalancedAccuracy'):
                        bal_acc = metrics_statistics['Test']
                    if (metric_name == 'AUC'):
                        auc = metrics_statistics['Test']
                    if (metric_name == 'ECE'):
                        ece = metrics_statistics['Test']
                    if (params_exp['predict_state_transitions']):
                        if (metric_name == 'SensitivityTransitions'):
                            sens_transitions = metrics_statistics['Test']
                        if (metric_name == 'SpecificityTransitions'):
                            spec_transitions = metrics_statistics['Test']
                        if (metric_name == 'BalancedAccuracyTransitions'):
                            bal_acc_transitions = metrics_statistics['Test']
                        if (metric_name == 'AUCTransitions'):
                            auc_transitions = metrics_statistics['Test']
                        if (metric_name == 'ECETransitions'):
                            ece_transitions = metrics_statistics['Test']

        print(f"\nAll plots saved to: {output_dir}")


        # # C. Plot AUC per class (one vs rest)
        # # Getting the repetition with the highest AUC to plot ROC
        # epochs_list = list(metrics_storage['AUC']['Val'].keys())
        # rep_max_val_aux = np.argmax(metrics_storage['AUC']['Val'][epochs_list[-1]])
        # # Plot
        # print("\n\n\n\n=========> Plot ROC curves PER CLASS <=========\n\n\n\n")
        # plt.figure(figsize=(8, 6))
        # classes = list(range(n_classes))
        # for i in range(n_classes):
        #     plt.plot(metrics_storage['PerClassFPR']['Test'][epochs_list[-1]][rep_max_val_aux][i], metrics_storage['PerClassTPR']['Test'][epochs_list[-1]][rep_max_val_aux][i], lw=2,
        #             label=f"ROC curve of class {classes[i]} (area = {metrics_storage['PerClassAUC']['Test'][epochs_list[-1]][rep_max_val_aux][i]:0.2f})")
        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Multi-class ROC (One-vs-Rest)')
        # plt.legend(loc="lower right")
        # plt.show()
        # print("\n\n\n\n=========> Plot PR curves PER CLASS <=========\n\n\n\n")
        # plt.figure(figsize=(8, 6))
        # colors = {0: 'b', 1: 'r', 2: 'g', 3: 'm', 4: 'tab:brown', 5: 'tab:orange'}
        # classes = list(range(n_classes))
        # for i in range(n_classes):
        #     color = colors[i]
        #     plt.plot(metrics_storage['PerClassRecall']['Test'][epochs_list[-1]][rep_max_val_aux][i], metrics_storage['PerClassPrecision']['Test'][epochs_list[-1]][rep_max_val_aux][i], lw=2,
        #             label=f"PR curve of class {classes[i]} (area = {metrics_storage['PerClassAUCPR']['Test'][epochs_list[-1]][rep_max_val_aux][i]:0.2f})", c=color)
        #     baseline = metrics_storage['PerClassPRBaseline']['Test'][epochs_list[-1]][rep_max_val_aux][i]
        #     plt.axhline(y=baseline, linestyle='--', label=f'Baseline ({baseline:0.3f}) for class {classes[i]}', c=color)
        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlabel('Recall (Sensitivity)')
        # plt.ylabel('Precision')
        # plt.title('Multi-class ROC (One-vs-Rest)')
        # plt.legend(loc="lower right")
        # plt.show()


        # C. Plot AUC per class (one vs rest)
        # Getting the repetition with the highest AUC to plot ROC
        epochs_list = list(metrics_storage['AUC']['Train'].keys())
        n_reps = len(metrics_storage['AUC']['Train'][epochs_list[-1]])
        classes = list(range(n_classes))
        colors = {0: 'b', 1: 'r', 2: 'g', 3: 'm', 4: 'tab:brown', 5: 'tab:orange'}
        # AUC ROC: Specific state prediction
        plt.figure(figsize=(8, 6))
        for i_class in range(n_classes):
            # Interpolation points for averaging curves
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = []
            for tmp_rep_ID in range(n_reps):
                # ROC Data
                fpr = metrics_storage['PerClassFPR']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
                tpr = metrics_storage['PerClassTPR']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(metrics_storage['PerClassAUC']['Test'][epochs_list[-1]][tmp_rep_ID][i_class])

            # Plot ROC with Shading
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(tprs, axis=0)

            # Create curves for current class
            plt.plot(mean_fpr, mean_tpr, label=f'Class {i_class}, Mean ROC (AUC = {round(np.mean(aucs)*100, 2):23f} ± {round(np.std(aucs)*100, 2):.2f})', lw=2, color=colors[i_class])
            plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color=colors[i_class], alpha=0.2)
            plt.plot([0, 1], [0, 1], 'r--', lw=2)
            plt.title(f'Specific States Prediction AUC')
            plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AUC"))
        if (plot_metrics_evolution):
            plt.show()
        # AUC ROC: State transitions prediction
        if (params_exp['predict_state_transitions']):
            n_transitions_classes = len(metrics_storage['PerClassTPRTransitions']['Test'][epochs_list[-1]][0])
            transitions_classes = list(range(n_transitions_classes))
            colors = {0: 'b', 1: 'r', 2: 'g', 3: 'm', 4: 'tab:brown', 5: 'tab:orange'}
            plt.figure(figsize=(8, 6))
            for i_class in range(n_transitions_classes):
                # Interpolation points for averaging curves
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                aucs = []
                for tmp_rep_ID in range(n_reps):
                    # ROC Data
                    fpr = metrics_storage['PerClassFPRTransitions']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
                    tpr = metrics_storage['PerClassTPRTransitions']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(metrics_storage['PerClassAUCTransitions']['Test'][epochs_list[-1]][tmp_rep_ID][i_class])

                # Plot ROC with Shading
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                std_tpr = np.std(tprs, axis=0)

                # Save in CSV
                fpr_tpr_for_csv = pd.DataFrame(
                                                {
                                                    'FPR': mean_fpr,
                                                    'TPR': mean_tpr,
                                                    'TPR_Std': std_tpr,
                                                }
                                            )
                fpr_tpr_for_csv.to_csv(output_dir + f'/FPR_TPR_for_AUC_Class-{i_class}.csv', index=False)

                # Create curves for current class
                plt.plot(mean_fpr, mean_tpr, label=f'Class {i_class}, Mean Transitions ROC (AUC = {round(np.mean(aucs)*100, 2):23f} ± {round(np.std(aucs)*100, 2):.2f})', lw=2, color=colors[i_class])
                plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color=colors[i_class], alpha=0.2)
                plt.plot([0, 1], [0, 1], 'r--', lw=2)
                plt.title(f'Transitions Prediction AUC')
                plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"AUC_Transitions"))
            if (plot_metrics_evolution):
                plt.show()

        # # AUC PR
        # plt.figure(figsize=(8, 6))
        # for i_class in range(n_classes):
        #     # Interpolation points for averaging curves
        #     mean_recall = np.linspace(0, 1, 100)
        #     precisions = []
        #     auprcs = []
        #     baselines = []
        #     for tmp_rep_ID in range(n_reps):
        #         # PR Data
        #         prec = metrics_storage['PerClassPrecision']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
        #         rec = metrics_storage['PerClassRecall']['Test'][epochs_list[-1]][tmp_rep_ID][i_class]
        #         # Interpolate precision over the mean_recall axis
        #         # (We reverse because rec is decreasing in sklearn)
        #         precisions.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
        #         auprcs.append(metrics_storage['PerClassAUCPR']['Test'][epochs_list[-1]][tmp_rep_ID][i_class])
        #         baselines.append(metrics_storage['PerClassPRBaseline']['Test'][epochs_list[-1]][tmp_rep_ID][i_class])

        #     # Plot PR with Shading
        #     mean_prec = np.mean(precisions, axis=0)
        #     std_prec = np.std(precisions, axis=0)
            
        #     plt.plot(mean_recall, mean_prec, label=f'Class {i_class}, Mean AUPRC (Area = {round(np.mean(auprcs)*100, 2):.2f} ± {round(np.std(auprcs)*100, 2):.2f})', lw=2, color=colors[i_class])
        #     plt.fill_between(mean_recall, np.maximum(mean_prec - std_prec, 0), np.minimum(mean_prec + std_prec, 1), color=colors[i_class], alpha=0.2)
            
        #     # Plot baseline (Prevalence)
        #     mean_prevalence = np.mean(baselines)
        #     std_prevalence = np.std(baselines)
        #     # Plot baseline with a thin shaded region if the variance is high
        #     plt.axhline(y=mean_prevalence, color=colors[i_class], linestyle='--', 
        #             label=f'Mean Baseline ({mean_prevalence:.3f} ± {std_prevalence:.3f})')
        #     if (std_prevalence > 1e-4): # Only shade if there's actual variation
        #         plt.axhspan(mean_prevalence - std_prevalence, mean_prevalence + std_prevalence, 
        #                     color=colors[i_class], alpha=0.1)

        #     plt.title(f'Mean PR Curve: {i_class}')
        #     plt.legend(loc="upper right")

        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f"AUPR_Per_Class"))
        # plt.show()
        

        #======================================================================#
        # Post-Hoc Threshold Sweep
        _ = threshold_sweep(
                                            h5_file_path=h5_path,
                                            rep_id=0, 
                                            split='Test',
                                            epoch_id=-1,
                                            print_report=True
                                        )
        

        #======================================================================#
        # Calibration curves for transitions
        plot_calibration_for_transitions(
                                            h5_file_path=h5_path,
                                            rep_id=0, 
                                            save_path=output_dir,
                                            split='Test', 
                                            epoch_id=-1,
                                            show_plot=plot_metrics_evolution
                                        )

        #======================================================================#
        # Compute the best thresholds BASED ON THE VALIDATION SET
        # Get number of repetitions
        with h5py.File(h5_path, 'r') as f:
            n_reps = len(list(f.keys()))
        # Iterating over the repetitions
        results_onset_infection_per_rep = {}
        for rep_id in range(n_reps):
            # Get threshold sweep for the validation set
            try:
                sweep_results_val = threshold_sweep(
                                                    h5_file_path=h5_path,
                                                    rep_id=rep_id, 
                                                    split='Val',
                                                    epoch_id=-1,
                                                    print_report=False
                                                )
            except:
                print("\n\n===>WARNING: THERE IS NO VALIDATION DATASET, computing best thresholds on the training set !!!")
                sweep_results_val = threshold_sweep(
                                                    h5_file_path=h5_path,
                                                    rep_id=rep_id, 
                                                    split='Train',
                                                    epoch_id=-1,
                                                    print_report=False
                                                )
            
            # Get optimal thresholds
            opt_thresh_dict = find_optimal_threshold(sweep_results_val)
        
        
            # Evaluate capacity of the model to predict infection transitions (S to I for SIR model, and E to I for SEIR and SEIRD-NS models)
            if (params_exp["dataset_name"].lower() == 'hug'):
                model_type = 'SIR'
            elif (params_exp["dataset_name"].lower() == 'sociopatterns'):
                model_type = 'SEIR'
            elif (params_exp["dataset_name"].lower() == 'murcia'):
                model_type = 'SEIRD-NS'
            results_onset_infection = evaluate_infection_onset(
                                                                    h5_file_path=h5_path,
                                                                    rep_id=rep_id,
                                                                    model_type=model_type,
                                                                    split='Test',
                                                                    epoch_id=-1,
                                                                    #threshold=opt_thresh_dict['Optimal_J_Threshold'] # best balance
                                                                    threshold=opt_thresh_dict['Optimal_F2_Threshold'] # prioritizing catching infections
                                                                )
            #print(f"\n\n==========> Results of the progression from S (or E) to I state for repetition {rep_id}: \n")
            #print(results_onset_infection)
            #print("\n\n\n\n")

            # Adding it to the dict of repetitions
            for metric_name in results_onset_infection:
                if (metric_name not in results_onset_infection_per_rep):
                    results_onset_infection_per_rep[metric_name] = []
                results_onset_infection_per_rep[metric_name].append(results_onset_infection[metric_name])
        # Plot AUC for Infection Entry Prediction
        if (model_type == 'SIR'):
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

        colors = {0: 'b', 1: 'r', 2: 'g', 3: 'm', 4: 'tab:brown', 5: 'tab:orange'}
        # Interpolation points for averaging curves
        mean_fpr_s_to_i = np.linspace(0, 1, 100)
        tprs_s_to_i = []
        aucs_s_to_i = []
        mean_fpr_e_to_i = np.linspace(0, 1, 100)
        tprs_e_to_i = []
        aucs_e_to_i = []
        for tmp_rep_ID in range(n_reps):
            # ROC Data
            fpr_s_to_i = results_onset_infection_per_rep['S_to_Infected_FPR'][tmp_rep_ID]
            tpr_s_to_i = results_onset_infection_per_rep['S_to_Infected_TPR'][tmp_rep_ID]
            tprs_s_to_i.append(np.interp(mean_fpr_s_to_i, fpr_s_to_i, tpr_s_to_i))
            tprs_s_to_i[-1][0] = 0.0
            aucs_s_to_i.append(results_onset_infection_per_rep['S_to_Infected_AUC'][tmp_rep_ID])
            if (model_type != 'SIR'):
                fpr_e_to_i = results_onset_infection_per_rep['E_to_I_FPR'][tmp_rep_ID]
                tpr_e_to_i = results_onset_infection_per_rep['E_to_I_TPR'][tmp_rep_ID]
                tprs_e_to_i.append(np.interp(mean_fpr_e_to_i, fpr_e_to_i, tpr_e_to_i))
                tprs_e_to_i[-1][0] = 0.0
                aucs_e_to_i.append(results_onset_infection_per_rep['E_to_I_AUC'][tmp_rep_ID])


        # Plot ROC with Shading
        mean_tpr_s_to_i = np.mean(tprs_s_to_i, axis=0)
        mean_tpr_s_to_i[-1] = 1.0
        std_tpr_s_to_i = np.std(tprs_s_to_i, axis=0)
        if (model_type != 'SIR'):
            mean_tpr_e_to_i = np.mean(tprs_e_to_i, axis=0)
            mean_tpr_e_to_i[-1] = 1.0
            std_tpr_e_to_i = np.std(tprs_e_to_i, axis=0)

        # Create curves for current class
        ax1.plot(mean_fpr_s_to_i, mean_tpr_s_to_i, label=f'Class {i_class}, Mean ROC (AUC = {round(np.mean(aucs_s_to_i)*100, 2):23f} ± {round(np.std(aucs_s_to_i)*100, 2):.2f})', lw=2, color='r')
        ax1.fill_between(mean_fpr_s_to_i, np.maximum(mean_tpr_s_to_i - std_tpr_s_to_i, 0), np.minimum(mean_tpr_s_to_i + std_tpr_s_to_i, 1), color='r', alpha=0.2)
        ax1.plot([0, 1], [0, 1], 'r--', lw=2)
        ax1.set_title(f'Mean ROC Curve S to I prediction')
        ax1.legend(loc="lower right")
        if (model_type != 'SIR'):
            ax2.plot(mean_fpr_e_to_i, mean_tpr_e_to_i, label=f'Class {i_class}, Mean ROC (AUC = {round(np.mean(aucs_e_to_i)*100, 2):23f} ± {round(np.std(aucs_e_to_i)*100, 2):.2f})', lw=2, color='b')
            ax2.fill_between(mean_fpr_e_to_i, np.maximum(mean_tpr_e_to_i - std_tpr_e_to_i, 0), np.minimum(mean_tpr_e_to_i + std_tpr_e_to_i, 1), color='b', alpha=0.2)
            ax2.plot([0, 1], [0, 1], 'r--', lw=2)
            ax2.set_title('Mean ROC Curve E to I prediction')
            ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AUC_S_to_I_Transitions_From_State_Prediction"))
        if (plot_metrics_evolution):
            plt.show()

        
        #======================================================================#
        # Print mean metrics OF ONSET IFECTION USING THE PREDICTED STATES
        print("\n\n==========> Results ONSET INFECTION <==========")
        for metric_name in results_onset_infection:
            if ('fpr' not in metric_name.lower()) and ('tpr' not in metric_name.lower()):
                print(f"\t===> {metric_name}: {round(np.mean(results_onset_infection_per_rep[metric_name])*100, 2):.2f} \pm {round(np.std(results_onset_infection_per_rep[metric_name])*100, 2):.2f}")
        print("\n\n")
    else:
        #======================================================================#
        # Opening images
        if (not os.path.exists(output_dir)):
            raise RuntimeError(f"Error: Folder '{output_dir}' not found.")

        # Filter for PNG files
        files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]

        if (not files):
            print("No PNG images found in the folder.")
            return

        for filename in files:
            plot_image = False
            if (plot_loss) and ('loss' in filename.lower()):
                plot_image = True
            if (plot_metrics_evolution) and ('loss' not in filename.lower()):
                plot_image = True 
            if (plot_image):
                img_path = os.path.join(output_dir, filename)
                
                try:
                    # Open the image
                    img = Image.open(img_path)
                    
                    # Create a new figure for each image
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.title(f"Viewing: {filename}")
                    plt.axis('off')  # Removes the X/Y axis scales
                    
                    print(f"Showing {filename}... Close the window to see the next one.")
                    plt.show()  # Execution stops here until the window is closed
                    
                except Exception as e:
                    print(f"Could not open {filename}: {e}")
    
    #======================================================================#
    #======================================================================#
    # Plot LaTeX ready results
    # Print STATE PREDICTION results in a LaTeX compatible form
    if (os.path.exists(perf_dir + '/states_preds.txt')):
        with open(perf_dir + '/states_preds.txt', "r", encoding="utf-8") as file:
            latex_state_pred_results_str = file.read()
        print(f"\n\n\n\n==========> STATE Prediction Performance: {latex_state_pred_results_str}\n\n\n\n")
    else:
        latex_state_pred_results_str = f" ${round(sens['Mean'], 2):.2f} \pm {round(sens['Std'], 2):.2f}$ & ${round(spec['Mean'], 2):.2f} \pm {round(spec['Std'], 2):.2f}$ & ${round(bal_acc['Mean'], 2):.2f} \pm {round(bal_acc['Std'], 2):.2f}$ & ${round(auc['Mean'], 2):.2f} \pm {round(auc['Std'], 2):.2f}$ & ${round(ece['Mean'], 2):.2f} \pm {round(ece['Std'], 2):.2f}$\\\\ "
        print(f"\n\n\n\n==========> STATE Prediction Performance: {latex_state_pred_results_str}\n\n\n\n")
        with open(perf_dir + '/states_preds.txt', "w") as file:
            file.write(latex_state_pred_results_str)

    # Print TRANSITIONS PREDICTION results in a LaTeX compatible form
    if (os.path.exists(perf_dir + '/transitions_preds.txt')):
        with open(perf_dir + '/transitions_preds.txt', "r", encoding="utf-8") as file:
            latex_trans_pred_results_str = file.read()
            print(f"\n\n\n\n==========> TRANSITIONS Prediction Performance: {latex_trans_pred_results_str}\n\n\n\n")
    else:
        if (params_exp['predict_state_transitions']):
            latex_trans_pred_results_str = f" ${round(sens_transitions['Mean'], 2):.2f} \pm {round(sens_transitions['Std'], 2):.2f}$ & ${round(spec_transitions['Mean'], 2):.2f} \pm {round(spec_transitions['Std'], 2):.2f}$ & ${round(bal_acc_transitions['Mean'], 2):.2f} \pm {round(bal_acc_transitions['Std'], 2):.2f}$ & ${round(auc_transitions['Mean'], 2):.2f} \pm {round(auc_transitions['Std'], 2):.2f}$ & ${round(ece_transitions['Mean'], 2):.2f} \pm {round(ece_transitions['Std'], 2):.2f}$\\\\ "
            print(f"\n\n\n\n==========> TRANSITIONS Prediction Performance: {latex_trans_pred_results_str}\n\n\n\n")
            with open(perf_dir + '/transitions_preds.txt', "w") as file:
                file.write(latex_trans_pred_results_str)

    # Print STATE TRANSITIONS FROM STATE PREDICTION results in a LaTeX compatible form
    if (os.path.exists(perf_dir + '/trans_from_pred_states.txt')):
        with open(perf_dir + '/trans_from_pred_states.txt', "r", encoding="utf-8") as file:
            latex_trans_from_pred_states_results_str = file.read()
            print(f"\n\n\n\n==========> Transitions FROM State Prediction Performance (FROM S TO I): {latex_trans_from_pred_states_results_str}\n\n\n\n")
    else:
        sens_trans_from_pred_states = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["S_to_Infected_Sensitivity"]]
        sens_trans_from_pred_states = {'Mean': np.mean(sens_trans_from_pred_states), 'Std': np.std(sens_trans_from_pred_states)}

        spec_trans_from_pred_states = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["S_to_Infected_Specificity"]]
        spec_trans_from_pred_states = {'Mean': np.mean(spec_trans_from_pred_states), 'Std': np.std(spec_trans_from_pred_states)}

        bal_acc_trans_from_pred_states = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["S_to_Infected_BalAcc"]]
        bal_acc_trans_from_pred_states = {'Mean': np.mean(bal_acc_trans_from_pred_states), 'Std': np.std(bal_acc_trans_from_pred_states)}

        auc_trans_from_pred_states = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["S_to_Infected_AUC"]]
        auc_trans_from_pred_states = {'Mean': np.mean(auc_trans_from_pred_states), 'Std': np.std(auc_trans_from_pred_states)}

        ece_trans_from_pred_states = results_onset_infection_per_rep["S_to_Infected_ECE"]
        ece_trans_from_pred_states = {'Mean': np.mean(ece_trans_from_pred_states), 'Std': np.std(ece_trans_from_pred_states)}

        latex_trans_from_pred_states_results_str = f" ${round(sens_trans_from_pred_states['Mean'], 2):.2f} \pm {round(sens_trans_from_pred_states['Std'], 2):.2f}$ & ${round(spec_trans_from_pred_states['Mean'], 2):.2f} \pm {round(spec_trans_from_pred_states['Std'], 2):.2f}$ & ${round(bal_acc_trans_from_pred_states['Mean'], 2):.2f} \pm {round(bal_acc_trans_from_pred_states['Std'], 2):.2f}$ & ${round(auc_trans_from_pred_states['Mean'], 2):.2f} \pm {round(auc_trans_from_pred_states['Std'], 2):.2f}$ & ${round(ece_trans_from_pred_states['Mean'], 2):.2f} \pm {round(ece_trans_from_pred_states['Std'], 2):.2f}$\\\\ "
        print(f"\n\n\n\n==========> Transitions FROM State Prediction Performance (FROM S TO I): {latex_trans_from_pred_states_results_str}\n\n\n\n")
        if (model_type != 'SIR'):
            sens_trans_from_pred_states_e_to_i = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["E_to_I_Sensitivity"]]
            sens_trans_from_pred_states_e_to_i = {'Mean': np.mean(sens_trans_from_pred_states_e_to_i), 'Std': np.std(sens_trans_from_pred_states_e_to_i)}

            spec_trans_from_pred_states_e_to_i = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["E_to_I_Specificity"]]
            spec_trans_from_pred_states_e_to_i = {'Mean': np.mean(spec_trans_from_pred_states_e_to_i), 'Std': np.std(spec_trans_from_pred_states_e_to_i)}

            bal_acc_trans_from_pred_states_e_to_i = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["E_to_I_BalAcc"]]
            bal_acc_trans_from_pred_states_e_to_i = {'Mean': np.mean(bal_acc_trans_from_pred_states_e_to_i), 'Std': np.std(bal_acc_trans_from_pred_states_e_to_i)}

            auc_trans_from_pred_states_e_to_i = [tmp_val*100 for tmp_val in results_onset_infection_per_rep["E_to_I_AUC"]]
            auc_trans_from_pred_states_e_to_i = {'Mean': np.mean(auc_trans_from_pred_states_e_to_i), 'Std': np.std(auc_trans_from_pred_states_e_to_i)}

            ece_trans_from_pred_states_e_to_i = results_onset_infection_per_rep["E_to_I_ECE"]
            ece_trans_from_pred_states_e_to_i = {'Mean': np.mean(ece_trans_from_pred_states_e_to_i), 'Std': np.std(ece_trans_from_pred_states_e_to_i)}

            latex_trans_from_pred_states_e_to_i_results_str = f" ${round(sens_trans_from_pred_states_e_to_i['Mean'], 2):.2f} \pm {round(sens_trans_from_pred_states_e_to_i['Std'], 2):.2f}$ & ${round(spec_trans_from_pred_states_e_to_i['Mean'], 2):.2f} \pm {round(spec_trans_from_pred_states_e_to_i['Std'], 2):.2f}$ & ${round(bal_acc_trans_from_pred_states_e_to_i['Mean'], 2):.2f} \pm {round(bal_acc_trans_from_pred_states_e_to_i['Std'], 2):.2f}$ & ${round(auc_trans_from_pred_states_e_to_i['Mean'], 2):.2f} \pm {round(auc_trans_from_pred_states_e_to_i['Std'], 2):.2f}$ & ${round(ece_trans_from_pred_states_e_to_i['Mean'], 2):.2f} \pm {round(ece_trans_from_pred_states_e_to_i['Std'], 2):.2f}$\\\\ "
            print(f"\n\n\n\n==========> Transitions FROM State Prediction Performance (FROM E TO I): {latex_trans_from_pred_states_e_to_i_results_str}\n\n\n\n")
            latex_trans_from_pred_states_results_str = "\n\n S to I:\n" + latex_trans_from_pred_states_results_str + '\n\n\n\n E to I: \n' + latex_trans_from_pred_states_e_to_i_results_str
        with open(perf_dir + '/trans_from_pred_states.txt', "w") as file:
            file.write(latex_trans_from_pred_states_results_str)


if __name__ == "__main__":
    main()