import os
import h5py
import yaml
import pickle
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

# Internal imports
from src.Utils.plot_metrics import get_metrics_per_rep,\
                                   plot_metric_curve,\
                                   evaluate_infection_onset,\
                                   threshold_sweep,\
                                   find_optimal_threshold
                                  

# Suppress undefined metric warnings (common in early epochs or rare classes)
warnings.filterwarnings('ignore') 


def main():
    #======================================================================#
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--main_results_folder', required=True, help="Path to the folder containing the subfolders with results of the experiment", type=str)
    ap.add_argument('--force_metrics_computation', help="Use it if want to force the metrics comuptation even if they were already previously computed", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    main_results_folder = args['main_results_folder']
    force_metrics_computation = args['force_metrics_computation']

    #======================================================================#
    #======================================================================#
    #===============Variables to iterate over the experiments===============#
    #======================================================================#
    #======================================================================#
    pred_horizons = [1, 3, 7]
    #pred_horizons = [7]
    models = ['GAT', 'GCN', 'GraphSAGE', 'TGN', 'STM-GNN']
    epi_informed_methods = ["None", "AutoDiff", "ODE-NSP"]

    #======================================================================#
    #======================================================================#
    #========================Getting results folders========================#
    #======================================================================#
    #======================================================================#
    results_folders = {}
    for pred_hor in pred_horizons:
        pred_hor_dir = main_results_folder + f'/PredHor-{pred_hor}/'
        if (os.path.exists(pred_hor_dir)):
            if (os.listdir(pred_hor_dir)): # Non empty dir
                simple_train_res_dir = pred_hor_dir + '/SimpleTraining/'
                if (os.path.exists(simple_train_res_dir)):
                    if (os.listdir(simple_train_res_dir)): # Non empty dir
                        if (f'PredHor-{pred_hor}' not in results_folders):
                            results_folders[f'PredHor-{pred_hor}'] = {model: {epi_informed_method: "" for epi_informed_method in epi_informed_methods} for model in models}
                        # Getting the paths of the results folders for each model
                        for res_dir in os.listdir(simple_train_res_dir):
                            for model in results_folders[f'PredHor-{pred_hor}']:
                                if (model == 'STM-GNN'):
                                    tmp_model_name = 'STM'
                                else:
                                    tmp_model_name = model
                                if (tmp_model_name.lower() in res_dir.lower()):
                                    if (model not in results_folders[f'PredHor-{pred_hor}']):
                                        results_folders[f'PredHor-{pred_hor}'][model] = {epi_informed_method: "" for epi_informed_method in epi_informed_methods}
                                    if ("notepidemioinformed" in res_dir.lower()):
                                        results_folders[f'PredHor-{pred_hor}'][model]["None"] = simple_train_res_dir + f'/{res_dir}/'
                                    if ("autodiff" in res_dir.lower()):
                                        results_folders[f'PredHor-{pred_hor}'][model]["AutoDiff"] = simple_train_res_dir + f'/{res_dir}/'
                                    if ("ode-nsp" in res_dir.lower()):
                                        results_folders[f'PredHor-{pred_hor}'][model]["ODE-NSP"] = simple_train_res_dir + f'/{res_dir}/'
        

    #======================================================================#
    #======================================================================#
    #===================Getting the metrics of each exp.===================#
    #======================================================================#
    #======================================================================#
    results_states_pred = {}
    results_trans_pred = {}
    for pred_hor in results_folders:
        int_pred_hor = int(pred_hor.split('-')[-1])
        results_states_pred[pred_hor] = "\n\n"
        results_trans_pred[pred_hor] = "\n\n"
        for model in models:
            for epi_informed_method in results_folders[pred_hor][model]:
                results_folder = results_folders[pred_hor][model][epi_informed_method]
                latex_state_pred_results_str_final = "\t"
                latex_trans_pred_results_str_final = "\t"
                if (model == 'GAT') and (epi_informed_method == 'None'):
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                    latex_state_pred_results_str_final += f"\t\\multirow{{18}}{{*}}{{{int_pred_hor}}} & \\multirow{{4}}{{*}}{{GAT}} & None &"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                    latex_trans_pred_results_str_final += f"\t\\multirow{{18}}{{*}}{{{int_pred_hor}}} & \\multirow{{4}}{{*}}{{GAT}} & None &"
                if (len(results_folder) == 0):
                    # We do not have results (yet) for this model
                    if (epi_informed_method == 'None'):
                        latex_state_pred_results_str_final += f" & \\multirow{{4}}{{*}}{{{model}}} & None & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $\\\\"
                        latex_trans_pred_results_str_final += f" & \\multirow{{4}}{{*}}{{{model}}} & None & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $\\\\"
                    else:
                        latex_state_pred_results_str_final += f" &  & {epi_informed_method} & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $\\\\"
                        latex_trans_pred_results_str_final += f" &  & {epi_informed_method} & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $ & $ \\pm $\\\\"
                else:
                    #======================================================================#
                    #===========================Writing Beginning===========================#
                    #======================================================================#
                    if (model == 'GAT') and (epi_informed_method == 'None'):
                        # Ignore car done just before
                        pass
                    else:
                        if (epi_informed_method == 'None'):
                            latex_state_pred_results_str_final += f" & \\multirow{{4}}{{*}}{{{model}}} & None &"
                            latex_trans_pred_results_str_final += f" & \\multirow{{4}}{{*}}{{{model}}} & None &"
                        else:
                            latex_state_pred_results_str_final += f" &  & {epi_informed_method} &"
                            latex_trans_pred_results_str_final += f" &  & {epi_informed_method} &"

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
                    h5_path = results_folder +  '/metrics/final_results_all_repetitions_0.hdf5'
                    if not os.path.exists(h5_path):
                        print(f"Error: File not found at {h5_path}")
                        return

                    # Folders for the results (to use to get the metrics if exist, or to create if not)
                    output_dir = results_folder +  'analysis_plots'
                    perf_dir = results_folder +  'perf_dir'
                    if (force_metrics_computation):
                        # Dir for plots
                        create_output_dir_files = True
                        # Dir for metrics
                        create_perf_dir_files = True
                    else:
                        # Dir for plots
                        create_output_dir_files = False
                        if (os.path.exists(output_dir)):
                            if (not os.listdir(output_dir)): # Empty dir
                                create_output_dir_files = True
                        else:
                            create_output_dir_files = True
                        # Dir for metrics
                        create_perf_dir_files = False
                        if (os.path.exists(perf_dir)):
                            if (not os.listdir(perf_dir)): # Empty dir
                                create_perf_dir_files = True
                        else:
                            create_perf_dir_files = True
                    

                    if (create_output_dir_files) or (create_perf_dir_files):
                        #======================================================================#
                        # Create folders to store the results
                        os.makedirs(output_dir, exist_ok=True)
                        os.makedirs(perf_dir, exist_ok=True)

                        #======================================================================#
                        # Load data
                        print(f"Loading results from {h5_path}...")
                        
                        # Get main metrics
                        loss_storage, metrics_storage, n_classes = get_metrics_per_rep(h5_path, params_exp)

                        #======================================================================#
                        # Plot losses
                        print("\n" + "="*50)
                        print("FINAL RESULTS SUMMARY (Last Epoch Average)")
                        print("="*50)
                        
                        if (create_output_dir_files):
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
                                plt.close()


                        #======================================================================#
                        # Plot Main Metrics
                        # Random model
                        print("\n\n\n========================================>RANDOM MODEL<========================================")
                        for metric_name, split_data in metrics_storage.items():
                            if ("random" in metric_name.lower()):
                                print("\n\n")
                                if ('perclass' not in metric_name.lower()):
                                    plot_metric_curve(metric_name, split_data, output_dir, show_plot=False)
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
                                    metrics_statistics = plot_metric_curve(metric_name, split_data, output_dir, show_plot=False)
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

                        
                        #======================================================================#
                        # Print mean metrics OF ONSET IFECTION USING THE PREDICTED STATES
                        print("\n\n==========> Results ONSET INFECTION <==========")
                        for metric_name in results_onset_infection:
                            if ('fpr' not in metric_name.lower()) and ('tpr' not in metric_name.lower()):
                                print(f"\t===> {metric_name}: {round(np.mean(results_onset_infection_per_rep[metric_name])*100, 2):.2f} \pm {round(np.std(results_onset_infection_per_rep[metric_name])*100, 2):.2f}")
                        print("\n\n")
                    
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
                    if (params_exp['predict_state_transitions']):
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

                    # Final LaTeX strings
                    latex_state_pred_results_str_final += latex_state_pred_results_str
                    if (params_exp['predict_state_transitions']):
                        latex_trans_pred_results_str_final += latex_trans_pred_results_str
                    else:
                        latex_trans_pred_results_str_final += f" $ \pm $ & $ \pm $ & $ \pm $ & $ \pm $ & $ \pm $\\\\ "

                #======================================================================#
                #==============================Writing End==============================#
                #======================================================================#
                if (model == 'STM-GNN') and (epi_informed_method == 'ODE-NSP'):
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                    if (pred_hor != 'PredHor-7'):
                        latex_state_pred_results_str_final += "\n\n\t \\midrule \n\n"
                        latex_trans_pred_results_str_final += "\n\n\t \\midrule \n\n"
                    else:
                        latex_state_pred_results_str_final += "\n\n\t \\bottomrule \n\n"
                        latex_trans_pred_results_str_final += "\n\n\t \\bottomrule \n\n"
                else:
                    if (epi_informed_method == 'ODE-NSP'):
                        latex_state_pred_results_str_final += "\n\t\\cmidrule(lr){2-8}"
                        latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                        latex_state_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                        latex_trans_pred_results_str_final += "\n\t\\cmidrule(lr){2-8}"
                        latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                        latex_trans_pred_results_str_final += "\n\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    else:
                        latex_state_pred_results_str_final += "\n\t\\cmidrule(lr){3-8}"
                        latex_trans_pred_results_str_final += "\n\t\\cmidrule(lr){3-8}"

                # Add to final variables
                results_states_pred[pred_hor] += latex_state_pred_results_str_final + "\n"
                results_trans_pred[pred_hor] += latex_trans_pred_results_str_final + "\n"

    
    #======================================================================#
    #======================================================================#
    #=============================Print results=============================#
    #======================================================================#
    #======================================================================#
    for pred_hor in results_trans_pred:
        # Transitions prediction
        print(f"\n\n\n\n\n\n\n=======================================================================================================================")
        print(f"=======================================================================================================================")
        print(f"========================================> TRANISTIONS PREDICTION RESULTS FOR {pred_hor}<========================================")
        print(f"=======================================================================================================================")
        print(f"=======================================================================================================================\n")
        print(results_trans_pred[pred_hor])

    for pred_hor in results_states_pred:
        # State prediciton
        print(f"\n\n\n\n\n\n\n=================================================================================================================")
        print(f"=================================================================================================================")
        print(f"========================================> STATE PREDICTION RESULTS FOR {pred_hor}<========================================")
        print(f"=================================================================================================================")
        print(f"=================================================================================================================\n")
        print(results_states_pred[pred_hor])


if __name__ == "__main__":
    main()