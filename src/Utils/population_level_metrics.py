"""
    Plot the metrics and results of an experiment
"""
import h5py
import yaml
import argparse
import pickle
from sklearn.metrics import mean_squared_error,\
                            roc_auc_score,\
                            average_precision_score
import torch
from torchmetrics.regression import ConcordanceCorrCoef
from src.Utils.classification_metrics import get_classification_metrics,\
                                             compute_ece,\
                                             get_classification_metrics_per_days
from src.Utils.learned_params_metrics import evaluate_learned_params,\
                                             get_local_infection_hazard
from src.Utils.uncertainty_quantification import get_evidential_uncertainties,\
                                                 get_uncertainties_per_rep_all_days,\
                                                 plot_metric_vs_uncertainty,\
                                                 plot_calibration_uncertainty_correctness,\
                                                 plot_calibrated_uncertainty_correctness
from src.Experiments.InfectionRiskPredMurcia import InfectionRiskPred
from src.DataManipulation.Murcia.data_exploration import estimate_epidemic_params_from_data,\
                                                         get_patients_data,\
                                                         load_movement_data,\
                                                         load_locations_data
from src.Utils.plot_metrics import plot_pred_epidemic_evolution


# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")


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
        dataset_type = 'Murcia'
    else:
        N_UNIQUE_CLASSES = [tmp_i for tmp_i in range(4)]
        dataset_type = 'SocioPatterns'

    

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
    #======================================================================#
    #======================================================================#
    #======================================================================#
    # Getting the metrics dicts for the different losses
    if ('Rep-0' in results_h5_file):
        base_name_main_group = 'Rep-'
        multiple_datasets = False
    else:
        base_name_main_group = 'Rep-0_Dataset-'
        multiple_datasets = True
        
    # Mapping
    if (params_exp['dataset_name'].lower() == 'sociopatterns'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    elif (params_exp['dataset_name'].lower() == 'murcia'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
    elif (params_exp['dataset_name'].lower() == 'hug'):
        SUSCEPTIBLE, INFECTIOUS, RECOVERED = 0, 1, 2 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'I' : INFECTIOUS, 'R' : RECOVERED}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}

    # Data to use
    #epoch_to_use = 0
    #data_split = "Train"
    #data_split = "Val"
    data_split = "Test"
    try:
        last_epoch = max([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"][data_split].keys())])
    except:
        last_epoch = 'Last'

    epoch_to_use = last_epoch

    # Plot
    plot_pred_epidemic_evolution(
                                    h5_results_file=results_h5_file,
                                    epoch_to_use=epoch_to_use,
                                    states_mapping=MAPPING,
                                    data_split=data_split,
                                    use_next_day_only=True,
                                    #use_next_day_only=False,
                                    #plot_error_bars=True
                                    plot_error_bars=False
                                )
        


if (__name__=='__main__'):
    main()