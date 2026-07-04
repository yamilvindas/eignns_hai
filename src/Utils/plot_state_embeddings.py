import os
import h5py
import yaml
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from src.Experiments.InfectionRiskPredMurcia import InfectionRiskPred
#import umap # Does not work because incompatibilities with the current Scikit-Learn Library that I have
from sklearn.manifold import TSNE

# Global variables
MAIN_TITLE_FONTSIZE = 70
AXIS_TITLE_FONTSIZE = 40
OTHER_FONTSIZE = 30
POINTS_SIZE = 80
MARKER_LEGEND_SIZE = 2

def plot_global_DR_analysis(data_dict, global_title=""):
    """
        IMPORTANT: This function was generated with Gemini Pro

        Performs DR on embeddings and plots 4 visualizations
        based on true/pred states and risks.
    """
    
    # 1. Extract and Process Data
    # Move tensors to CPU and convert to numpy
    embeddings = data_dict["Embeddings"].detach().cpu().numpy()
    true_states = data_dict["TrueStates"].detach().cpu().numpy()
    true_risk = data_dict["TrueInfRisk"].detach().cpu().numpy()
    
    # Process Probabilities to get Predicted Classes (Argmax)
    # Assuming shape is (N_samples, N_classes), we take the index of the highest probability
    pred_states_probs = data_dict["PredStatesProbs"].detach().cpu().numpy()
    pred_states = np.argmax(pred_states_probs, axis=1)
    
    pred_risk_probs = data_dict["PredInfRiskProbs"].detach().cpu().numpy()
    pred_risk = np.argmax(pred_risk_probs, axis=1)

    # Handling forecast horizon > 1
    if (len(pred_states_probs.shape) == 3): # This means that the forecast horizon is greater than 1
        # IMPORTANT: For consistency with the results in the paper, we only take the next day prediction
        pred_states_probs = pred_states_probs[:, :, 0]
        pred_states = pred_states[:, 0]
        true_states = true_states[:, 0]

    # 2. Run UMAP
    print(f"Running dimensionality reduction on {embeddings.shape[0]} samples...")
    #reducer = umap.UMAP(n_components=2, random_state=42)
    reducer = TSNE(n_components=2, init='pca', learning_rate='auto')
    embedding_2d = reducer.fit_transform(embeddings)

    # 3. Setup Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define plot configurations
    plot_configs = [
        {
            "title": "Colored by True State",
            "data": true_states,
            "cmap": "tab10", # Good for categorical
            "ax": axes[0]
        },
        {
            "title": "Colored by Predicted State",
            "data": pred_states,
            "cmap": "tab10",
            "ax": axes[1]
        },
        {
            "title": "Colored by True Infection Risk",
            "data": true_risk,
            "cmap": "viridis", # Good for ordinal/risk intensity
            "ax": axes[2]
        },
        {
            "title": "Colored by Predicted Infection Risk",
            "data": pred_risk,
            "cmap": "viridis",
            "ax": axes[3]
        }
    ]

    # 4. Generate Plots
    for config in plot_configs:
        ax = config["ax"]
        labels = config["data"]
        
        # Create scatter plot
        scatter = ax.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=labels, 
            cmap=config["cmap"], 
            s=POINTS_SIZE, 
            alpha=0.7
        )
        
        ax.set_title(config["title"], fontweight='bold', fontsize=AXIS_TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Add a legend/colorbar
        # If the number of unique classes is small (<15), use a discrete legend
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 15:
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="best", title="Classes", fontsize=OTHER_FONTSIZE, markerscale=MARKER_LEGEND_SIZE)
            ax.add_artist(legend1)
        else:
            # Otherwise use a colorbar
            plt.colorbar(scatter, ax=ax)

    plt.suptitle(global_title, 
                 fontsize=MAIN_TITLE_FONTSIZE, 
                 fontweight='bold', 
                 y=1.0) # y adjusts the vertical position to avoid overlapping subplots
                 #y=1.02) # y adjusts the vertical position to avoid overlapping subplots

    plt.tight_layout()
    plt.show()

    # Generate separate plots
    for config in plot_configs:
        plt.figure()
        labels = config["data"]

        # Create scatter plot
        scatter = plt.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=labels, 
            cmap=config["cmap"], 
            s=POINTS_SIZE, 
            alpha=0.7
        )
        
        plt.title(global_title + " -- " + config["title"], fontsize=MAIN_TITLE_FONTSIZE, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("")
        
        # Add a legend/colorbar
        # If the number of unique classes is small (<15), use a discrete legend
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 15:
            legend1 = plt.legend(*scatter.legend_elements(),
                                loc="best", title="Classes", fontsize=OTHER_FONTSIZE, markerscale=MARKER_LEGEND_SIZE)
        else:
            # Otherwise use a colorbar
            plt.colorbar(scatter)

        plt.show()



def plot_DR_inf_risk_analysis(data_dict, global_title=""):
    """
        IMPORTANT: This function was generated with Gemini Pro
        
        Performs t-SNE on PredStatesProbs and plots two figures:
        one for True Infection Risk and one for Predicted Infection Risk.
    """
    
    # 1. Prepare Data
    # We use the probabilities as the input features for t-SNE
    probs = data_dict["PredStatesProbs"].detach().cpu().numpy()
    
    true_risk = data_dict["TrueInfRisk"].detach().cpu().numpy()
    
    # Get Predicted Risk class by taking the argmax of the risk probabilities
    pred_risk_probs = data_dict["PredInfRiskProbs"].detach().cpu().numpy()
    pred_risk = np.argmax(pred_risk_probs, axis=1)

    # Handling forecast horizon > 1
    if (len(probs.shape) == 3): # This means that the forecast horizon is greater than 1
        # Reshape probs to get the input features for the infection risk predictor
        probs = probs.reshape(probs.shape[0], -1)

    # 2. Run t-SNE
    print(f"Running DR on probability vectors (shape: {probs.shape})...")
    # perplexity: related to the number of nearest neighbors (usually 30-50)
    # init='pca': ensures more global structure preservation and reproducibility
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    tsne_results = tsne.fit_transform(probs)

    # 3. Setup Plotting (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Define plot data
    plots = [
        {"ax": ax1, "labels": true_risk, "title": "True Infection Risk"},
        {"ax": ax2, "labels": pred_risk, "title": "Predicted Infection Risk"}
    ]

    for item in plots:
        ax = item["ax"]
        scatter = ax.scatter(
            tsne_results[:, 0], 
            tsne_results[:, 1], 
            c=item["labels"], 
            cmap='viridis', 
            s=POINTS_SIZE, 
            alpha=0.6
        )
        
        ax.set_title(item["title"], fontweight='bold', fontsize=AXIS_TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Risk Class", fontsize=OTHER_FONTSIZE, markerscale=MARKER_LEGEND_SIZE)
        ax.add_artist(legend)

    # 4. Add Global Title
    plt.suptitle(global_title, 
                 fontsize=MAIN_TITLE_FONTSIZE, fontweight='bold',
                 y=1.0) # y adjusts the vertical position to avoid overlapping subplots
                 #y=1.02) # y adjusts the vertical position to avoid overlapping subplots

    plt.tight_layout()
    plt.show()

    # Generate separate plots
    for item in plots:
        plt.figure()
        scatter = plt.scatter(
            tsne_results[:, 0], 
            tsne_results[:, 1], 
            c=item["labels"], 
            cmap='viridis', 
            s=POINTS_SIZE, 
            alpha=0.6
        )
        
        plt.title(global_title + ' -- ' + item["title"], fontsize=MAIN_TITLE_FONTSIZE, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("")
        
        # Add legend
        legend = plt.legend(*scatter.legend_elements(), loc="upper right", title="Risk Class", fontsize=OTHER_FONTSIZE, markerscale=MARKER_LEGEND_SIZE)

        plt.show()

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
    #==================Creating instance of the experiment==================
    #======================================================================#
    # Create instance
    tmp_exp = InfectionRiskPred(parameters_exp)

    # Create untrained models
    tmp_exp.modelCreation()

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
    #========================Getting the projections========================
    #======================================================================#
    # Get embeddings
    sorted_keys = sorted(list(models_dicts.keys()))
    for rep_ID in sorted_keys:
        # Create data
        if (len(tmp_exp.parameters_exp['multiple_hdf5_dataset_filenames']) == 1): 
            # In this case all the repetitions were trained and evaluated on the same datasets
            DS_ID = 0
        else:
            DS_ID = rep_ID
        tmp_exp.hdf5_dataset_filename = tmp_exp.parameters_exp['multiple_hdf5_dataset_filenames'][DS_ID]
        tmp_exp.current_dataset_ID = DS_ID
        tmp_exp.h5_file = h5py.File(tmp_exp.hdf5_dataset_filename, 'r')
        tmp_exp.createTorchDatasets()
        tmp_exp.dataloadersCreation()

        # Load models weights
        tmp_exp.modelCreation() # Reinitialize models weights
        for submodel in tmp_exp.models:
            tmp_exp.models[submodel].load_state_dict(models_dicts[rep_ID][submodel]['model_state_dict'])
            #tmp_exp.models[submodel] = tmp_exp.models[submodel].to(torch.device('cpu'))


        # Compute embeddings
        node_embeddings = tmp_exp.get_nodes_embeddings()

        # Plots
        for data_loader_type in node_embeddings:
            # Getting the predictions
            plot_global_DR_analysis(node_embeddings[data_loader_type], global_title=f"DR of the node embeddings for {data_loader_type.upper()} dataset for repetition {rep_ID}")
            plot_DR_inf_risk_analysis(node_embeddings[data_loader_type], global_title=f"DR of the Epidemic states for {data_loader_type.upper()} dataset for repetition {rep_ID}")



if __name__=='__main__':
   main() 