#!/usr/bin/env python3
"""
    Code to generate synthetic SocioPatterns datasets
"""
import os
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import datetime
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from tqdm import tqdm

from src.DataManipulation.socio_patterns.data_utils import reindex, extension
from src.DataManipulation.socio_patterns.epidemic import GraphSEIRPerIndividual, infection_probability, incubation_duration, recovery_duration


def collapsed_per_day(data, setting='HET'):
    """
        Groups the interactions between individuals by days. This means that if
        two individuals i and j interact N times in one day, the weight between
        those two individuals is N*20 for the graph of that (as each interaction
        correspond to 20 s.

        Arguments:
        ----------
        data: pd.DataFrame
          Pandas DataFrame containing the SFHH data to group.
        setting: str
          Choice of modelling the weights between individuals. Two options:
          - HET : the weights between nodes (individuals) correspond to the total
          time (in seconds) that the individuals were in contact that day.
          - HOM : constant edge weight of 1

        Returns:
        --------
        collapsed: pd.DataFrame
          Pandas DataFrame with three columns: i, j, weight.
          i and j corresponds to the individuals in contact, and "weight"
          to the weight between the two nodes (of individuals), in seconds.
    """
    collapsed = data.copy()
    collapsed['day'] = collapsed.t.dt.date
    if ('Si' in collapsed.keys() and 'Sj' in collapsed): # HospitalWard dataset
        collapsed = collapsed.groupby(['day', 'i', 'j', 'Si', 'Sj']).count().reset_index().rename(columns={'t': 'weight'})
    else: # SFHH dataset
        collapsed = collapsed.groupby(['day', 'i', 'j']).count().reset_index().rename(columns={'t': 'weight'})
    if setting == 'HET':
        collapsed['weight'] = collapsed['weight'] * 20 # each edge is the total time spent together
    elif setting == 'HOM':
        collapsed['weight'] = 1 # all edges are the same
    else :
        raise NotImplementedError(f'{setting} is not implemented')

    return collapsed



def to_graphs_SFHH(data, temporal_granularity="day"):
  """
      Converts a pandas datafame into a list of graphs corresponding to the
      evolution of interactions over time. The nodes are the individuals
      and the weights are the time (in seconds) they spend together in the
      considered granularity of time.

      Arguments:
      ----------
      data: pd.DataFrame
        Pandas DataFrame containing the SFHH data to convert to graphs.
      temporal_granularity: str
        Granularity to consider time evolution. Options:
          - day: Groups the interaction by day. This means that if two individuals
          i and j interact N times in one day, the weight between those two
          individuals is N*20 for the graph of that (as each interaction
          correspond to 20 s)
          - hours: Groups the interactions by set of NON-OVERLAPPING hours
          (first hour, second hours, etc.). This means that if two individuals
          i and j interact N times in the k-th hour (w.r.t the beginning
          of the recording) them the weight between them for the graph of that
          hour is N*20 (as each interaction correspond to 20 s).

      Returns:
      --------
      graphs: list
        List of graphs representing the temporal evolution based on the
        chosen temporal granularity

  """
  # Grouping the data according to the chosen granularity
  if (temporal_granularity.lower() == "day"): # Damien's original implenetation
    collapsed = collapsed_per_day(data)
  else:
    raise ValueError("Temporal granularity {} not valid".format(temporal_granularity))
  timecolumn = temporal_granularity.lower()

  # Creating the list of graphs
  collapsed.sort_index(inplace=True)
  graphs = []
  for time_scale, df in collapsed.groupby(timecolumn):
      g = nx.Graph()
      g.add_nodes_from(df.i.unique())
      g.add_nodes_from(df.j.unique())
      g.add_edges_from([(row.i, row.j, {'weight': row.weight}) for row in df.itertuples()])
      graphs.append(g)

  return graphs

def to_graphs_HospitalWard(data):
  """
      Converts a pandas datafame into a list of graphs corresponding to the
      evolution of interactions over time. The nodes are the individuals
      and the weights are the time (in seconds) they spend together in the
      considered granularity of time.

      Arguments:
      ----------
      data: pd.DataFrame
        Pandas DataFrame containing the SFHH data to convert to graphs.
     
      Returns:
      --------
      graphs: list
        List of graphs representing the temporal evolution based on the
        chosen temporal granularity

  """
  # Grouping the data according to the chosen granularity
  timecolumn = "day"

  # Creating the list of graphs
  collapsed = collapsed_per_day(data)
  collapsed.sort_index(inplace=True)
  graphs = []
  for time_scale, df in collapsed.groupby(timecolumn):
      # Create graph
      g = nx.Graph()
      # Add nodes
      g.add_nodes_from(df.i.unique())
      g.add_nodes_from(df.j.unique())
      # Add attributes to nods
      for row in df.itertuples():
          # IMPORTANT: using BOTH i and j allow us to be sure that we have ALL the individuals. As the nature of an individual does not change over time, there is no problem to re-write its value
          g.nodes[row.i]["Nature"] = row.Si # row.i is the identifier of the individual, and row.Si indicates if it is a patient, a nurse, etc.
          g.nodes[row.j]["Nature"] = row.Sj # row.j is the identifier of the individual, and row.Sj indicates if it is a patient, a nurse, etc.
      # Add edges
      g.add_edges_from([(row.i, row.j, {'weight': row.weight}) for row in df.itertuples()])
      graphs.append(g)

  return graphs

def raw_data_loading(raw_data_path, do_plots=True):
    """
        Loads the raw data into memory and plot some basic 
        information and figures.

        Parameters:
        -----------
        raw_data_path: str
            Path to the raw data.
        do_plots: true
            True if want to get some basic plots.

        Returns:
        --------
        data: pd.DataFrame
    """
    # Data loading
    if ('sfhh' in raw_data_path.lower()):
        data = pd.read_csv(raw_data_path, sep=' ', header=None)
        data.columns = ['t', 'i', 'j']
        data.t = pd.to_datetime(data.t, unit='s', origin='2009-06-03 00:00:00') # collection started the 3rd June 2009 at 9am
    else:
        data = pd.read_csv(raw_data_path, sep='\t', header=None)
        data.columns = ['t', 'i', 'j', 'Si', 'Sj']
        data.t = pd.to_datetime(data.t, unit='s', origin='2010-12-06 13:00:00') # collection started the 6th December 2010 at 1pm


    # Number of individuals
    n_uniques = set(data['i'].unique()).union(set(data['j'].unique()))
    print("Number of unique individuals: {}".format(len(n_uniques)))

    # Do some basic plots
    if (do_plots):
        # Number of interactions per days
        fig, ax = plt.subplots(1, 1)
        sns.histplot(data.t, bins=100, ax=ax)
        fig.suptitle('Interactions')
        ax.set_xlabel('Time')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Number of contacts')
        plt.tight_layout()
        plt.show()

        # Co-occurence matrix of contacts
        cooc = pd.crosstab(data.i, data.j, normalize="index") # IMPORTANT: Normalize over each row as the matrix is very sparse (only some individuals interact more than 20 s between them)
        plt.figure(figsize=(10, 10))
        plt.title('Co-occurence matrix of contacts')
        sns.heatmap(cooc, cmap='rocket', cbar=True)
        plt.tight_layout()
        plt.show()

    # Reindex the indices of the nodes to be consecutive integers
    # "data" contains the data with the new indices, and "mapping" contains the correspondence between the new and old indices
    data, mapping = reindex(data)

    # Basic plot of graphs
    if (do_plots):
        # Converting to graph
        if ('sfhh' in raw_data_path.lower()):
            graphs = to_graphs_SFHH(data) 
        else:
            graphs = to_graphs_HospitalWard(data)
        n_graphs_plot = min(7, len(graphs))
        fig, axs = plt.subplots(1, n_graphs_plot, figsize=(15, 5))
        for d, (ax, g) in enumerate(zip(axs, graphs)):
            if (d >= n_graphs_plot):
                print("Plotting only the first 5 graphs")
                break
            ax.set_title("DAY {}".format(d+1))
            nx.draw(g, ax=ax, with_labels=True)
        plt.show()

        # Node degree histogram
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle('Node degree histogram')
        df_degree = pd.DataFrame(columns=['day', 'node', 'degree'])
        for i, g in enumerate(graphs):
            df_degree = pd.concat([df_degree, pd.DataFrame({'day': i, 'node': list(dict(g.degree).keys()), 'degree': list(dict(g.degree).values())})])

        ax.set_xlabel('Degree')
        ax.set_ylabel('Number of nodes')
        sns.histplot(df_degree, x='degree', hue='day', bins=100, ax=ax, kde=True)
        print(df_degree.mean())
        df_degree.describe()

    return data


def get_extended_data(raw_data, dataset_type='SFHH', n_days_use=50, do_plots=True):
    """
        As the SocioPatterns datasets are only a couple of days, this function
        extends the datasets to several days using the strategies described
        in Stehlé et al. (2011) "Simulation of an SEIR infectious disease model on
        the dynamic contact network of conference attendees".

        Parameters:
        -----------
        raw_data: pd.Dataframe
            Raw data to extend.
        dataset_type: str
            Dataset type. Two options: SFHH or HospitalWard.
        do_plots: bool
            True if want to do some basic plots.

        Returns:
        --------
        extended_data: pd.DataFrame
            Extended data having more days.
    """
    # Getting initial days
    if (dataset_type.lower() == 'sfhh'):
        day1 = raw_data[raw_data.t.dt.date == datetime.date(2009, 6, 3)]
        day2 = raw_data[raw_data.t.dt.date == datetime.date(2009, 6, 4)]
        strategy = 'CONSTR-SH'
        strategy_kwargs = {'b': 1e4, 'n_shuffle': 20}
        n_days_add = n_days_use - 2
        days = [day1, day2]
        for _ in tqdm(range(n_days_add)):
            days.extend(extension(days, strategy='CONSTR-SH', strategy_kwargs=strategy_kwargs, verbose=False))
        days = pd.concat(days)
        extended_data = collapsed_per_day(days, setting='HET')
    else:
        # Get collpased raw_data as it is needed by the method simulate of GraphSEIR
        # This was done by Damien to "simulate" a longer period (more days)?
        day1 = raw_data[raw_data.t.dt.date == datetime.date(2010, 12, 6)]
        day2 = raw_data[raw_data.t.dt.date == datetime.date(2010, 12, 7)]
        day3 = raw_data[raw_data.t.dt.date == datetime.date(2010, 12, 8)]
        day4 = raw_data[raw_data.t.dt.date == datetime.date(2010, 12, 9)]
        day5 = raw_data[raw_data.t.dt.date == datetime.date(2010, 12, 10)]
        strategy = 'CONSTR-SH'
        strategy_kwargs = {'b_in': 2, 'b_out':1, 'n_shuffle': 20}
        days = [day1, day2, day3, day4, day5]
        days = extension(days, strategy='CONSTR-SH', n_days_use=n_days_use, strategy_kwargs=strategy_kwargs, dataset_type='HospitalWard', verbose=False)
        days = pd.concat(days)
        extended_data = collapsed_per_day(days, setting='HET')

    # Basic plot
    if (do_plots):
        # Duration of interactions distribution (seconds)
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharey=True, sharex=True)
        fig.suptitle('Interactions')
        ax.set(xscale="log", yscale="log")
        sns.scatterplot(data=extended_data.rename(columns={'weight' : 'duration'}).duration.value_counts().sort_index(), ax=ax)
        ax.set_xlabel('Duration of interactions (seconds)')
        ax.set_ylabel('Number of interactions')
        plt.show()
        print(extended_data.weight.describe())

    return extended_data



def simulate_per_individual_SEIR(
                                    extended_data,
                                    n_inital_infected,
                                    pathogen="Coronavirus",
                                    n_simulations=10,
                                    fix_features=False,
                                    do_plots=True,
                                    do_tests=True
                                ):
    """
        Simulates one SEIR run using the extended data. It simulates the propagation
        some basic features of the individuals, as well as the propagation over time
        of the SEIR states, assuming that initially one subject is infected.

        Parameters:
        -----------
        extended_data: pd.DataFrame
            Extended data having more days. 
        pathogen: str
            Pathogen to simulate.
        n_simulations: int
            Number of simulations to run (each simulation generates a dataset).
        fix_features: bool
            If True, the simulated features are fixed (simulated once for all the datasets)
            for all the simulations. In that case, the only thing that changes between
            two simulations is the initial infected individual.
        do_plots: bool
        do_tests: bool
                
        Returns:
        --------
        SEIR_Per_Ind_model: GraphSEIRPerIndividual
            Model representing the evolution of the SEIR model for the selected 
            SocioPattern dataset.
        list_nodes_ids_per_rep: list
        list_edges_indices_per_rep: list
        list_edges_weights_per_rep: list
        list_timestamps_per_rep: list
        list_features_per_rep: list
        list_targets_per_rep: list
        list_states_per_rep: list
        list_stats_per_rep: list
        list_states_history_per_rep: list
        list_infection_rates_per_rep: list
        list_incubation_durations_per_rep: list
        list_recovery_durations_per_rep: list
    """
    #======================================================================#
    #======================================================================#
    # Getting the parameters for the selected pathogen
    if (pathogen.lower() == "influenza"):
        base_incubation = 2
        base_recovery = 1
        base_infection_factor = 0.02 # To match old's damien infection probability
    elif (pathogen.lower() == "norovirus"): # IMPORTANT: THIS WAS THE ONE USED IN EXPERIMENTS BEFORE: 03/10/2025
        base_incubation = 1
        base_recovery = 2
        base_infection_factor = 0.02 # To match old's damien infection probability
    elif (pathogen.lower() == "coronavirus") or (pathogen.lower() == "covid"):
        base_incubation = 5
        base_recovery = 14 
        base_infection_factor = 1 # Chosen such that the average infection probability is close to a Covid transmission risk to get R0 = beta/gamma = 2.5) 
    else:
        raise ValueError(f"\npathogen {pathogen} is not valid for simulation.\n")
    
    #======================================================================#
    #======================================================================#
    # Creating an instance of the model
    SEIR_Per_Ind_model = SEIR_Per_Ind_model = GraphSEIRPerIndividual(
                                                                        base_incubation=base_incubation,
                                                                        base_recovery=base_recovery,
                                                                        base_infection_factor=base_infection_factor
                                                                    )
    
    # Unique IDs of each individual
    n_uniques = set(extended_data['i'].unique()).union(set(extended_data['j'].unique()))
    
    # Run SEIR model with parameters per individual
    list_nodes_ids_per_rep = []
    list_edges_indices_per_rep = []
    list_edges_weights_per_rep = []
    list_timestamps_per_rep = []
    list_features_per_rep = []
    list_targets_per_rep = []
    list_states_per_rep = []
    list_stats_per_rep = []
    list_states_history_per_rep = []
    list_infection_rates_per_rep = []
    list_incubation_durations_per_rep = []
    list_recovery_durations_per_rep = []
    initial_features = None
    initial_raw_features = None
    initial_corr_features = None
    for sim_ID in range(n_simulations):       
        #======================================================================#
        # Simulate the evolution of the SEIR model
        num_individuals = len(n_uniques)
        states = np.array(['S']*num_individuals)
        idx_infected = np.random.choice(num_individuals, n_inital_infected, replace=False)
        states[idx_infected] = 'I'

        # Fix features if asked
        if (fix_features):
            if (initial_features is None):
                precomputed_features = None 
            else:
                precomputed_features = initial_features
        else:
            precomputed_features = None

        # Simulate propagation
        nodes_ids,\
        edges_indices,\
        edges_weights,\
        timestamps,\
        features,\
        targets,\
        states,\
        stats,\
        roles,\
        corr_features,\
        raw_features= SEIR_Per_Ind_model.simulate(
                                                states,
                                                extended_data.groupby("day"),
                                                dt=1,
                                                scale=1,
                                                #scale=12*3600,
                                                precomputed_features=precomputed_features
                                            )
        list_nodes_ids_per_rep.append(nodes_ids)
        list_edges_indices_per_rep.append(edges_indices)
        list_edges_weights_per_rep.append(edges_weights)
        list_timestamps_per_rep.append(timestamps)
        list_features_per_rep.append(features)
        list_targets_per_rep.append(targets)
        list_states_per_rep.append(states)
        list_stats_per_rep.append(stats)
        list_states_history_per_rep.append(SEIR_Per_Ind_model._states_history)
        list_infection_rates_per_rep.append(SEIR_Per_Ind_model.infection_rates)
        list_incubation_durations_per_rep.append(SEIR_Per_Ind_model.incubation_durations)
        list_recovery_durations_per_rep.append(SEIR_Per_Ind_model.recovery_durations)

        # Saving first features
        if (initial_features is None):
            initial_features = deepcopy(features)
            initial_raw_features = deepcopy(raw_features)
            initial_corr_features = deepcopy(corr_features)

        #======================================================================#
        # Plot features histograms if asked
        if (do_plots):
            if (fix_features):
                raw_features = initial_raw_features
                corr_features = initial_corr_features
            # Age
            ages_raw, ages = raw_features['age'], features[0, :, corr_features["age"]]
            # =======> BEFORE NORMALIZATION (before robust or standard scaling as Geissbuhler et al. 2025) <=======
            fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
            fig.suptitle('Age distribution (before robust scaling)')
            plt.xlabel("Age")
            sns.histplot(ages_raw, ax=ax, kde=True)
            # =======> AFTER NORMALIZATION (before robust or standard scaling as Geissbuhler et al. 2025) <=======
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
            fig.suptitle('Age distribution (after robust scaling)')
            plt.xlabel("Age")
            sns.histplot(ages, ax=ax, kde=True)

            # ORIGIN (acute care unit, Residence, RSOP)
            origins_raw = raw_features['origin']
            if (do_plots):
                # Histogram
                fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
                fig.suptitle("Origins distribution")
                plt.xlabel("Origin")
                sns.histplot(origins_raw, ax=ax, kde=False)

            # GENDER
            genders_raw = raw_features['gender']
            if (do_plots):
                # Histogram
                fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
                fig.suptitle("Genders distribution")
                plt.xlabel("Gender")
                sns.histplot(genders_raw, ax=ax, kde=False)

            # COMORBIDITIES (multi-hot encoding: Diabetes, Stroke, Cancer, Obesity, Smoke)
            comorbidities = raw_features['comorbidities']
            if (do_plots):
                # Histogram
                # IMPORTANT: AS ONE PATIENT CAN HAVE SEVERAL COMORBIDITIES, THE SIZE OF THE TOTAL COMORBIDITIES CAN BE GREATER THAM THE NUÉBER OF PATIENTS
                comorbidities_unrolled_all = []
                for comorb in comorbidities:
                    comorbidities_unrolled_all += [comorb for _ in range(comorbidities[comorb].sum())]
                fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True, sharey=True)
                fig.suptitle("Comorbidities distribution")
                plt.xlabel("Comorbidity")
                sns.histplot(comorbidities_unrolled_all, ax=ax, kde=False)

            # Verifying that, over the sequence of graphs, a given node have the same static features
            if (do_tests):
                static_features_id_init, static_features_id_end = 0, 10
                test_passed = True
                for day_id in range(features.shape[0]-1): # -1 because we are going to compare node i with node i+1
                    if ( (features[day_id, :, 0:10] == features[day_id+1, :, 0:10]).sum() != features.shape[1]*(static_features_id_end-static_features_id_init)):
                        print(f"\nPROBLEM: between day {day_id} and {day_id+1} there are some static features that changed!")
                        test_passed =  False
                        break
                if (test_passed):
                    print("TEST PASSED: Nodes have the same static features between sequences of graphs.")
                else:
                    print("TEST NOT PASSED: Nodes DO NOT have the same static features between sequences of graphs.")

            plt.show()

    
    return SEIR_Per_Ind_model,\
           list_nodes_ids_per_rep,\
           list_edges_indices_per_rep,\
           list_edges_weights_per_rep,\
           list_timestamps_per_rep,\
           list_features_per_rep,\
           list_targets_per_rep,\
           list_states_per_rep,\
           list_stats_per_rep,\
           list_states_history_per_rep,\
           list_infection_rates_per_rep,\
           list_incubation_durations_per_rep,\
           list_recovery_durations_per_rep

def get_SEIR_epi_params_per_day(features, corr_features, base_infection_factor):
    # Infection rates
    infection_rates_per_day = []
    infection_rates_only_contact_ind_per_day = []
    for day in tqdm(range(features.shape[0])):
        tmp_interactions = collapsed_data.groupby("day")
        interactions_durations = []
        for i, (d, df) in enumerate(tmp_interactions):
            if (i == day):
                src, dst, weights = df.i, df.j, df.weight
                for a, b, w in zip(src, dst, weights):
                    interactions_durations.append((a, b, w))
            else:
                pass
        infection_probabilities = np.zeros((features.shape[1], features.shape[1]))
        infection_probabilities_only_contact_ind = []
        for ind_a_id, ind_b_id, inter_dur in interactions_durations:
            ind_a = features[day, ind_a_id]
            ind_b = features[day, ind_b_id]
            inf_prob = infection_probability(ind_a, ind_b, inter_dur, corr_features, base_infection_factor)
            infection_probabilities[ind_a_id, ind_b_id] = inf_prob
            infection_probabilities_only_contact_ind.append(inf_prob)
        infection_probabilities = np.array(infection_probabilities)
        infection_rates_per_day.append(infection_probabilities)
        infection_rates_only_contact_ind_per_day.append(infection_probabilities_only_contact_ind)
    
    
    # Recovery and incubation durations
    recovery_durations_per_day = []
    incubation_durations_per_day  = []
    for day in tqdm(range(features.shape[0])):
        recovery_durations = []
        incubation_durations = []
        for ind_id in range(features.shape[1]):
            ind = features[day, ind_id]
            rec_dur = recovery_duration(ind, corr_features)
            inc_dur = incubation_duration(ind, corr_features)
            recovery_durations.append(rec_dur)
            incubation_durations.append(inc_dur)
        recovery_durations = np.array(recovery_durations)
        incubation_durations = np.array(incubation_durations)
        recovery_durations_per_day.append(recovery_durations)
        incubation_durations_per_day.append(incubation_durations)

    # Tranform into numpy arrays
    infection_rates_per_day = np.array(infection_rates_per_day)
    # Important: infection_rates_only_contact_ind_per_day cannot be transformed into np.arrays as it is a list of lists of different shapes (as interactions change over time)
    incubation_durations_per_day = np.array(incubation_durations_per_day)
    recovery_durations_per_day = np.array(recovery_durations_per_day)
    
    return infection_rates_per_day,\
           infection_rates_only_contact_ind_per_day,\
           incubation_durations_per_day,\
           recovery_durations_per_day


def plot_SEIR_evolution(stats):
    """
        Plot SEIR evolution over time (for one simulation)
    """
    # Plotting the results of the simulation
    # Re-ordering the results
    # In Pandas, the .melt() method is used to transform a DataFrame from a wide format to a long format. It essentially "unpivots" the DataFrame, turning columns into rows.
    SEIR_Per_Ind_evolution_values = stats.melt(id_vars='step', var_name='state', value_name='count')

    # Plot
    fig, axs = plt.subplots(1,1,figsize=(15,5))
    fig.suptitle('SEIR model with parameters per individual')
    ax = axs
    sns.lineplot(data=SEIR_Per_Ind_evolution_values, x='step', y='count', hue='state', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of individuals')
    plt.show()




#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
def main():
    #==========================================================================#
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--raw_data_path', required=True, help="Path to the SocioPattern raw data", type=str)
    # nargs='+' allows to specify that at least one value is expected
    ap.add_argument('--n_days_use', default=[50, 100, 200], help="Number of days to use for the datasets (extend the days to that number of days).", type=int)
    ap.add_argument('--n_datasets_generate', required=True, help="Number of datasets to generate", type=int)
    ap.add_argument('--do_plots', help="Use it if want to get some basic plots", action='store_true')
    ap.add_argument('--analyze_generated_data', help="Use it if want to analyze the generated data", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    raw_data_path = args['raw_data_path']
    n_days_use = args['n_days_use']
    n_datasets_generate = args['n_datasets_generate']
    do_plots = args['do_plots']
    analyze_generated_data = args['analyze_generated_data']


    #==========================================================================#
    #==========================================================================#
    # Loading raw data
    data = raw_data_loading(raw_data_path=raw_data_path, do_plots=do_plots)

    # Get dataset type
    if ('sfhh' in raw_data_path.lower()):
        dataset_type = 'SFHH'
        n_inital_infected = 100
    else:
        dataset_type = 'HospitalWard'
        n_inital_infected = 20

    #==========================================================================#
    #==========================================================================#
    for dataset_ID in tqdm(range(n_datasets_generate)):
        print(f"\n\n======================================================================")
        print(f"=====================Generating dataset {dataset_ID}=====================")
        print(f"======================================================================\n\n")
        #==========================================================================#
        # Extend days in the data
        extended_data = get_extended_data(raw_data=data, dataset_type=dataset_type, n_days_use=n_days_use, do_plots=do_plots)

        #==========================================================================#
        # Simulating per individual data
        SEIR_Per_Ind_model,\
        list_nodes_ids_per_rep,\
        list_edges_indices_per_rep,\
        list_edges_weights_per_rep,\
        list_timestamps_per_rep,\
        list_features_per_rep,\
        list_targets_per_rep,\
        list_states_per_rep,\
        list_stats_per_rep,\
        list_states_history_per_rep,\
        list_infection_rates_per_rep,\
        list_incubation_durations_per_rep,\
        list_recovery_durations_per_rep = simulate_per_individual_SEIR(
                                                                            extended_data,
                                                                            n_inital_infected,
                                                                            pathogen="Coronavirus",
                                                                            n_simulations=2,
                                                                            fix_features=True,
                                                                            do_plots=do_plots,
                                                                            do_tests=True
                                                                    )
        
        #==========================================================================#
        # Saving the dataset
        i = 0
        if (dataset_type.lower() == 'sfhh'):
            dataset_filename = "./data/SocioPatterns/SFHH/preprocessed/SFHH_SEIR_per_individual_{}_Days_Dataset-{}_".format(n_days_use, dataset_ID)
        else:
            dataset_filename = "./data/SocioPatterns/HospitalWard/preprocessed/HospitalWard_per_individual_{}_Days_Dataset-{}_".format(n_days_use, dataset_ID)
            pass 
        while (os.path.exists(dataset_filename + f"{i}.pkl")):
            i += 1
        dataset_filename = dataset_filename + f"{i}.pkl"
        with open(dataset_filename, 'wb') as f:
            pickle.dump(
                            (
                                list_nodes_ids_per_rep[0],
                                list_edges_indices_per_rep[0],
                                list_edges_weights_per_rep[0],
                                list_timestamps_per_rep[0],
                                list_features_per_rep[0],
                                list_targets_per_rep[0],
                                list_states_history_per_rep[0],
                                list_infection_rates_per_rep[0],
                                list_incubation_durations_per_rep[0],
                                list_recovery_durations_per_rep[0],
                            ),
                        f)
        
        #==========================================================================#
        # Analyze data
        if (analyze_generated_data):
            # SELECTING ONE REPETITION OF THE LAST GENERATED DATASET
            nodes_ids = list_nodes_ids_per_rep[0]
            edges_indices = list_edges_indices_per_rep[0]
            edges_weights = list_edges_weights_per_rep[0]
            timestamps = list_timestamps_per_rep[0]
            features = list_features_per_rep[0]
            targets = list_targets_per_rep[0]
            states = list_states_per_rep[0]
            stats = list_stats_per_rep[0]

            # Plot evolution
            plot_SEIR_evolution(stats)



if __name__=="__main__":
    main()