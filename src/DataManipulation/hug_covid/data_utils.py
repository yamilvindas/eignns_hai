import os
import yaml
import h5py
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Internal imports
from src.DataManipulation.DynamicHeteroGraphTemporalSignal import DynamicHeteroGraphTemporalSignal
from src.Utils.tools import one_hot_encoding_np

# Defining function that builds the dictionary
def build_ward_patient_dict(df: pd.DataFrame):
    """
        Generated with the help of ChatGPT
    """
    # Find global time span
    global_start = df["ward_start"].min().normalize()
    global_end = df["ward_end"].max().normalize()
    
    # Initialize nested defaultdict structure
    ward_dict = defaultdict(lambda: defaultdict(list))
    
    # Iterate rows
    for _, row in tqdm(df.iterrows()):
        ward = row["ward"]
        patient = row["patient_id"]
        start = row["ward_start"].normalize()
        end = row["ward_end"].normalize()
        
        # Generate all days the patient was in the ward (inclusive)
        days = pd.date_range(start, end, freq="1D")
        for day in days:
            ward_dict[ward][day.strftime("%Y-%m-%d")].append(patient)
    
    # Sort inner dictionaries by date
    ward_dict = {
        w: dict(sorted(days.items())) for w, days in ward_dict.items()
    }
    
    return dict(ward_dict)


# Defining function that builds the dictionary
def build_room_patient_dict(df: pd.DataFrame):
    """
        Generated with the help of ChatGPT
    """
    # Find global time span
    global_start = df["room_start"].min().normalize()
    global_end = df["room_end"].max().normalize()
    
    # Initialize nested defaultdict structure
    room_dict = defaultdict(lambda: defaultdict(list))
    
    # Iterate rows
    for _, row in tqdm(df.iterrows()):
        room = row["room"]
        patient = row["patient_id"]
        start = row["room_start"].normalize()
        end = row["room_end"].normalize()
        
        # Generate all days the patient was in the room (inclusive)
        days = pd.date_range(start, end, freq="1D")
        for day in days:
            room_dict[room][day.strftime("%Y-%m-%d")].append(patient)
    
    # Sort inner dictionaries by date
    room_dict = {
        w: dict(sorted(days.items())) for w, days in room_dict.items()
    }
    
    return dict(room_dict)



def get_nx_graph_snapshots(
                            pat_traj_DS_df,
                            place_dict_patients_per_day,
                            n_pats_per_place_per_day,
                            fully_connect_patients_same_place,
                            place_type='Ward'
                          ):
    # Params
    # Getting all possible dates
    all_dates_str = []
    for place in tqdm(place_dict_patients_per_day):
        for date_str in place_dict_patients_per_day[place]:
            all_dates_str.append(date_str)
    all_dates_str = np.unique(all_dates_str)
    # Sorting by increasing order
    all_dates_datetimes = [pd.to_datetime(tmp_date) for tmp_date in all_dates_str]
    sorted_idx = np.argsort(all_dates_datetimes)
    all_dates_str = [all_dates_str[i] for i in sorted_idx]

    # Creating the snapshots
    graph_snapshots = {}
    for date_str in tqdm(all_dates_str):
        g = nx.Graph()
        for place in place_dict_patients_per_day:
            if (date_str in place_dict_patients_per_day[place]):
                # Add patient nodes
                g.add_nodes_from(np.unique(place_dict_patients_per_day[place][date_str]), category="Patient")
                # Add place node
                g.add_node(place, category="Place")
                # Create edges
                for patient_id in place_dict_patients_per_day[place][date_str]:
                    g.add_edge(patient_id, place)                
                    g.add_edge(place, patient_id)
                    if (fully_connect_patients_same_place):
                        for patient_id_bis in place_dict_patients_per_day[place][date_str]:
                            if (patient_id_bis != patient_id):
                                g.add_edge(patient_id, patient_id_bis)                
                                g.add_edge(patient_id_bis, patient_id)
        graph_snapshots[date_str] = g
    # Build the dictionary
    if (place_type.lower() == 'ward'):
        place_dict_patients_per_day = build_ward_patient_dict(pat_traj_DS_df)
    else:
        place_dict_patients_per_day = build_room_patient_dict(pat_traj_DS_df)

    # Test if there is a coherence between the number of patients per day in place_dict_patients_per_day and n_pats_per_place_per_day
    for place in tqdm(place_dict_patients_per_day):
        for date_str in place_dict_patients_per_day[place]:
            datetime_date = pd.to_datetime(date_str).date()
            if (datetime_date in n_pats_per_place_per_day[place]):
                if (len(place_dict_patients_per_day[place][date_str]) != n_pats_per_place_per_day[place][datetime_date]):
                    raise RuntimeError(f"The number of patients in {place_type.lower()}_dict_patients_per_day ({len(place_dict_patients_per_day[place][date_str])}) for place {place} and day {date_str} is not consistent with the one in n_pats_per_{place_type.lower()}_per_day ({n_pats_per_place_per_day[place][datetime_date]})")
    print("\n=========> Test passed <=========\n")
    # Plot some of the graphs
    n_graphs_to_plot = 5
    for i in tqdm(range(n_graphs_to_plot)):
        # Getting the date to use
        date_str = all_dates_str[i]
        
        # Get the graph
        g = graph_snapshots[date_str]
        
        # Defining a Color Map for the different types of nodes
        nodes_types_to_color = {
                                    'Patient': 'skyblue',
                                    'Place': 'salmon'
                            }
        
        # Create a List of Colors for all Nodes
        # We iterate through the nodes in the order they appear in g.nodes()
        color_map = [nodes_types_to_color[g.nodes[node]['category']] for node in g]
        
        # Draw the Graph with the Custom Colors
        pos = nx.spring_layout(g, seed=42) # Define layout for consistent visualization
        
        plt.figure()
        
        nx.draw_networkx(
            g,
            pos,
            node_color=color_map, # Pass the list of colors here
            with_labels=True,
            node_size=1500,
            font_color='black',
            font_weight='bold',
            edge_color='gray'
        )
        
        # Optional: Add a legend to explain the node colors
        for category, color in nodes_types_to_color.items():
            plt.plot([], [], marker='o', color=color, linestyle='', label=category)
        
        plt.legend(scatterpoints=1)
        plt.title(f"Heterogeneous Graph Visualization for day {date_str} (Place type: {place_type})")
        plt.axis('off')
        plt.show()


def get_place_dict_with_infections_per_day(
                                                place_dict_patients_per_day,
                                                covid_cases_DS_df,
                                                place_type='Ward'
                                          ):
    # Forecast horizon to consider
    n_susceptible_nodes = 0
    n_susceptible_nodes_per_day = {}
    n_susceptible_nosoc_nodes_per_day = {}
    n_infected_nodes = 0
    n_infected_nodes_per_day = {}
    n_infected_nodes_nosoc_per_day = {}
    n_recovered_nodes = 0
    n_recovered_nodes_per_day = {}
    n_recovered_nodes_nosoc_per_day = {}
    place_dict_patients_per_day_with_infections = deepcopy(place_dict_patients_per_day)
    for place in tqdm(place_dict_patients_per_day):
        for date_str in place_dict_patients_per_day[place]:
            if (date_str not in n_susceptible_nodes_per_day):
                n_susceptible_nodes_per_day[date_str] = 0
                n_susceptible_nosoc_nodes_per_day[date_str] = 0
            if (date_str not in n_infected_nodes_per_day):
                n_infected_nodes_per_day[date_str] = 0
                n_infected_nodes_nosoc_per_day[date_str] = 0
            if (date_str not in n_recovered_nodes_per_day):
                n_recovered_nodes_per_day[date_str] = 0
                n_recovered_nodes_nosoc_per_day[date_str] = 0
            date = pd.to_datetime(date_str).date()
            for patient_id_idx in range(len(place_dict_patients_per_day[place][date_str])):
                # Get the patient ID
                patient_id = place_dict_patients_per_day[place][date_str][patient_id_idx]
                # Getting the swab date and the symptoms date
                swab_date = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == patient_id]['swab_date'].item()
                symptoms_date = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == patient_id]['symptoms_date'].item()
                infection_start_date = min(swab_date, symptoms_date).date()
                case_type = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == patient_id]['case_type'].item()
                # Infected or not?
                recovered = 0 # 0 means that the patient has never recovered from having had the virus, either because they have never had it or because they are currently infected.
                if (date >= infection_start_date): # In this case the patient is considered as infected WITHIN 14 DAYS AFTER THE DATE (AFTER THAT IT IS CONSIDERED AS RECOVERED)
                    # We have trajectories for ALL the hospital (not only geriatrics) but when a patient goes to a geriatric ward, it is because it has been recovered from COVID
                    if ( (date - infection_start_date) > timedelta(days=14) ): # After 14 days, the patient is considered as recovered
                        infection = 0
                        recovered = 1
                    else: 
                        infection = 1
                else:
                    infection = 0

                # Increasing counters
                if (infection == 0):
                    if (recovered == 0):
                        n_susceptible_nodes += 1
                        n_susceptible_nodes_per_day[date_str] += 1
                        if (case_type == 'nosocomial'):
                            n_susceptible_nosoc_nodes_per_day[date_str] += 1
                else:
                    n_infected_nodes += 1
                    n_infected_nodes_per_day[date_str] += 1
                    if (case_type == 'nosocomial'):
                        n_infected_nodes_nosoc_per_day[date_str] += 1
                if (recovered == 1):
                    n_recovered_nodes += 1
                    n_recovered_nodes_per_day[date_str] += 1
                    if (case_type == 'nosocomial'):
                        n_recovered_nodes_nosoc_per_day[date_str] += 1
                # Adding this information
                place_dict_patients_per_day_with_infections[place][date_str][patient_id_idx] = {
                                                                                                    "PatientID": patient_id,
                                                                                                    "Infection": infection,
                                                                                                    "Recovered": recovered,
                                                                                                    "CaseType": case_type
                                                                                                }
    print(f"\n=========> Number of susceptible nodes over ALL the days (Place Type: {place_type}): {n_susceptible_nodes}\n")
    print(f"\n=========> Number of infected nodes over ALL the days (Place Type: {place_type}): {n_infected_nodes}\n")
    print(f"\n=========> Number of recovered nodes over ALL the days (Place Type: {place_type}): {n_recovered_nodes}\n")

    return n_susceptible_nodes,\
           n_susceptible_nodes_per_day,\
           n_susceptible_nosoc_nodes_per_day,\
           n_infected_nodes,\
           n_infected_nodes_per_day,\
           n_infected_nodes_nosoc_per_day,\
           n_recovered_nodes,\
           n_recovered_nodes_per_day,\
           n_recovered_nodes_nosoc_per_day,\
           place_dict_patients_per_day_with_infections


def get_labels_per_snapshot(
                                covid_cases_DS_df,
                                place_dict_patients_per_day_with_infections,
                                forecast_horizon,
                                recovery_after_14_days,
                                place_type='Ward',
                                only_nosocomial_for_pred=True
                            ):
    # Computing the labels for each snapshot
    # Getting the labels
    place_nodes_features_per_day = {}
    n_pos_labels = 0
    n_pos_labels_nodes_per_day = {}
    n_neg_labels = 0
    n_neg_labels_nodes_per_day = {}
    n_invalid_labels = 0
    n_invalid_labels_nodes_per_day = {}

    for place in tqdm(place_dict_patients_per_day_with_infections):
        # For place features
        if (place not in place_nodes_features_per_day):
            place_nodes_features_per_day[place] = {}
        # Iterating over the days
        for date_str in place_dict_patients_per_day_with_infections[place]:
            # Counts
            if (date_str not in n_pos_labels_nodes_per_day):
                n_pos_labels_nodes_per_day[date_str] = 0
            if (date_str not in n_neg_labels_nodes_per_day):
                n_neg_labels_nodes_per_day[date_str] = 0
            if (date_str not in n_invalid_labels_nodes_per_day):
                n_invalid_labels_nodes_per_day[date_str] = 0
            # Transform to datetime
            date = pd.to_datetime(date_str).date()
            for patient_idx in range(len(place_dict_patients_per_day_with_infections[place][date_str])):
                # Get the patient ID
                patient_dict = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]
                patient_id = patient_dict['PatientID']
                
                # Metadata retrieval
                pat_data = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == patient_id]
                swab_date = pat_data['swab_date'].item()
                symptoms_date = pat_data['symptoms_date'].item()
                infection_start_date = min(swab_date, symptoms_date).date()
                case_type = pat_data['case_type'].item()

                # Determine CURRENT state at time 'date'
                if (recovery_after_14_days):
                    is_currently_infected = (date >= infection_start_date) and (date <= infection_start_date + timedelta(days=14))
                    is_currently_recovered = (date > infection_start_date + timedelta(days=14))
                else:
                    is_currently_infected = date >= infection_start_date
                    is_currently_recovered = False # Or based on your specific logic
                
                # Getting a list of labels, for each of the following days within the forecast horizon
                labels_per_day_in_forecast_horizon = []
                for day_increase in range(1, forecast_horizon + 1):
                    # Next day
                    next_day = date + timedelta(days=day_increase)
                    
                    # Add label
                    if (next_day >= infection_start_date):
                        if (recovery_after_14_days):
                            if (next_day <= infection_start_date + timedelta(days=14)):
                                state_val = 1 # Infected
                            else:
                                state_val = 2 # Recovered
                        else:
                            TOL_VALID_LABEL = 2 # in days
                            if (next_day <= infection_start_date + timedelta(days=TOL_VALID_LABEL)):
                                state_val = 1
                            else:
                                state_val = -1
                    else:
                        state_val = 0 # Susceptible
                    labels_per_day_in_forecast_horizon.append(state_val)

                # Generate SPECIFIC TRANSITION Label
                # 3. Generate TRAJECTORY-AWARE Transition Label
                if (is_currently_infected):
                    current_state_val = 1
                else:
                    if (is_currently_recovered):
                        current_state_val = 2
                    else:
                        current_state_val = 0

                # Get unique states visited in the window, preserving order
                # e.g., if window is [0, 0, 1, 1, 1, 2, 2], trajectory is [0, 1, 2]
                future_states = [s for s in labels_per_day_in_forecast_horizon if s != -1]
                trajectory = [current_state_val]
                for s in future_states:
                    if (s != trajectory[-1]):
                        trajectory.append(s)

                # Define Specific "Correct" Transition Categories
                # DETAILED TRANSITIONS
                # if (len(trajectory) == 1):
                #     transition_type = 0  # STAY: No change (e.g., S -> S, I -> I)
                # elif (trajectory == [0, 1]):
                #     transition_type = 1  # INFECTION: S -> I (and stayed I)
                # elif (trajectory == [1, 2]):
                #     transition_type = 2  # RECOVERY: I -> R (and stayed R)
                # elif (trajectory == [0, 1, 2]):
                #     transition_type = 3  # FULL CYCLE: S -> I -> R (happened fast)
                # elif (trajectory == [2, 0]):
                #     transition_type = 4  # WANING IMMUNITY: R -> S (if your model allows it)
                # # else:
                # #     transition_type = 5  # IRREGULAR: Anything else (e.g., S -> R skip, or jumps)
                # SIMPLIFIED VERSION
                if (len(trajectory) == 1):
                    transition_type = 0  # STAY: No change (e.g., S -> S, I -> I)
                elif (trajectory == [0, 1]) or (trajectory == [0, 1, 2]):
                    transition_type = 1  # INFECTION: S -> I (and stayed I) or FULL CYCLE: S -> I -> R (happened fast)
                else:
                    transition_type = 2 # Other transitions
                # else:
                #     transition_type = 5  # IRREGULAR: Anything else (e.g., S -> R skip, or jumps)
                # Store both in the dictionary
                place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Label'] = labels_per_day_in_forecast_horizon
                place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['TransitionLabel'] = transition_type

                # 4. Use for Prediction?
                # We only want to predict transitions FOR susceptible patients
                if (is_currently_infected or is_currently_recovered):
                    place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction'] = False
                else:
                    if only_nosocomial_for_pred:
                        place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction'] = (case_type == 'nosocomial')
                    else:
                        place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction'] = True

                # Updating counts for INFECTION RISK PREDICTION
                if (place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction']):
                    if (1 in place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Label']):
                        n_pos_labels_nodes_per_day[date_str] += 1 
                        n_pos_labels += 1
                    elif (sum(place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Label']) == -forecast_horizon):
                        n_invalid_labels_nodes_per_day[date_str] += 1
                        n_invalid_labels += 1
                        place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction'] = False
                    else:
                        n_neg_labels_nodes_per_day[date_str] += 1 
                        n_neg_labels += 1
            
            # Computing colonization pressure
            n_pats_in_place = len(place_dict_patients_per_day_with_infections[place][date_str])
            # Identify patients currently infected at time 't'
            currently_infected_indices = []
            for idx, pat in enumerate(place_dict_patients_per_day_with_infections[place][date_str]):
                swab_date = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == pat['PatientID']]['swab_date'].item()
                symptoms_date = covid_cases_DS_df[covid_cases_DS_df['patient_id'] == pat['PatientID']]['symptoms_date'].item()
                inf_start = min(swab_date, symptoms_date).date()
                if (date >= inf_start):
                    currently_infected_indices.append(idx)
            n_inf_total = len(currently_infected_indices)
            # Compute PLACE-level Colonization Pressure
            # This is the global state of the ward at time 't'
            if n_pats_in_place > 0:
                place_col_pres = n_inf_total / n_pats_in_place
            else:
                place_col_pres = 0.0
            if (date_str not in place_nodes_features_per_day[place]):
                place_nodes_features_per_day[place][date_str] = {}
            place_nodes_features_per_day[place][date_str]['NPatients'] = n_pats_in_place
            place_nodes_features_per_day[place][date_str]['ColonizationPressure'] = place_col_pres
            # Compute PATIENT-level Colonization Pressure (Leave-One-Out)
            for idx in range(n_pats_in_place):
                if (n_pats_in_place > 1):
                    # Subtract current patient from numerator if they are infected
                    n_inf_others = n_inf_total - 1 if idx in currently_infected_indices else n_inf_total
                    pat_col_pres = n_inf_others / (n_pats_in_place - 1)
                else:
                    pat_col_pres = 0.0
                    
                place_dict_patients_per_day_with_infections[place][date_str][idx]['ColonizationPressure'] = pat_col_pres

    # Number of samples per class
    print(f"\nNumber of TOTAL positive (future infection) labels (Place type: {place_type}): {n_pos_labels}\n")
    print(f"\nNumber of TOTAL negative (no future infection) labels (Place type: {place_type}): {n_neg_labels}\n")
    print(f"\nNumber of TOTAL invalid (dates after infection date) labels (Place type: {place_type}): {n_invalid_labels}\n")

    return n_pos_labels,\
           n_pos_labels_nodes_per_day,\
           n_neg_labels,\
           n_neg_labels_nodes_per_day,\
           n_invalid_labels,\
           n_invalid_labels_nodes_per_day,\
           place_nodes_features_per_day



def get_data_snapshots_list_for_PyGeo(
                                        place_dict_patients_per_day_with_infections,
                                        place_nodes_features_per_day,
                                        place_dict_patients_per_day,
                                        n_pats_per_place_per_day,
                                        forecast_horizon,
                                        fully_connect_patients_same_place,
                                        uniform_weights_patients_to_patient_edge,
                                        place_type='Ward'
                                     ):
    # Getting the snapshots of the hospital per day
    # Getting all possible dates
    all_dates_str = []
    for place in tqdm(place_dict_patients_per_day_with_infections):
        for date_str in place_dict_patients_per_day_with_infections[place]:
            all_dates_str.append(date_str)
    all_dates_str = np.unique(all_dates_str)
    # Sorting by increasing order
    all_dates_datetimes = [pd.to_datetime(tmp_date) for tmp_date in all_dates_str]
    sorted_idx = np.argsort(all_dates_datetimes)
    all_dates_str = [all_dates_str[i] for i in sorted_idx]

    # State tracker
    # Stores {patient_id: {'last_state': [S, I, R], 'duration': int}}
    patient_state_tracker = {}

    # Variables to store for the graph
    nodes_types = ["Patient", "Place"]
    #edges_types = ["Patient-Patient", "Patient-Place", "Place-Patient"]
    edges_types = [('Patient', 'Place'), ('Place', 'Patient')]
    if (fully_connect_patients_same_place):
        edges_types.append(('Patient', 'Patient'))
    
    list_edge_index_dicts = [{edge_type: [] for edge_type in edges_types} for _ in range(len(all_dates_str))]
    list_edge_weight_dicts = [{edge_type: [] for edge_type in edges_types} for _ in range(len(all_dates_str))]
    list_node_feature_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]
    list_node_target_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]
    list_node_target_transitions_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]
    list_node_timestamps_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]
    list_nodes_ids_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]
    other_attributes = {"list_nodes_use_for_pred_dicts": [{node_type: [] for node_type in nodes_types} for _ in range(len(all_dates_str))]}

    # Creating the snapshots
    for step_ID in tqdm(range(len(all_dates_str))): 
        date_str = all_dates_str[step_ID]
        
        # --- PRE-PASS: Update Durations for this day ---
        # (Optional: If a patient disappears for a day, you might want to reset them)
        current_day_patients = set()

        for place in place_dict_patients_per_day_with_infections:
            if (date_str in place_dict_patients_per_day_with_infections[place]):
                # Add places nodes and features
                # Node ID
                list_nodes_ids_dicts[step_ID]["Place"].append(place)
                # Timestamp
                list_node_timestamps_dicts[step_ID]["Place"].append(date_str)
                # Features
                n_pats_in_place = place_nodes_features_per_day[place][date_str]['NPatients']
                col_pres_place = place_nodes_features_per_day[place][date_str]['ColonizationPressure']
                place_features = np.array([n_pats_in_place, col_pres_place])
                list_node_feature_dicts[step_ID]['Place'].append(place_features)
                other_attributes['list_nodes_use_for_pred_dicts'][step_ID]["Place"].append(False)
                
                # Add patients nodes
                for patient_idx in range(len(place_dict_patients_per_day_with_infections[place][date_str])):
                    # Nodes IDs
                    patient_ID = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['PatientID']
                    list_nodes_ids_dicts[step_ID]["Patient"].append(patient_ID)
                    # Timestamps
                    list_node_timestamps_dicts[step_ID]["Patient"].append(date_str)

                    # SIR state logic
                    is_infected = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Infection']
                    is_recovered = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Recovered']
                    
                    if ( (is_infected == 0) and (is_recovered == 0) ): # Susceptible patient
                        current_state = [1, 0, 0]
                    elif ( (is_infected == 1) and (is_recovered == 0) ): # Infected patient
                        current_state = [0, 1, 0]
                    elif ( (is_infected == 0) and (is_recovered == 1) ): # Recovered patient
                        current_state = [0, 0, 1]
                    else: # Invalid state
                        current_state = [0, 0, 0]

                    # --- NEW: Logic for Duration in State ---
                    if (patient_ID not in patient_state_tracker):
                        # First time seeing this patient
                        patient_state_tracker[patient_ID] = {'last_state': current_state, 'duration': 1}
                    else:
                        if (patient_state_tracker[patient_ID]['last_state'] == current_state):
                            # State is the same, increment duration
                            patient_state_tracker[patient_ID]['duration'] += 1
                        else:
                            # State changed, reset duration to 1
                            patient_state_tracker[patient_ID]['last_state'] = current_state
                            patient_state_tracker[patient_ID]['duration'] = 1
                    
                    current_duration = patient_state_tracker[patient_ID]['duration']
                    
                    # Features: Added current_duration to the array
                    col_pres = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['ColonizationPressure']
                    
                    # vector: [Infection_Flag, S, I, R, ColPres, Duration]
                    patient_features = np.array([
                                                    is_infected, 
                                                    current_state[0], current_state[1], current_state[2], 
                                                    col_pres,
                                                    current_duration  
                                                ])
                    
                    list_node_feature_dicts[step_ID]["Patient"].append(patient_features)
                    
                    # Update indices for documentation/downstream use
                    patients_features_names_idx = {
                        'Infection': 0,
                        'State': [1, 4],
                        'ColonizationPressure': 4,
                        'DurationInState': 5
                    }
                    
                    # Label
                    if ('Label' in place_dict_patients_per_day_with_infections[place][date_str][patient_idx]):
                        target_patient = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['Label']
                    else:
                        target_patient = [-1 for _ in range(forecast_horizon)]
                    list_node_target_dicts[step_ID]["Patient"].append(target_patient)
                    # Use node for prediction?
                    if ('UseForPrediction' in place_dict_patients_per_day_with_infections[place][date_str][patient_idx]):
                        use_for_pred = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['UseForPrediction']
                    else:
                        use_for_pred = False
                    # Transition label
                    if ('TransitionLabel' in place_dict_patients_per_day_with_infections[place][date_str][patient_idx]):
                        target_transition_patient = place_dict_patients_per_day_with_infections[place][date_str][patient_idx]['TransitionLabel']
                    else:
                        target_transition_patient = [-1 for _ in range(forecast_horizon)]
                    list_node_target_transitions_dicts[step_ID]["Patient"].append(target_transition_patient)
                    other_attributes['list_nodes_use_for_pred_dicts'][step_ID]["Patient"].append(use_for_pred)

                    # Creating the edges
                    # Between patients if asked
                    if (fully_connect_patients_same_place):
                        for partient_bis_idx in range(len(place_dict_patients_per_day_with_infections[place][date_str])):
                            patient_bis_ID = place_dict_patients_per_day_with_infections[place][date_str][partient_bis_idx]['PatientID']
                            if (patient_ID != patient_bis_ID):
                                list_edge_index_dicts[step_ID][('Patient', 'Patient')].append([patient_ID, patient_bis_ID])
                                weight = (1.0/n_pats_in_place) if uniform_weights_patients_to_patient_edge else 1.0
                                list_edge_weight_dicts[step_ID][('Patient', 'Patient')].append(weight)

                    # Creating edges between the nodes patients in the place and the node corresponding to that place
                    # For edges between Places and Patients we need to add the edges in both directions for undirected graph
                    list_edge_index_dicts[step_ID][('Patient', 'Place')].append([patient_ID, place])
                    list_edge_index_dicts[step_ID][('Place', 'Patient')].append([place, patient_ID])
                    list_edge_weight_dicts[step_ID][('Patient', 'Place')].append(np.array([1.0]))
                    list_edge_weight_dicts[step_ID][('Place', 'Patient')].append(np.array([1.0]))

    # Test if there is a coherence between the number of patients per day in place_dict_patients_per_day and n_pats_per_place_per_day
    for step_ID in tqdm(range(len(list_node_timestamps_dicts))):
        # Getting the date
        date_str = list_node_timestamps_dicts[step_ID]['Patient'][0]
        datetime_date = pd.to_datetime(date_str).date()
        # Getting the place
        tmp_n_pats_per_place = {}
        for place, pat in list_edge_index_dicts[step_ID][('Place', 'Patient')]:
            if (place not in tmp_n_pats_per_place):
                tmp_n_pats_per_place[place] = 0
            tmp_n_pats_per_place[place] += 1
        # Iterating over the places
        for place in tmp_n_pats_per_place:
            if (datetime_date in n_pats_per_place_per_day[place]):
                if (tmp_n_pats_per_place[place] != n_pats_per_place_per_day[place][datetime_date]):
                    raise RuntimeError(f"The number of patients in place_dict_patients_per_day ({len(place_dict_patients_per_day[place][date_str])}) for place {place} and day {date_str} is not consistent with the one in n_pats_per_place_per_day ({n_pats_per_place_per_day[place][datetime_date]})")
    print("\n=========> Test passed <=========\n")

    # IDx for the places features
    places_features_names_idx = {'NPatients': 0, 'ColonizationPressure': 1}
    
    return list_edge_index_dicts,\
           list_edge_weight_dicts,\
           list_node_feature_dicts,\
           list_node_target_dicts,\
           list_node_target_transitions_dicts,\
           list_node_timestamps_dicts,\
           list_nodes_ids_dicts,\
           other_attributes,\
           places_features_names_idx,\
           patients_features_names_idx

# Create HDF5 file
def create_HDF5_file(
                        h5_fn,
                        list_edge_index_dicts,
                        list_edge_weight_dicts,
                        list_node_feature_dicts,
                        list_node_target_dicts,
                        list_node_target_transitions_dicts,
                        list_node_timestamps_dicts,
                        list_nodes_ids_dicts,
                        list_nodes_use_for_pred_dicts,
                        str_start_date=None,
                        str_end_date=None
                    ):
    """
        Creates an HDF5 representing the data for the graph construction
        in Pytorch Geometric.

        Parameters:
        -----------
        h5_fn: str
            Path where the HDF5 should be created.
        
        Returns:
        --------
        h5_fn: str
            Path where the HDF5 is effectively created.
            If the input h5_fn already exists, it is not
            overwritten, but a suffix is added to the 
            new file.
    """
    # File name
    i_file = 0
    h5_fn = h5_fn.split(".hdf5")[0] + '_'
    while (os.path.exists(h5_fn + str(i_file) + '.hdf5')):
        i_file += 1
    h5_fn = h5_fn + str(i_file) + '.hdf5'
    
    # Creating the file
    hdf5_file = h5py.File(h5_fn, "w")

    # Getting forecast horizon value
    if ('ForecastHorizon' in h5_fn):
        forecast_hor = int(h5_fn.split('ForecastHorizon-')[-1].split('_')[0])

    # Getting recovery after 14 days bool value
    if ('RecoveryAfter14Days-' in h5_fn):
        rec_after_14_days = h5_fn.split('RecoveryAfter14Days-')[-1].split('_')[0]
        if (rec_after_14_days.lower() == 'true'):
            rec_after_14_days = True
        else:
            rec_after_14_days = False
    else:
        rec_after_14_days = False

    # Using Only Nosocomial Patients for prediciton?
    if ('OnlyNosocForPred-' in h5_fn):
        only_nosocomial_for_pred = h5_fn.split('OnlyNosocForPred-')[-1].split('_')[0]
        if (only_nosocomial_for_pred.lower() == 'true'):
            only_nosocomial_for_pred = True
        else:
            only_nosocomial_for_pred = False
    else:
        only_nosocomial_for_pred = False

    # Caracteristics of the DS
    is_fully_connected = h5_fn.split('/')[-1].split('_')[4].split('-')[1]
    if (is_fully_connected.lower() == 'true'):
        is_fully_connected = True
    elif (is_fully_connected.lower() == 'false'):
        is_fully_connected = False
    else:
        raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_fully_connected})\n")
    is_uniform_patTopat_edges = h5_fn.split('/')[-1].split('_')[5].split('-')[1].split('.hdf5')[0]
    if (is_uniform_patTopat_edges.lower() == 'true'):
        is_uniform_patTopat_edges = True
    elif (is_uniform_patTopat_edges.lower() == 'false'):
        is_uniform_patTopat_edges = False
    else:
        raise RuntimeError(f"\nFullyConnected should be True or False (current value {is_uniform_patTopat_edges})\n")
    
    # Creating the different main groups
    main_groups = {
                    "EdgeIndexDicts": hdf5_file.create_group("EdgeIndexDicts"),
                    "EdgeFeaturesDicts": hdf5_file.create_group("EdgeFeaturesDicts"),
                    "NodeFeaturesDicts": hdf5_file.create_group("NodeFeaturesDicts"),
                    "NodeTargetsDict": hdf5_file.create_group("NodeTargetsDict"),
                    "NodeTargetsTransitionsDict": hdf5_file.create_group("NodeTargetsTransitionsDict"),
                    "NodeTimestampsDicts": hdf5_file.create_group("NodeTimestampsDicts"),
                    "NodesIdsDicts": hdf5_file.create_group("NodesIdsDicts"),
                  }
    py_murcia_data = {
                        "EdgeIndexDicts": list_edge_index_dicts,
                        "EdgeFeaturesDicts": list_edge_weight_dicts,
                        "NodeFeaturesDicts": list_node_feature_dicts,
                        "NodeTargetsDict": list_node_target_dicts,
                        "NodeTargetsTransitionsDict": list_node_target_transitions_dicts,
                        "NodeTimestampsDicts": list_node_timestamps_dicts,
                        "NodesIdsDicts": list_nodes_ids_dicts,
                     }

    # Creating a dictionary to convert Patients and Places IDs from str to integers
    all_pats_IDs = []
    all_places_IDs = []
    for step_ID in tqdm(range(len(list_nodes_ids_dicts))):
        for pat_str_ID in list_nodes_ids_dicts[step_ID]['Patient']:
            all_pats_IDs.append(pat_str_ID)
        for place_str_ID in list_nodes_ids_dicts[step_ID]['Place']:
            all_places_IDs.append(place_str_ID)
    all_pats_IDs = np.unique(all_pats_IDs)
    all_places_IDs = np.unique(all_places_IDs)
    dict_intID_to_strID_mapping = {int_ID: str(all_pats_IDs[int_ID]) for  int_ID in range(len(all_pats_IDs))}
    dict_strID_to_intID_mapping = {str(all_pats_IDs[int_ID]): int_ID for  int_ID in range(len(all_pats_IDs))}
    for int_ID in range(len(all_places_IDs)):
        current_int_ID = len(all_pats_IDs) + int_ID
        dict_intID_to_strID_mapping[current_int_ID] = str(all_places_IDs[int_ID])
        dict_strID_to_intID_mapping[str(all_places_IDs[int_ID])] = current_int_ID

    # Storing the mappings
    dict_intID_to_strID_mapping_fn = '/'.join(h5_fn.split('/')[:-1]) + f"/Mapping_IntIDs_to_StrIDs_ForecastHorizon-{forecast_hor}_RecoveryAfter14Days-{rec_after_14_days}_FullyConnected-{is_fully_connected}_UniformPatToPatEdges-{is_uniform_patTopat_edges}_StartDate-{str_start_date}_EndDate-{str_end_date}_OnlyNosocForPred-{only_nosocomial_for_pred}_{i_file}.yaml"
    
    with open(dict_intID_to_strID_mapping_fn, 'w') as file:
        yaml.dump(dict_intID_to_strID_mapping, file, sort_keys=False)
    dict_strID_to_intID_mapping_fn = '/'.join(h5_fn.split('/')[:-1]) + f"/Mapping_StrIDs_to_IntIDs_ForecastHorizon-{forecast_hor}_RecoveryAfter14Days-{rec_after_14_days}_FullyConnected-{is_fully_connected}_UniformPatToPatEdges-{is_uniform_patTopat_edges}_StartDate-{str_start_date}_EndDate-{str_end_date}_OnlyNosocForPred-{only_nosocomial_for_pred}_{i_file}.yaml"
    with open(dict_strID_to_intID_mapping_fn, 'w') as file:
        yaml.dump(dict_strID_to_intID_mapping, file, sort_keys=False)

    # Iterating over the steps
    for step_ID in tqdm(range(len(list_nodes_ids_dicts))):
        for main_group_name in main_groups:
            step_group = main_groups[main_group_name].create_group(str(step_ID))
            for key in py_murcia_data[main_group_name][step_ID]:
                if (len(key) == 2): # We have an edge
                    dataset_name = f"{key[0]}-{key[1]}"
                else: # We have a node
                    dataset_name = key
                # Creating dataset
                if (type(py_murcia_data[main_group_name][step_ID][key]) == list):
                    data = np.array(py_murcia_data[main_group_name][step_ID][key])
                else:
                    data = py_murcia_data[main_group_name][step_ID][key]

                # Converting the str indinces of the nodes and edges into ints
                if (main_group_name == 'NodesIdsDicts'):
                    new_data = []
                    for node_str_ID in data:
                        node_int_ID = dict_strID_to_intID_mapping[node_str_ID]
                        new_data.append(node_int_ID)
                    data = np.array(new_data)
                elif (main_group_name == 'EdgeIndexDicts'):
                    new_data = []
                    for edge in data:
                        new_edge = [ dict_strID_to_intID_mapping[edge[0]], dict_strID_to_intID_mapping[edge[1]] ]
                        new_data.append(new_edge)
                    data = np.array(new_data).T # Necessary to be of shape (2, n_edges) for Pytorch Geometric

                # Create variable-length UTF-8 string dtype
                if (data.dtype == '<U9') or (data.dtype == '<U10'):
                    # Creating string type
                    new_dtype = h5py.string_dtype(encoding='utf-8')
                    # Convert your array to this dtype
                    data = data.astype(new_dtype)

                step_group.create_dataset(dataset_name, data=data)
                
    # Close file
    hdf5_file.close()

    return h5_fn


# Pre-process dataset
def preprocessHUGCovidDataset(
                                h5_file,
                                parameters_exp,
                                patients_features_names_idx,
                                places_features_names_idx,
                                data_splits_thresholds_IDs_dict,
                                prefix_paths=""
                            ):
        """
            Pre-process a HUG COVID datasets (real data)
            It creates an dictionary attribute. This dictionary containing 
            the following keys: EdgeIndexDicts, EdgeFeaturesDicts, 
            NodeFeaturesDicts, NodeTargetsDict, NodeTimestampsDicts,
            NodesIdsDicts, and NodesTagUseForPrediction. Each value is a 
            list containing the data for each time step in the simulation.
            It also creates a non-normalized version of the train, val, 
            and test datasets.
        """           
        # Get the lists of dicts to create the Murcia Heterogeneous Dataset
        #n_steps = len(self.h5_file["EdgeFeaturesDicts"])
        n_steps = len(h5_file["EdgeFeaturesDicts"])
        #self.data = {
        data = {
                    "EdgeIndexDicts": [None for _ in range(n_steps)],
                    "EdgeFeaturesDicts": [None for _ in range(n_steps)],
                    "NodeFeaturesDicts": [None for _ in range(n_steps)],
                    "NodeTargetsDict": [None for _ in range(n_steps)],
                    "NodeTargetsTransitionsDict": [None for _ in range(n_steps)],
                    "NodeTimestampsDicts": [None for _ in range(n_steps)],
                    "NodesIdsDicts": [None for _ in range(n_steps)],
                }
        #for main_group in self.data:
        for main_group in data:
            #for str_step_ID in tqdm(self.h5_file[main_group]):
            for str_step_ID in tqdm(h5_file[main_group]):
                step_ID = int(str_step_ID)
                #keys = list(self.h5_file[main_group][str_step_ID])
                keys = list(h5_file[main_group][str_step_ID])
                #self.data[main_group][step_ID] = {}
                data[main_group][step_ID] = {}
                for key in keys:
                    if (len(key.split('-')) == 2):
                        new_key = tuple(key.split('-'))
                    else:
                        new_key = key
                    #self.data[main_group][step_ID][new_key] = self.h5_file[main_group][str_step_ID][key][:]
                    data[main_group][step_ID][new_key] = h5_file[main_group][str_step_ID][key][:]

        # Close HDF5 file
        #self.h5_file.close()
        h5_file.close()

        # Separating into training and testing data
        # Train
        if (parameters_exp["predict_state_transitions"]):
            train_other_attributes = {
                                        "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                    }
        else:
            train_other_attributes = {}
        train_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                            **train_other_attributes
                                                        )

        # Val
        if (parameters_exp["predict_state_transitions"]):
            val_other_attributes = {
                                        "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                    }
        else:
            val_other_attributes = {}
        if (data_splits_thresholds_IDs_dict['init_val_step'] is not None) and (data_splits_thresholds_IDs_dict['last_val_step'] is not None):
            val_ds = DynamicHeteroGraphTemporalSignal(
                                                        edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        **val_other_attributes
                                                     )
        else:
            val_ds = []

        # Test
        if (data_splits_thresholds_IDs_dict['init_test_step'] is None):
            raise RuntimeError("init_test_step CANNOT be None")
        if (data_splits_thresholds_IDs_dict['last_test_step'] is None):
            last_test_step = len(data["EdgeIndexDicts"])
        else:
            last_test_step = data_splits_thresholds_IDs_dict['last_test_step']
        if (parameters_exp["predict_state_transitions"]):
            test_other_attributes = {
                                        "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                    }
        else:
            test_other_attributes = {}
        test_ds = DynamicHeteroGraphTemporalSignal(
                                                            edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                            **test_other_attributes
                                                        )
        
        # Normalizing the timestamps and edges weights for more stable training
        if (parameters_exp["normalize_ds"]):
            train_ds, val_ds, test_ds = normalizeDataset(data, parameters_exp, patients_features_names_idx, places_features_names_idx, data_splits_thresholds_IDs_dict)

        # Split percentages
        n_total_samples = len(train_ds) + len(val_ds) + len(test_ds)
        print(f"\n\n=========>DATASET CREATED: {100*len(train_ds)/n_total_samples}% ({len(train_ds)} samples) for training, {100*len(val_ds)/n_total_samples}% ({len(val_ds)} samples) for validation, {100*len(test_ds)/n_total_samples}% ({len(test_ds)} samples) for testing\n\n")

        return train_ds, val_ds, test_ds


def normalizeDataset(
                        data,
                        parameters_exp,
                        patients_features_names_idx,
                        places_features_names_idx,
                        data_splits_thresholds_IDs_dict
                    ):
    # Getting the TRAINING features
    patients_training_features = {feat_name: [] for feat_name in patients_features_names_idx}
    places_training_features = {feat_name: [] for feat_name in places_features_names_idx}
    train_timestamps = []
    for train_step_ID in range(data_splits_thresholds_IDs_dict['init_train_step'], data_splits_thresholds_IDs_dict['last_train_step']):
        # Timestamp
        train_timestamps.append( pd.to_datetime(data['NodeTimestampsDicts'][train_step_ID]['Patient'][0].decode('utf-8')).date() )
        # Patients features
        for feat_name in patients_features_names_idx:
            feature_idx = patients_features_names_idx[feat_name]
            if (type(feature_idx) == list):
                init_feat_idx, last_feat_idx = feature_idx[0], feature_idx[1]
                patients_training_features[feat_name].extend(data['NodeFeaturesDicts'][train_step_ID]['Patient'][:, init_feat_idx:last_feat_idx].tolist()) 
            else:
                patients_training_features[feat_name].extend(data['NodeFeaturesDicts'][train_step_ID]['Patient'][:, feature_idx].tolist())

        # Places features 
        for feat_name in places_features_names_idx:
            feature_idx = places_features_names_idx[feat_name]
            places_training_features[feat_name].extend(data['NodeFeaturesDicts'][train_step_ID]['Place'][:, feature_idx].tolist())     
    patients_training_features = {feat_name: np.array(patients_training_features[feat_name]) for feat_name in patients_features_names_idx}
    places_training_features = {feat_name: np.array(places_training_features[feat_name]) for feat_name in places_features_names_idx}
   
    # Creating the scalers
    # For patients features
    # IMP0RTANT: For the feature of the patients, no need for normalization (State (binary) and colonization pressure (between 0 and 1))

    # For places features
    # IMPORTANT: Only normalize the number of patients in the ward (the colonization pressure is already between 0 and 1 so no need for normlaization)
    scaler_n_patients_place = StandardScaler().fit(places_training_features['NPatients'].reshape(-1, 1))
    #scaler_n_patients_place = MinMaxScaler().fit(places_training_features['NPatients'].reshape(-1, 1))

    # For timesteps
    # Convert in days since the minimal day
    first_day = min(train_timestamps)
    n_days_since_begining = np.array([ (tmp_date - first_day).days for tmp_date in train_timestamps ] )
    #timesteps_scaler = MinMaxScaler().fit(n_days_since_begining.reshape(-1, 1))
    timesteps_scaler = StandardScaler().fit(n_days_since_begining.reshape(-1, 1))
    
    # Applying scalers to features
    n_steps = len(data["NodeFeaturesDicts"])
    for step_ID in range(n_steps):
        # NPatients for a ward normalization
        feature_idx = places_features_names_idx['NPatients']
        data_to_scale = deepcopy(data["NodeFeaturesDicts"][step_ID]['Place'][:, feature_idx])
        scaled_feature = scaler_n_patients_place.transform(data_to_scale.reshape(-1, 1)).reshape(data_to_scale.shape)
        data["NodeFeaturesDicts"][step_ID]['Place'][:, feature_idx] = deepcopy(scaled_feature)

        # Normalizing the timestamps
        for node_type in ['Patient', 'Place']:
            timesteps_to_normalize = deepcopy(data['NodeTimestampsDicts'][step_ID][node_type])
            n_days_since_begining = np.array([ (pd.to_datetime(tmp_date.decode('utf-8')).date() - first_day).days for tmp_date in timesteps_to_normalize ] )
            data['NodeTimestampsDicts'][step_ID][node_type] = timesteps_scaler.transform(n_days_since_begining.reshape(-1, 1)).reshape(n_days_since_begining.shape)

        # Log transformation for duration of state of a patient
        dur_in_state_feature_idx = patients_features_names_idx['DurationInState']
        data["NodeFeaturesDicts"][step_ID]['Patient'][:, dur_in_state_feature_idx] = np.log1p(data["NodeFeaturesDicts"][step_ID]['Patient'][:, dur_in_state_feature_idx])

    # Creating the normalized train and test datasets
    # Train
    if (parameters_exp["predict_state_transitions"]):
        train_other_attributes = {
                                    "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                }
    else:
        train_other_attributes = {}
    train_ds = DynamicHeteroGraphTemporalSignal(
                                                    edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_train_step']:data_splits_thresholds_IDs_dict['last_train_step']],
                                                    **train_other_attributes
                                                )
    # Val
    if (parameters_exp["predict_state_transitions"]):
        val_other_attributes = {
                                    "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                }
    else:
        val_other_attributes = {}
    if (data_splits_thresholds_IDs_dict['init_val_step'] is not None) and (data_splits_thresholds_IDs_dict['last_val_step'] is not None):
        val_ds = DynamicHeteroGraphTemporalSignal(
                                                        edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_val_step']:data_splits_thresholds_IDs_dict['last_val_step']],
                                                        **val_other_attributes
                                                    )
    else:
        val_ds = []

    # Test
    if (data_splits_thresholds_IDs_dict['last_test_step'] is None):
        last_test_step = len(data["EdgeIndexDicts"])
    else:
        last_test_step = data_splits_thresholds_IDs_dict['last_test_step']
    if (parameters_exp["predict_state_transitions"]):
        test_other_attributes = {
                                    "transitions_targets": data["NodeTargetsTransitionsDict"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                }
    else:
        test_other_attributes = {}
    test_ds = DynamicHeteroGraphTemporalSignal(
                                                        edge_index_dicts=data["EdgeIndexDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        edge_weight_dicts=data["EdgeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        feature_dicts=data["NodeFeaturesDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        target_dicts=data["NodeTargetsDict"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        timestamps=data["NodeTimestampsDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        ids=data["NodesIdsDicts"][data_splits_thresholds_IDs_dict['init_test_step']:last_test_step],
                                                        **test_other_attributes
                                                    )


    return train_ds, val_ds, test_ds




