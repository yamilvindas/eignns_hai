"""
    Data exploration and analysis for the Murcia dataset
"""
import os
import csv
import ast
import argparse
import h5py
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from copy import deepcopy
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from tqdm import tqdm
import seaborn as sns
sns.set_style("darkgrid")

import torch

from src.DataManipulation.DynamicHeteroGraphTemporalSignal import DynamicHeteroGraphTemporalSignal
from src.Utils.tools import one_hot_encoding_np


#====================================================================================================#
#====================================================================================================#
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20, 15) # To increase the size of the plots
params = {
            'axes.titlesize': 20,      # Title font size
            'axes.labelsize': 20,      # X and Y axis labels
            'xtick.labelsize': 20,     # X tick labels
            'ytick.labelsize': 20,     # Y tick labels
            'legend.fontsize': 20      # Legend font size
        }
plt.rcParams.update(params)



#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Mapping of the states
SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
MAPPING_STATES = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
INV_MAPPING_STATES = {v: k for k, v in MAPPING_STATES.items()}

# Possible values of different variables for one-hot encoding
# FOR SERVICES
POSSIBLE_SERVICES = ["ER", "ICU", "Radiology", "Surgery", "Ward"]
one_hot_df_services = pd.get_dummies(POSSIBLE_SERVICES, dtype=np.uint8)
ONE_HOT_ENC_SERVICES = {service: one_hot_df_services[service].to_numpy() for service in one_hot_df_services.columns}
INV_ONE_HOT_ENC_SERVICES = {tuple(ONE_HOT_ENC_SERVICES[key]):key for key in ONE_HOT_ENC_SERVICES}

# Encoding possible states
POSSIBLE_PATIENT_STATES = list(MAPPING_STATES.keys())
one_hot_df_patient_states = pd.get_dummies(POSSIBLE_PATIENT_STATES, dtype=np.uint8)
ONE_HOT_PATIENT_STATES = {state: one_hot_df_patient_states[state].to_numpy() for state in one_hot_df_patient_states.columns}
INV_ONE_HOT_PATIENT_STATES = {tuple(ONE_HOT_PATIENT_STATES[key]):key for key in ONE_HOT_PATIENT_STATES}


#====================================================================================================#
#=======================================Loading locations data=======================================#
#====================================================================================================#
def load_locations_data(locations_data_path='./data/murcia/locations.csv'):
    """
        Creates a dictionary giving the information about the different locations in the
        hospital.

        Parameters:
        -----------
        locations_data_path: str
            Path to the csv file containing the information about the different locations 
            in the hosppital

        Returns:
        --------
        locations_info: dict
            Dictionary giving the information about the different locations in the
            hospital. The keys are the IDs of the locations, and the values are
            also dictionaries with the following keys:
            - init_infected: True if the place is considered as initially infected.
            - ids_adjacent: List IDs of the locations that are adjacent to this location.
            - ids_children: List of IDs of the locations located inside this location (for 
              instance the beds within a room, or the rooms withing a ward).
            - name: String corresponding to the name of the location (for instance, ER_0).
            - id_parent: ID of the parent location where this location is located. If no
              parent, then it is None.
            - floor: Floor in which the location is located (None if not known).
            - service: String corresponding to the name of the service where the location is
              located ('ER', 'ICU', None, 'Radiology', 'Surgery', 'Ward').
        main_parent_places: dict
            Mapping indicating for each location, its main service.
    """
    # Loading the data
    locations_data_rows = []
    with open(locations_data_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            locations_data_rows.append(row)

    # Re-structuring
    columns_names_mapping = {locations_data_rows[0][i].replace(" ", ""):i for i in range(len(locations_data_rows[0]))}
    locations_info = {}
    for location_row in locations_data_rows[1:]: # First row are the names of the columns
        location_ID = int(location_row[columns_names_mapping['id_location']])
        locations_info[location_ID] = {}
        # Is place initially infected?
        locations_info[location_ID]['init_infected'] = ast.literal_eval(location_row[columns_names_mapping['infected']])
        # Adjecent places
        if (location_row[columns_names_mapping['ids_adjacents']].lower() == 'null'):
            locations_info[location_ID]['ids_adjacents'] = None
        else:
            locations_info[location_ID]['ids_adjacents'] = ast.literal_eval(location_row[columns_names_mapping['ids_adjacents']])
        # Places that are inside this current place
        if (location_row[columns_names_mapping['ids_children']].lower() == 'null'):
            locations_info[location_ID]['ids_children'] = None
        else:
            locations_info[location_ID]['ids_children'] = ast.literal_eval(location_row[columns_names_mapping['ids_children']])
        # Name of the place
        locations_info[location_ID]['name'] = location_row[columns_names_mapping['name']]
        # ID of the place where the current place is located
        if (location_row[columns_names_mapping['id_parent']].lower() == 'null'):
            locations_info[location_ID]['id_parent'] = None
        else:
            locations_info[location_ID]['id_parent'] = ast.literal_eval(location_row[columns_names_mapping['id_parent']])
        # Floor where this place is located
        if (location_row[columns_names_mapping['floor']].lower() == 'null'):
            locations_info[location_ID]['floor'] = None
        else:
            locations_info[location_ID]['floor'] = ast.literal_eval(location_row[columns_names_mapping['floor']])
        # Service assigned to this place
        if (location_row[columns_names_mapping['service']].lower() == 'null'):
            locations_info[location_ID]['service'] = None
        else:
            locations_info[location_ID]['service'] = location_row[columns_names_mapping['service']]

        # Creating a list of the main "parent" places, where we consider that if two patients are in a location with the same
        # parent at a given moment, then they are in the same place
        main_parent_places = {}
        for location_ID in locations_info:
            # If they are in the same service, they potentially are in the same place (except for wards , where in addition we need the room)
            if (locations_info[location_ID]['service'] is not None):
                if ('ward' not in locations_info[location_ID]['service'].lower()):
                    main_parent_places[location_ID] = locations_info[location_ID]['service']
            else:
                if ('room' in locations_info[location_ID]['name'].lower()):
                    main_parent_places[location_ID] = locations_info[location_ID]['name'].split('_')[0]

    return locations_info, main_parent_places

#====================================================================================================#
#===============================Loading the movement of patients data===============================#
#====================================================================================================#
def load_movement_data(movement_data_path, locations_info):
    """
        Loads the data about the movement of the patients in the hospital.

        Parameters:
        -----------
        movement_data_path: str
            Path to the csv file containing the information about the movement of patients 
            within in the hospital.

        Returns:
        --------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.
        locations_info: dict
            Dictionary giving the information about the different locations in the
            hospital. The keys are the IDs of the locations, and the values are
            also dictionaries with the following keys:
            - init_infected: True if the place is considered as initially infected.
            - ids_adjacent: List IDs of the locations that are adjacent to this location.
            - ids_children: List of IDs of the locations located inside this location (for 
              instance the beds within a room, or the rooms withing a ward).
            - name: String corresponding to the name of the location (for instance, ER_0).
            - id_parent: ID of the parent location where this location is located. If no
              parent, then it is None.
            - floor: Floor in which the location is located (None if not known).
            - service: String corresponding to the name of the service where the location is
              located ('ER', 'ICU', None, 'Radiology', 'Surgery', 'Ward').
        places_list: list
            List of main places IDs in the simulation.
        specific_places_list: list
            List of specific places IDs (mainly beds) in the simulation.
        rooms_ward_mapping: dict
            Dictionary giving for each room ID (keys) the ID of the ward where it is located.
        one_hot_enc_main_places: dict
            Dictionary where the IDs are the places IDs and the values are their respective
            one-hot encodings.
        inv_one_hot_enc_main_places: dict
            Dictionary where the IDs are the one-hot encodings of the places IDs and the values
            the places IDs.
        one_hot_enc_specific_places: dict
            Dictionary where the IDs are the specific places IDs and the values are their respective
            one-hot encodings.
        inv_one_hot_enc_specific_places: dict
            Dictionary where the IDs are the one-hot encodings of the specific places IDs and the values
            the places IDs.
    """
    # Loading the data
    movement_data_rows = []
    with open(movement_data_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            movement_data_rows.append(row)

    # Restructuring the data
    movement_data = {}
    for movement_data_row in movement_data_rows:
        # Getting the step ID
        step_ID = int(movement_data_row[0])
        movement_data[step_ID] = {}
        movement_data[step_ID]["ContaminatedPlaces"] = []
        movement_data[step_ID]["PlacesToClean"] = []

        # Rooms to clean + number of patients in the step
        n_patients_in_step = None
        if ("places" in movement_data_row):
            places_tag_idx = movement_data_row.index("places")
            for place_ID_clean in movement_data_row[places_tag_idx+1:]:
                movement_data[step_ID]["PlacesToClean"].append(int(place_ID_clean))
            n_patients_in_step = len(movement_data_row[1:places_tag_idx]) / 4
            n_patients_in_step_rest = len(movement_data_row[1:places_tag_idx]) % 4
        else:
            n_patients_in_step = len(movement_data_row[1:]) / 4
            n_patients_in_step_rest = len(movement_data_row[1:]) % 4
            
        # Verification that we have an integer number of groups
        if (n_patients_in_step_rest != 0):
            raise RuntimeError(f"\nThe number of patients is not an integer: {n_patients_in_step}\n")
        else:
            n_patients_in_step = int(n_patients_in_step)
        
        # Getting the data for each patient at that step (groups of 4 having Patient_ID, Patient State,  Place ID, Place Contaminated)
        for i in range(n_patients_in_step):
            # Getting the data about the current patient
            patient_ID,\
            patient_state,\
            place_ID,\
            place_contaminated = movement_data_row[4*(i+1)-3: 4*(i+1) + 1]

            # Transforming from str into the right formats
            if ("id" in patient_ID.lower()):
                patient_ID = int(patient_ID.split('_')[0].split('-')[-1])
            else:
                patient_ID = int(patient_ID)
            patient_state = int(patient_state)
            if ("id" in place_ID.lower()):
                place_ID = int(place_ID.split('_')[0].split('-')[-1])
            else:
                place_ID = int(place_ID)

            if (place_contaminated.lower() == "true"):
                place_contaminated = True
            else:
                place_contaminated = False

            # Getting the parent place
            parent_place_ID = locations_info[place_ID]['id_parent']
            if (parent_place_ID is None):
                main_place_ID = place_ID
            else:
                main_place_ID = parent_place_ID

            # Adding the main place to the places seen in the current step
            if (main_place_ID not in movement_data[step_ID]):
                movement_data[step_ID][main_place_ID] = {}

            # Adding the patients to that place at that state
            if (patient_ID not in movement_data[step_ID][main_place_ID]):
                movement_data[step_ID][main_place_ID][patient_ID] = {
                                                                                    "State": int(patient_state),
                                                                                    "PlaceId": place_ID
                                                                                }
            else:
                print(f"\n===> In step {step_ID} place {main_place_ID} has already patient {patient_ID}.\n")

            # Adding the place to the list of contaminated places
            if (place_contaminated) and (place_ID not in movement_data[step_ID]["ContaminatedPlaces"]):
                movement_data[step_ID]["ContaminatedPlaces"].append(place_ID)
            else:
                if (place_contaminated):
                    print(f"\n===> In step {step_ID} place {place_ID} has already been added to the list of contaminated places at that step.\n")

    # Getting the number of unique places
    places_list = []
    for step_ID in movement_data:
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                if (main_place_ID not in places_list):
                    places_list.append(main_place_ID)
    places_list = sorted(places_list)


    # Getting the number of specific places the patients effectively are (mainly beds)
    specific_places_list = []
    for step_ID in movement_data:
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                for patient_ID in movement_data[step_ID][main_place_ID]:
                    specific_place_patient = movement_data[step_ID][main_place_ID][patient_ID]["PlaceId"]
                    if (specific_place_patient not in specific_places_list):
                        specific_places_list.append(specific_place_patient)

    # Rooms and wards mapping
    rooms_ward_mapping = {}
    for place_id in locations_info:
        if (locations_info[place_id]["service"] is not None) and (locations_info[place_id]["service"].lower() == "ward"):
            for room in locations_info[place_id]["ids_children"]:
                if (room not in rooms_ward_mapping):
                    rooms_ward_mapping[room] = place_id
                else:
                    raise RuntimeError(f"\nPROBLEM: Room {room} has already been assigned to a ward (ward {rooms_ward_mapping[room]}).\n")
                
    # One-hot encoding of the main places
    one_hot_df_places = pd.get_dummies(places_list, dtype=np.uint8)
    one_hot_enc_main_places = {place: one_hot_df_places[place].to_numpy() for place in one_hot_df_places.columns}
    inv_one_hot_enc_main_places = {tuple(one_hot_enc_main_places[key]):key for key in one_hot_enc_main_places}

    # One-hot encoding of the specific places
    one_hot_df_specific_places = pd.get_dummies(specific_places_list, dtype=np.uint8)
    one_hot_enc_specific_places = {place: one_hot_df_specific_places[place].to_numpy() for place in one_hot_df_specific_places.columns}
    inv_one_hot_enc_specific_places = {tuple(one_hot_enc_specific_places[key]):key for key in one_hot_enc_specific_places}

    return movement_data, places_list, specific_places_list, rooms_ward_mapping, one_hot_enc_main_places, inv_one_hot_enc_main_places, one_hot_enc_specific_places, inv_one_hot_enc_specific_places


def simulation_information(movement_data, locations_info):
    """
        Gives some basic information about the simulated data (number of days, number of main places, etc.).

        Parameters:
        -----------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.
        locations_info: dict
            Dictionary giving the information about the different locations in the
            hospital. The keys are the IDs of the locations, and the values are
            also dictionaries with the following keys:
            - init_infected: True if the place is considered as initially infected.
            - ids_adjacent: List IDs of the locations that are adjacent to this location.
            - ids_children: List of IDs of the locations located inside this location (for 
              instance the beds within a room, or the rooms withing a ward).
            - name: String corresponding to the name of the location (for instance, ER_0).
            - id_parent: ID of the parent location where this location is located. If no
              parent, then it is None.
            - floor: Floor in which the location is located (None if not known).
            - service: String corresponding to the name of the service where the location is
              located ('ER', 'ICU', None, 'Radiology', 'Surgery', 'Ward').
            
    """
    # Duration of the simulation
    duration_years = len(movement_data)*8 / (24*365.25)
    print(f"\n========> Duration of the simulation: {duration_years} years\n")

    # Getting the number of unique places
    places_list = []
    for step_ID in movement_data:
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                if (main_place_ID not in places_list):
                    places_list.append(main_place_ID)
    places_list = sorted(places_list)

    print(f"\nThere are {len(places_list)} MAIN places in the simulation\n")
    print(f"\nThe IDs of the different places are: {places_list}\n")

    # Getting the number of specific places the patients effectively are (mainly beds)
    specific_places_list = []
    for step_ID in movement_data:
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                for patient_ID in movement_data[step_ID][main_place_ID]:
                    specific_place_patient = movement_data[step_ID][main_place_ID][patient_ID]["PlaceId"]
                    if (specific_place_patient not in specific_places_list):
                        specific_places_list.append(specific_place_patient)

    print(f"\nThere are {len(specific_places_list)} SPECIFIC places in the simulation\n")
    print(f"\nThe IDs of the different SPECIFIC places are: {specific_places_list}\n")   

    # Rooms and wards mapping
    rooms_ward_mapping = {}
    for place_id in locations_info:
        if (locations_info[place_id]["service"] is not None) and (locations_info[place_id]["service"].lower() == "ward"):
            for room in locations_info[place_id]["ids_children"]:
                if (room not in rooms_ward_mapping):
                    rooms_ward_mapping[room] = place_id
                else:
                    raise RuntimeError(f"\nPROBLEM: Room {room} has already been assigned to a ward (ward {rooms_ward_mapping[room]}).\n")
                
    # Intersection between the main and specific places
    intersection = list(set(places_list) & set(specific_places_list))
    print(f"\n\nPlaces that are both in the main and specific list of places: {intersection}")

    return places_list, specific_places_list


def get_patients_per_state_over_time(movement_data, plot_evolution=True, init_step_use=None, last_step_use=None):
    """
        TODO

        Parameters:
        -----------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.
        plot_evolution: bool
            True if want to plot the evolution over time.
        init_step_use: int
            First time step to use to get the data. If None, the first step is the first one
            available.
        last_step_use: int
            Last time step to use to get the data. If None, the first step is the last one
            available.

        Returns:
        --------
        patients_per_state_over_time: dict
            Dictionary where the keys are the possible states (S, E, I, R, D, NS) and the values
            are list containing the total quantity of of patients in that state over time.
    """
    # Number of patients per state in each step
    patients_per_state_over_time = {tmp_state: [0 for _ in movement_data] for tmp_state in range(6)}
    steps_list = sorted(list(movement_data.keys()))
    if (init_step_use is None):
        init_step_use = steps_list[0]
    else:
        assert (init_step_use >= steps_list[0])
    if (last_step_use is None):
        last_step_use = steps_list[-1]
    else:
        if (last_step_use == -1):
            last_step_use = steps_list[-1]
        else:
            assert (last_step_use <= steps_list[-1])
    for step_ID in steps_list:
        if (step_ID >= init_step_use) and (step_ID <= last_step_use):
            for main_place_ID in movement_data[step_ID]:
                if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                    for patient_ID in movement_data[step_ID][main_place_ID]:
                        patient_state = movement_data[step_ID][main_place_ID][patient_ID]['State']
                        patients_per_state_over_time[int(patient_state)][step_ID] += 1

    # Plot evolution
    if (plot_evolution):
        fig = plt.figure()
        for state in patients_per_state_over_time:
            plt.plot(list(range(len(patients_per_state_over_time[state]))), patients_per_state_over_time[state], label=INV_MAPPING_STATES[state])
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Number of individuals")
        plt.show()

    return patients_per_state_over_time

def get_number_contaminated_places_per_step(movement_data, plot_evolution=True):
    """
        Gets a list of the number of infected places at each step.

        Parameters:
        -----------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.

        Returns:
        --------
        n_contaminated_places_per_step: list
            List of the number of infected places at each time step.
    """
    # Number of contaminated places per step
    n_contaminated_places_per_step = [0 for _ in movement_data]
    for step_ID in movement_data:
        n_contaminated_places_per_step[step_ID] = len(movement_data[step_ID]["ContaminatedPlaces"])
    
    # Plot evolution
    if (plot_evolution):
        fig = plt.figure()
        plt.plot(list(range(len(n_contaminated_places_per_step))), n_contaminated_places_per_step)
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Number of individuals")
        plt.show()

    return n_contaminated_places_per_step


def get_number_places_to_clean_per_step(movement_data, plot_evolution=True):
    """
        Gets a list of the number of places to clean at each time step.

        Parameters:
        -----------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.

        Returns:
        --------
        n_places_to_clean_per_step: list
            List of the number of places to clean at each time step.
    """
    # Number of to clean places per step
    n_places_to_clean_per_step = [0 for _ in movement_data]
    for step_ID in movement_data:
        n_places_to_clean_per_step[step_ID] = len(movement_data[step_ID]["PlacesToClean"])
    
    # Plot evolution
    if (plot_evolution):
        fig = plt.figure()
        plt.plot(list(range(len(n_places_to_clean_per_step))), n_places_to_clean_per_step)
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Number of individuals")
        plt.show()

    return n_places_to_clean_per_step

#====================================================================================================#
#=====================================Loading the patients data=====================================#
#====================================================================================================#
def get_patients_data(patients_data_path):
    """
        Loads the data of the patients (mainly static features) in a Pandas
        DataFrame.

        Parameters:
        -----------
        patients_data_path: str
            Path to the csv file containing the data of the patients.

        Returns:
        --------
        patients_data_df: pandas.core.frame.DataFrame
            DataFrame containing the data of the patient.
            It has the following columns:
            - ID
            - Age
            - Gender
            - LoS
            - IncubationDurationInHours
            - InfectionDurationInHours
            - TreatmentDurationInDays
            - Treatment
            - Antibiotic
            - Microorganism
            - AdmissionDay
            - LastDayHospital
            - Died
            - TrueLoS
            - Colonized
            - NonSusceptible
    """
    # Loading as a Pandas DataFrame
    patients_data_df = pd.read_csv(patients_data_path, sep=';', header=None)
    patients_data_df.columns = [
                                    'ID',
                                    'Age',
                                    'Gender',
                                    'LoS',
                                    'IncubationDurationInHours',
                                    'InfectionDurationInHours',
                                    'TreatmentDurationInDays',
                                    'Treatment',
                                    'Antibiotic',
                                    'Microorganism',
                                    'AdmissionDay',
                                    'LastDayHospital',
                                    'Died',
                                    'TrueLoS',
                                    'Colonized',
                                    'NonSusceptible'
                            ]

    # If treatment duration is 1000, then the patient did not receive any treatment.
    # We are going to transform this into None
    patients_data_df.loc[patients_data_df['TreatmentDurationInDays'] == 1000, 'TreatmentDurationInDays'] = None
    
    return patients_data_df


#====================================================================================================#
#=====================================Transformation into graphs=====================================#
#====================================================================================================#
def to_graphs_murcia(movements_data, places_as_nodes=False, fully_connect_patients_same_place=True):
    """
        Converts a movements data from a synthetic dataset generated by
        the Murcia team into graphs for visualization.
        
        Arguments:
        ----------
        movements_data: dict
            Dictionary where the keys are the steps IDs and the values
            are also dictionaries having three types of keys: Place_ID_i 
            corresponding to the ID of the place i, ContaminatedPlaces
            corresponding to a list of the contaminated places, PlacesToClean
            corresponding to the list of places to clean. For the keys
            corresponding to the places IDs, the values are dictionaries
            corresponding to patients that were in that place, and their 
            epidemiological state.
        places_as_nodes: bool
            If True, places are considered as nodes, and patients that
            are in a place at a given time, are linked to that node.
        fully_connect_patients_same_place: bool
            If True, all the patients that are in the same places are
            connected with a weight of 1.
        
        
        Returns:
        --------
        graphs: list
            List of graphs representing the temporal evolution.
    """
    # Creating the list of graphs
    graphs = []
    for step_ID in movements_data:
        # Creating current graph
        G = nx.Graph()

        # Getting the list of nodes
        patients_nodes = []
        places_nodes = []
        for main_place_ID in movements_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):                
                # Node for the place
                if (main_place_ID not in places_nodes):
                    places_nodes.append(f"Place-{main_place_ID}")
                    
                # Node for patients
                for patient_ID in movements_data[step_ID][main_place_ID]:
                    if (patient_ID not in patients_nodes):
                        patients_nodes.append(f"Patient-{patient_ID}")
       
        # Creating nodes
        G.add_nodes_from(patients_nodes)
        if (places_as_nodes):
            G.add_nodes_from(places_nodes)      
                        
        # Creating the edges
        for main_place_ID in movements_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):                
                patients_in_main_place = [f"Patient-{patient_ID}" for patient_ID in movements_data[step_ID][main_place_ID]]
                for patient_i_ID in patients_in_main_place:
                    # Creating edges between all the patients of the place
                    if (fully_connect_patients_same_place) or (not places_as_nodes):
                        for patient_j_ID in patients_in_main_place:
                            if (patient_i_ID != patient_j_ID):
                                G.add_edge(patient_i_ID, patient_j_ID)

                    # Creating edges between the nodes patients in the place and the node corresponding to the place
                    if (places_as_nodes):
                        G.add_edge(patient_i_ID, f"Place-{main_place_ID}")
                    
        # Adding the graph to the list of graphs
        graphs.append(G)
    
    return graphs

def plot_nx_graphs(movement_data, places_as_nodes=True, fully_connect_patients_same_place=True, n_graphs_plot=2):
    """
        Plots the graphs representing the interaction of the patients over 
        the time steps (one time step is one snapshot).

        Parameters:
        -----------
        movement_data: dict
            Dictionary where the keys are the time steps IDs (in slots of 8 hours, i.e. 
            between step i and step i+1 there are 8 hours).
            The values are dictionaries with the following keys:
            - Ids of the main places (wards, surgical rooms, etc.): The values are dictionaries 
              where the keys are patients IDs, and the values are dicts with two keys: 'State'
              and 'PlaceId' which corresponds to the bed ID if the patients is in a bed,
              and None if not.
            - 'ContaminatedPlaces': List of IDs of the places that are contaminated at that time.
            - 'PlacesToClean': List of IDs of the places that are going to be cleaned at the 
              end of the step.
        places_as_nodes: bool
            True if want to include nodes for the places. In that case, all the patients
            in that place are linked to that node.
        fully_connect_patients_same_place: bool
            If True, all the patients that are in the same places are
            connected with a weight of 1.
        n_graphs_plot: int
            Number of graphs to plot (the first ones of the sequence).
    """
    # Graph without the places
    nx_graph = to_graphs_murcia(movement_data, places_as_nodes=places_as_nodes, fully_connect_patients_same_place=fully_connect_patients_same_place)
    
    # Plot
    fig, axs = plt.subplots(1, n_graphs_plot, figsize=(50, 25))
    for d, (ax, g) in enumerate(zip(axs, nx_graph)):
        if (d >= n_graphs_plot):
          print(f"Plotting only the first {n_graphs_plot} graphs")
          break
        ax.set_title("Step {}".format(d+1))
        nx.draw(g, ax=ax, with_labels=True)

    plt.show()


#====================================================================================================#
#======================= Get necessary data to create Pytorch Geometric Graphs =======================#
#====================================================================================================#

def get_features_patient(
                            patients_data_df,
                            movement_data,
                            locations_info,
                            patient_ID,
                            step_ID,
                            one_hot_enc_main_places,
                            one_hot_enc_specific_places
                        ):
    """
        Get features of the patients to be used as features of 
        nodes in a graph

        Parameters:
        -----------
        patients_data_df: pandas.DataFrame
            Data frame containing all the static features of the 
            patients.
        movement_data: dict
            Dictionary containing the movements of the patients
            over time, which can be used to compute dynamic 
            features of patients.
        locations_info: dict
            Dictionary containing information about the differnt 
            places. The values are dictionaries with the following
            keys: 'init_infected', 'ids_adjacents', 'ids_children',
            'name', 'id_parent', 'floor', 'service'.
        patient_ID: int
            Identifier of the patient
        step_ID: int
            Current time step in the simulation.

        Returns:
        --------
        patient_features: np.array
            Array containing the features of the patient
        patient_features_names_idx: dict
            Dict indicating, for each feature, the initial and last indices
            corresponding to that feature in patient_features
    """
    # Variable for the features of the patient
    patient_features = []
    patient_features_names_idx = {}

    # Static features: "Age", "Gender", "AdmissionDay"
    # WE CANNOT USE: "TrueLoS" AS IT GIVES INFORMATION ABOUT THE FUTURE
    static_features_types = ["Age", "Gender", "AdmissionDay"]
    for feat_type_ID in range(len(static_features_types)):
        statict_feature_type = static_features_types[feat_type_ID]
        stat_feature = patients_data_df.loc[patients_data_df["ID"] == patient_ID, statict_feature_type].values[0]
        if (statict_feature_type.lower() == "gender"):
            if (stat_feature.lower() == 'f'):
                stat_feature = 0
            else:
                stat_feature = 1
        patient_features.append(stat_feature)
        patient_features_names_idx[statict_feature_type] = [feat_type_ID, feat_type_ID+1]
    last_idx_features = feat_type_ID+1

    # Rooms and wards mapping
    rooms_ward_mapping = {}
    for place_id in locations_info:
        if (locations_info[place_id]["service"] is not None) and (locations_info[place_id]["service"].lower() == "ward"):
            for room in locations_info[place_id]["ids_children"]:
                if (room not in rooms_ward_mapping):
                    rooms_ward_mapping[room] = place_id
                else:
                    raise RuntimeError(f"\nPROBLEM: Room {room} has already been assigned to a ward (ward {rooms_ward_mapping[room]}).\n")

    # Get the number of patients for the different wards
    patients_per_ward = {}
    for tmp_main_place_ID in movement_data[step_ID]:
        if (tmp_main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
            for tmp_patient_ID in movement_data[step_ID][tmp_main_place_ID]:
                if (locations_info[tmp_main_place_ID]["service"] is None): # then the patient is in a bed in a room of a ward
                    tmp_current_ward = rooms_ward_mapping[tmp_main_place_ID]
                    if (tmp_current_ward not in patients_per_ward):
                        patients_per_ward[tmp_current_ward] = 0
                    patients_per_ward[tmp_current_ward] += 1
                    
    # Dynamic features: LoS in place, Current Place, Current Specific Place (bed or surgery room) Previous Place,
    # Current Service, Number of patients in place, Current State
    # IMPORTANT: IF THE MAIN PLACE OF THE PATIENT IS A ROOM, THEN THE LoS and Number of patients in place is computed using the WHOLE WARD
    los_in_place = 0
    current_place = -1
    specific_patient_place = -1
    previous_place = -1 # IMPORTANT: DEFAULT VALUE THAT I DEFINED; IT IS GOING TO BE TRANSFORMED IN ONE HOT ENCODING WHERE IT IS ALL 0 IF NO PREVIOUS PLACE
    current_service = -1
    n_patients_same_place = -1
    current_state = -1
    for main_place_ID in movement_data[step_ID]:
        for tmp_patient_ID in movement_data[step_ID][main_place_ID]:
            if (patient_ID == tmp_patient_ID):
                # Current place
                current_place = main_place_ID
                if (type(main_place_ID) == int):
                    specific_patient_place = movement_data[step_ID][main_place_ID][tmp_patient_ID]["PlaceId"]
                    current_state = movement_data[step_ID][main_place_ID][tmp_patient_ID]["State"]
                    # Current service
                    current_service = locations_info[main_place_ID]["service"]
                    if (current_service is None): # then the patient is in a bed in a room of a ward
                        current_service = "Ward"
                    # Number of patients
                    if (current_service.lower() != "ward"):
                        n_patients_same_place = len(movement_data[step_ID][main_place_ID])
                    else:
                        current_ward = rooms_ward_mapping[current_place]
                        n_patients_same_place = patients_per_ward[current_ward]
                    break
    if (step_ID > 0):
        # Get the previous place and the history of places the patient has been
        history_places_patient = [None for _ in range(step_ID+1)]
        history_places_patient[-1] = current_place # Last value is current place
        for previous_step_ID in range(step_ID):
            for main_place_ID in movement_data[previous_step_ID]:
                for tmp_patient_ID in movement_data[previous_step_ID][main_place_ID]:
                    if (patient_ID == tmp_patient_ID):
                        # Previous place
                        previous_place = main_place_ID
                        # Add to history of places
                        history_places_patient[previous_step_ID] = previous_place
        
        # Get LoS in place
        los_in_place = 0
        if (current_service.lower() != "ward"):
            for tmp_main_place_ID in reversed(history_places_patient):
                if (tmp_main_place_ID == current_place):
                    los_in_place += 1
                else:
                    break # In this case the patient was changed of place
        else: # LoS in ward
            current_ward = rooms_ward_mapping[current_place]
            for tmp_main_place_ID in reversed(history_places_patient):
                if (tmp_main_place_ID is not None):
                    previous_place_service = locations_info[tmp_main_place_ID]["service"] if locations_info[tmp_main_place_ID]["service"] is not None else "Ward"
                else:
                    previous_place_service = "None"
                if (previous_place_service.lower() == current_service.lower()):
                    previous_ward = rooms_ward_mapping[tmp_main_place_ID]
                    if (previous_ward == current_ward):
                        los_in_place += 1
    if (current_place == -1):
        raise RuntimeError(f"\nPROBLEM: current_place = -1 for patient {patient_ID}")
    if (current_service == -1):
        raise RuntimeError(f"\nPROBLEM: current_service = -1 for patient {patient_ID}")
    if (n_patients_same_place == -1):
        raise RuntimeError(f"\nPROBLEM: n_patients_same_place = -1 for patient {patient_ID}")
        
    patient_features.append(los_in_place)
    patient_features_names_idx["los_in_place"] = [last_idx_features, last_idx_features+1]
    last_idx_features = last_idx_features + 1

    current_place_one_hot = one_hot_enc_main_places[current_place]
    patient_features.extend(current_place_one_hot.tolist())
    patient_features_names_idx["current_place"] = [last_idx_features, last_idx_features+len(current_place_one_hot)]
    last_idx_features = last_idx_features+len(current_place_one_hot)
    
    if (previous_place == -1):
        previous_place_one_hot = np.zeros(len(one_hot_enc_main_places))
    else:
        previous_place_one_hot = one_hot_enc_main_places[previous_place]
    patient_features.extend(previous_place_one_hot.tolist())
    patient_features_names_idx["previous_place"] = [last_idx_features, last_idx_features+len(previous_place_one_hot)]
    last_idx_features = last_idx_features+len(previous_place_one_hot)

    current_service_one_hot = ONE_HOT_ENC_SERVICES[current_service]
    patient_features.extend(current_service_one_hot.tolist())
    patient_features_names_idx["current_service"] = [last_idx_features, last_idx_features+len(current_service_one_hot)]
    last_idx_features = last_idx_features+len(current_service_one_hot)
    
    specific_patient_place_one_hot = one_hot_enc_specific_places[specific_patient_place]
    patient_features.extend(specific_patient_place_one_hot.tolist())
    patient_features_names_idx["specific_patient_place"] = [last_idx_features, last_idx_features+len(specific_patient_place_one_hot)]
    last_idx_features = last_idx_features+len(specific_patient_place_one_hot)
    
    patient_features.append(n_patients_same_place)
    patient_features_names_idx["n_patients_same_place"] = [last_idx_features, last_idx_features+1]
    last_idx_features = last_idx_features + 1

    # Without one hot encoding
    #patient_features.append(current_state)
    #patient_features_names_idx["current_state"] = [last_idx_features, last_idx_features+1]
    #last_idx_features = last_idx_features + 1
    # With one hot encoding
    current_state_one_hot = ONE_HOT_PATIENT_STATES[INV_MAPPING_STATES[current_state]]
    patient_features.extend(current_state_one_hot.tolist())
    patient_features_names_idx["current_state"] = [last_idx_features, last_idx_features+len(current_state_one_hot)]
    last_idx_features = last_idx_features+len(current_state_one_hot)
    
    return np.array(patient_features), patient_features_names_idx


def get_features_places(movement_data, locations_info, place_ID, step_ID):
    """
        Get features of the places to be used as features of 
        nodes in a graph

        Parameters:
        -----------
        movement_data: dict
            Dictionary containing the movements of the patients
            over time, which can be used to compute the features 
            of the places.
        locations_info: dict
            Dictionary containing information about the differnt 
            places. The values are dictionaries with the following
            keys: 'init_infected', 'ids_adjacents', 'ids_children',
            'name', 'id_parent', 'floor', 'service'.
        place_ID: int
            Identifier of the place
        step_ID: int
            Current time step in the simulation.

        Returns:
        --------
        places_features: np.array
            Array containing the features of the places
        places_features_names_idx: dict
            Dict indicating, for each feature, the initial and last indices
            corresponding to that feature in places_features
        
    """
    # Variable for the features of the patient
    places_features = []
    places_features_names_idx = {}

    # Static features: Service
    service = locations_info[place_ID]["service"]
    if (service is None):
        service = "Ward"
    service_one_hot = ONE_HOT_ENC_SERVICES[service]
    places_features.extend(service_one_hot.tolist())
    places_features_names_idx["service"] = [0, len(service_one_hot)]
    last_idx_features = len(service_one_hot)

    # Dynamic features: isContaminated, toClean, nPatients
    isContaminated = int(place_ID in movement_data[step_ID]["ContaminatedPlaces"])
    places_features.append(isContaminated)
    places_features_names_idx["isContaminated"] = [last_idx_features, last_idx_features+1]
    last_idx_features = last_idx_features + 1
    
    toClean = int(place_ID in movement_data[step_ID]["PlacesToClean"])
    places_features.append(toClean)
    places_features_names_idx["toClean"] = [last_idx_features, last_idx_features+1]
    last_idx_features = last_idx_features + 1
    
    nPatients = 0
    for main_place_ID in movement_data[step_ID]:
        if (place_ID == main_place_ID):
            nPatients = len(movement_data[step_ID][place_ID])
    places_features.append(nPatients)
    places_features_names_idx["nPatients"] = [last_idx_features, last_idx_features+1]
    last_idx_features = last_idx_features + 1    

    # TODO: Compute colonization pressure !!!

    return np.array(places_features), places_features_names_idx


def create_lists_nodes_edges(
                                movement_data,
                                locations_info,
                                patients_data_df,
                                one_hot_enc_main_places,
                                one_hot_enc_specific_places,
                                forecast_window_length=1,
                                fully_connect_patients_same_place=True,
                                uniform_weights_patients_to_patient_edge=False
                            ):
    """
        Create the list of nodes ids, nodes features, nodes targets, nodes
        timestamps, edges indices and edges weights necessary to create 
        a Pythorch Geometric dynamical data.
        
        Parameters:
        -----------
        movement_data: dict
            Dictionary containing the movements of the patients
            over time, which can be used to compute dynamic 
            features of patients.
        locations_info: dict
            Dictionary containing information about the differnt 
            places. The values are dictionaries with the following
            keys: 'init_infected', 'ids_adjacents', 'ids_children',
            'name', 'id_parent', 'floor', 'service'.
        patients_data_df: pandas.DataFrame
            Data frame containing all the static features of the 
            patients.
        one_hot_enc_main_places: dict
            Dictionary where the IDs are the places IDs and the values are their respective
            one-hot encodings.
        one_hot_enc_specific_places: dict
            Dictionary where the IDs are the specific places IDs and the values are their respective
            one-hot encodings.
        forecast_window_length: int
            Length of the forecast window allowing to define if a patient is infected in the future
            or not.
        fully_connect_patients_same_place: bool
            If True, all the patients that are in the same places are
            connected with a weight of 1 or 1/n_patient_in_place.
        uniform_weights_patients_to_patient_edge: bool
            If true, the weights between two patients is 1/n_patient_in_place
            with n_patient_in_place is the number of patients that are in the
            sample place as those patients.     
            
        Returns:
        --------
        list_edge_index_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the edges indices for each edge type.
        list_edge_weight_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the edges weights for each edge type.
        list_node_feature_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the nodes features for each node type.
        list_node_target_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the nodes targets for each node type.
        list_node_timestamps_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the nodes timestamps for each node type.
        list_node_inf_risk_target_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the nodes infection risk target (not infected, infected, other) for each node type.
        list_node_transition_target_dicts: list
            List of dicts, where each dict contains the nodes state transitions.
        list_nodes_ids_dicts: list
            List of dicts (of size number of timesteps - forecast_window_length), where each dict contains
            the nodes IDs for each node type.
        patient_features_names_idx: dict
            Dictionary indicating the place of each feature in the final feature vector for the patients.
        places_features_names_idx: dict
            Dictionary indicating the place of each feature in the final feature vector for the places.

    """
    nodes_types = ["Patient", "Place"]
    #edges_types = ["Patient-Patient", "Patient-Place", "Place-Patient"]
    edges_types = [('Patient', 'Place'), ('Place', 'Patient')]
    if (fully_connect_patients_same_place):
        edges_types.append(('Patient', 'Patient'))
    list_edge_index_dicts = [{edge_type: [] for edge_type in edges_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_edge_weight_dicts = [{edge_type: [] for edge_type in edges_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_node_feature_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_node_target_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_node_timestamps_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_node_inf_risk_target_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_node_transition_target_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    list_nodes_ids_dicts = [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
    other_attributes = {
                            "list_edge_attrs_dicts": [{edge_type: [] for edge_type in edges_types} for _ in range(len(movement_data)-forecast_window_length)],
                            "IncubationDurationInHours": [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)],
                            "InfectionDurationInHours": [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)],
                            "TreatmentDurationInDays": [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)],
                            "Died": [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)],
                            "NonSusceptible": [{node_type: [] for node_type in nodes_types} for _ in range(len(movement_data)-forecast_window_length)]
                       }
    
    # Filling the previous variables
    for step_ID in tqdm(movement_data):
        #if (step_ID < 10):
        if (step_ID < len(movement_data)-forecast_window_length):
            # Getting the list of nodes
            for main_place_ID in movement_data[step_ID]:
                if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):                
                    # Node for the place
                    # Nodes IDs and timestamps
                    list_nodes_ids_dicts[step_ID]["Place"].append(main_place_ID)
                    list_node_timestamps_dicts[step_ID]["Place"].append(step_ID)
                    # Nodes targets
                    target_place = False
                    if (forecast_window_length == 1):
                        future_steps_IDs = [step_ID+1]
                    else:
                        future_steps_IDs = list(range(step_ID+1, step_ID + forecast_window_length + 1))
                    if (len(future_steps_IDs) == 0):
                        raise RuntimeError(f"\nNot possible to get future forecast window for current step ID {step_ID} with a forecast window length of {forecast_window_length}\n")
                    for future_step_ID in future_steps_IDs:
                        if (main_place_ID in movement_data[future_step_ID]["ContaminatedPlaces"]):
                            target_place = True
                    list_node_target_dicts[step_ID]["Place"].append(target_place)
                    list_node_inf_risk_target_dicts[step_ID]["Place"].append(target_place)
                    
                    # Nodes features
                    places_features, places_features_names_idx = get_features_places(
                                                                                        movement_data,
                                                                                        locations_info,
                                                                                        main_place_ID,
                                                                                        step_ID
                                                                                    )
                    list_node_feature_dicts[step_ID]["Place"].append(places_features)
                    list_node_transition_target_dicts[step_ID]["Place"].append(0) # Places don't have SIR transitions

                    # Node for patients
                    for patient_ID in movement_data[step_ID][main_place_ID]:
                        # Nodes IDs and timestamps
                        list_nodes_ids_dicts[step_ID]["Patient"].append(patient_ID)
                        list_node_timestamps_dicts[step_ID]["Patient"].append(step_ID)

                        # Capture CURRENT state
                        current_state = movement_data[step_ID][main_place_ID][patient_ID]['State']
                        
                        # Nodes target
                        # IMPORTANT: AS THE PATIENT CAN CHANGE OF STATE DURING THE FORECAST WINDOW, WE CONSIDER ONLY THE LAST STATE IN THE WINDOW
                        if (forecast_window_length == 1):
                            target_patient = None
                        else:
                            target_patient = [None for _ in range(forecast_window_length)]
                        target_inf_risk_patient = None # No need to have a list as we have one label per window
                        if (forecast_window_length == 1):
                            future_steps_IDs = [step_ID+1]
                        else:
                            future_steps_IDs = list(range(step_ID+1, step_ID + forecast_window_length + 1))
                        for future_step_ID in reversed(future_steps_IDs):
                            # As the patient may move in the future, we have to search everywhere in the hospital for the patient
                            for future_main_place_ID in movement_data[future_step_ID]:
                                if (patient_ID in movement_data[future_step_ID][future_main_place_ID]):
                                    if (type(future_main_place_ID) == int):
                                        future_state = movement_data[future_step_ID][future_main_place_ID][patient_ID]['State']
                                        # FORECAST HORIZON OF 1
                                        if (forecast_window_length == 1):
                                            # Target for the SEIR state
                                            target_patient = future_state
                                            # Target for the infection risk
                                            if (future_state == 2):
                                                target_inf_risk_patient = 1
                                            else:
                                                if (target_inf_risk_patient is None):
                                                    # If target_inf_risk_patient is None and we are here, we have found the patient
                                                    # in the hospital, and in the current future step, they are not infected
                                                    # so we temporarily put is as not infected (0) in the future, but we
                                                    # only do this once as if the patient gets infected in the future, then
                                                    # it is considered as a patient with risk of infection (1).
                                                    target_inf_risk_patient = 0
                                        # FORECAST HORIZON GREATER 1
                                        else:
                                            # Target for the SEIR state
                                            translated_ID = future_step_ID - step_ID - 1
                                            target_patient[translated_ID] = future_state

                                            # Target for the infection risk
                                            if (future_state == 2): # The patient will be infected in the future
                                                target_inf_risk_patient = 1 # Infected, ternary classification (not infected, infected, other)
                                            else:
                                                if (target_inf_risk_patient is None):
                                                    # If target_inf_risk_patient is None and we are here, we have found the patient
                                                    # in the hospital, and in the current future step, they are not infected
                                                    # so we temporarily put is as not infected (0) in the future, but we
                                                    # only do this once as if the patient gets infected in the future, then
                                                    # it is considered as a patient with risk of infection (1).
                                                    target_inf_risk_patient = 0 
                            if (target_patient is not None) and (forecast_window_length == 1):
                                break
                        # Processing cases where the patients were not found
                        if (target_patient is None): # Case where the forecast horizon is 1
                            if (patients_data_df.loc[patients_data_df["ID"] == patient_ID, "Died"].values[0]): # In this case the patient is no present in the future because they died
                                target_patient = MAPPING_STATES['D']
                            else: # In this case the patient is no present in the future because they were discharged
                                target_patient = MAPPING_STATES['NS']
                            #raise RuntimeError(f"Patient {patient_ID} present at step {step_ID} is not present at any of the future steps {future_steps_IDs} (dead or discharge) (current state: {movement_data[step_ID][main_place_ID][patient_ID]['State']})\n")
                        else: # Case where the forecast horizon is greater than 1
                            if (type(target_patient) == list) and (None in target_patient):
                                if (patients_data_df.loc[patients_data_df["ID"] == patient_ID, "Died"].values[0]): # In this case the patient is no present in the future because they died
                                    imputed_state = MAPPING_STATES['D']
                                else: # In this case the patient is no present in the future because they were discharged
                                    imputed_state = MAPPING_STATES['NS']
                                for tmp_ID in range(len(target_patient)):
                                    if (target_patient[tmp_ID] is None):
                                        target_patient[tmp_ID] = imputed_state

                        # For the case where we have a forecast horizon > 1, we regroup the D and NS in one class
                        if (forecast_window_length > 1):
                            if (MAPPING_STATES['D'] in target_patient) or (MAPPING_STATES['NS'] in target_patient):
                                target_inf_risk_patient = 2
                        else:
                            if (MAPPING_STATES['D'] == target_patient) or (MAPPING_STATES['NS'] == target_patient):
                                target_inf_risk_patient = 2

                        list_node_target_dicts[step_ID]["Patient"].append(target_patient)
                        list_node_inf_risk_target_dicts[step_ID]["Patient"].append(target_inf_risk_patient)
        

                        # SEIRD-NS Trajectory Tracking
                        # We build the trajectory list to identify the FIRST transition
                        trajectory = [current_state]
                        for future_step_ID in future_steps_IDs:
                            found_in_future = False
                            for f_place_ID in movement_data[future_step_ID]:
                                if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):  
                                    if (patient_ID in movement_data[future_step_ID][f_place_ID]):
                                        if (type(f_place_ID) == int):
                                            f_state = movement_data[future_step_ID][f_place_ID][patient_ID]['State']
                                            if (f_state != trajectory[-1]):
                                                trajectory.append(f_state)
                                            found_in_future = True
                                            break
                            # Handle Discharge (NS) or Death (D) if they disappear
                            if (not found_in_future):
                                died = patients_data_df.loc[patients_data_df["ID"] == patient_ID, "Died"].values[0]
                                final_state = MAPPING_STATES['D'] if died else MAPPING_STATES['NS']
                                if (final_state != trajectory[-1]):
                                    trajectory.append(final_state)
                                break # Patient is no longer in the hospital flow
                        # Categorize the Transition Label
                        # MAPPING (Example): S=0, E=1, I=2, R=3, D=4, NS=5
                        # DETAILED TRANSITIONS
                        # if (len(trajectory) == 1):
                        #     trans_label = 0  # STAY (Identity)
                        # else:
                        #     # We look at the first transition from current_state
                        #     next_s = trajectory[1]
                        #     if (current_state == 0): # From Susceptible
                        #         if (next_s == 1):
                        #             trans_label = 1 # S -> E (Infection)
                        #         elif (next_s == 2):
                        #             trans_label = 2 # S -> I (Direct/Rapid)
                        #         else:
                        #             trans_label = 5 # S -> NS/D (Discharge/Other)
                        #     elif (current_state == 1): # From Exposed
                        #         if (next_s == 2):
                        #             trans_label = 3 # E -> I (Progression)
                        #         else:
                        #             trans_label = 5
                        #     elif (current_state == 2): # From Infected
                        #         if (next_s == 3):
                        #             trans_label = 4 # I -> R (Recovery)
                        #         elif (next_s == 4):
                        #             trans_label = 6 # I -> D (Death)
                        #         else:
                        #             trans_label = 5
                        #     else:
                        #         trans_label = 5 # Others (R -> NS, etc.)
                        # SIMPLIFIED VERSION
                        if (len(trajectory) == 1):
                            trans_label = 0  # STAY (Identity)
                        else:
                            # We look at the first transition from current_state
                            next_s = trajectory[1]
                            if (current_state == 0): # From Susceptible
                                if (next_s == 1) or (next_s == 2): # S -> E (Infection) or S -> I (Direct/Rapid)
                                    trans_label = 1 
                                else:
                                    trans_label = 3 # S -> NS/D (Discharge/Other)
                            elif (current_state == 1): # From Exposed
                                if (next_s == 2):
                                    trans_label = 1 # E -> I (Progression)
                                else:
                                    trans_label = 3
                            elif (current_state == 2): # From Infected
                                if (next_s == 3):
                                    trans_label = 2 # I -> R (Recovery)
                                else:
                                    trans_label = 3
                            else:
                                trans_label = 3 # Others (R -> NS, etc.)
                        list_node_transition_target_dicts[step_ID]["Patient"].append(trans_label)
                        
                        # Nodes features
                        patient_features, patient_features_names_idx = get_features_patient(
                                                                                                patients_data_df,
                                                                                                movement_data,
                                                                                                locations_info,
                                                                                                patient_ID,
                                                                                                step_ID,
                                                                                                one_hot_enc_main_places=one_hot_enc_main_places,
                                                                                                one_hot_enc_specific_places=one_hot_enc_specific_places
                                                                                            )
                        list_node_feature_dicts[step_ID]["Patient"].append(patient_features)

            # Creating the edges
            for main_place_ID in movement_data[step_ID]:
                if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):                
                    patients_in_main_place = [f"Patient-{patient_ID}" for patient_ID in movement_data[step_ID][main_place_ID]]
                    n_patients_in_place = len(patients_in_main_place)
                    # The followin nested loop will create undirected edges
                    for patient_i_ID in patients_in_main_place:
                        # Creating edges between all the patients of the place
                        for patient_j_ID in patients_in_main_place:
                            if (fully_connect_patients_same_place):
                                if (patient_i_ID != patient_j_ID):
                                    list_edge_index_dicts[step_ID][('Patient', 'Patient')].append([patient_i_ID, patient_j_ID])
                                    if (uniform_weights_patients_to_patient_edge):
                                        list_edge_weight_dicts[step_ID][('Patient', 'Patient')].append(1.0/n_patients_in_place)
                                    else:
                                        list_edge_weight_dicts[step_ID][('Patient', 'Patient')].append(1.0)
        
                        # Creating edges between the nodes patients in the place and the node corresponding to the place
                        # For edges between Places and Patients we need to add the edges in both directions for undirected graph
                        list_edge_index_dicts[step_ID][('Patient', 'Place')].append([patient_i_ID, f"Place-{main_place_ID}"])
                        list_edge_index_dicts[step_ID][('Place', 'Patient')].append([f"Place-{main_place_ID}", patient_i_ID])
                        list_edge_weight_dicts[step_ID][('Patient', 'Place')].append(np.array([1.0]))
                        list_edge_weight_dicts[step_ID][('Place', 'Patient')].append(np.array([1.0]))


    # Simplifying the list_edge_index_dicts to be able to use it in the DS
    new_list_edge_index_dicts = deepcopy(list_edge_index_dicts)
    for step_ID in tqdm(range(len(list_edge_index_dicts))):
        for edge_type in list_edge_index_dicts[step_ID]:
            n_edges = len(list_edge_index_dicts[step_ID][edge_type])
            for edge_ID in range(n_edges):
                edge = list_edge_index_dicts[step_ID][edge_type][edge_ID]
                new_list_edge_index_dicts[step_ID][edge_type][edge_ID] = [int(edge[0].split('-')[-1]), int(edge[1].split('-')[-1])]
            # Transpose to use it with Pytorch Geometric models: (2, n_nodes)
            new_list_edge_index_dicts[step_ID][edge_type] = np.array(new_list_edge_index_dicts[step_ID][edge_type])
            new_list_edge_index_dicts[step_ID][edge_type] = new_list_edge_index_dicts[step_ID][edge_type].T
    list_edge_index_dicts = new_list_edge_index_dicts

    # Converting the targets into lists of dicts of np arrays
    list_node_np_target_dicts = deepcopy(list_node_target_dicts)
    list_node_inf_risk_np_target_dicts = deepcopy(list_node_inf_risk_target_dicts)
    for step_ID in tqdm(range(len(list_node_target_dicts))):
        for node_type in list_node_target_dicts[step_ID]:
            list_node_np_target_dicts[step_ID][node_type] = np.array(list_node_target_dicts[step_ID][node_type])
            list_node_inf_risk_np_target_dicts[step_ID][node_type] = np.array(list_node_inf_risk_target_dicts[step_ID][node_type], dtype=int)
    list_node_target_dicts = list_node_np_target_dicts
    list_node_inf_risk_target_dicts = list_node_inf_risk_np_target_dicts

    # Converting the timestamps into lists of dicts of np arrays
    list_node_np_timestamps_dicts = deepcopy(list_node_timestamps_dicts)
    for step_ID in tqdm(range(len(list_node_timestamps_dicts))):
        for node_type in list_node_timestamps_dicts[step_ID]:
            list_node_np_timestamps_dicts[step_ID][node_type] = np.array(list_node_timestamps_dicts[step_ID][node_type])
    list_node_timestamps_dicts = list_node_np_timestamps_dicts

    # Converting the nodes ids into lists of dicts of np arrays
    list_np_nodes_ids_dicts = deepcopy(list_nodes_ids_dicts)
    for step_ID in tqdm(range(len(list_nodes_ids_dicts))):
        for node_type in list_nodes_ids_dicts[step_ID]:
            list_np_nodes_ids_dicts[step_ID][node_type] = np.array(list_nodes_ids_dicts[step_ID][node_type])
    list_nodes_ids_dicts = list_np_nodes_ids_dicts

    # Convert to numpy arrays 
    for step_ID in tqdm(range(len(list_node_transition_target_dicts))):
        for node_type in list_node_transition_target_dicts[step_ID]:
            list_node_transition_target_dicts[step_ID][node_type] = np.array(
                list_node_transition_target_dicts[step_ID][node_type], dtype=int
            )

    # Return the new list along with others
    return list_edge_index_dicts,\
           list_edge_weight_dicts,\
           list_node_feature_dicts,\
           list_node_target_dicts,\
           list_node_timestamps_dicts,\
           list_node_inf_risk_target_dicts,\
           list_node_transition_target_dicts,\
           list_nodes_ids_dicts, \
           patient_features_names_idx,\
           places_features_names_idx



def create_HDF5_file(
                        h5_fn,
                        list_edge_index_dicts,
                        list_edge_weight_dicts,
                        list_node_feature_dicts,
                        list_node_target_dicts,
                        list_node_timestamps_dicts,
                        list_node_inf_risk_target_dicts,
                        list_node_transition_target_dicts,
                        list_nodes_ids_dicts
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
    i = 0
    h5_fn = h5_fn.split(".hdf5")[0] + '_'
    while (os.path.exists(h5_fn + str(i) + '.hdf5')):
        i += 1
    h5_fn = h5_fn + str(i) + '.hdf5'
    
    # Creating the file
    hdf5_file = h5py.File(h5_fn, "w")
    
    # Creating the different main groups
    main_groups = {
                    "EdgeIndexDicts": hdf5_file.create_group("EdgeIndexDicts"),
                    "EdgeFeaturesDicts": hdf5_file.create_group("EdgeFeaturesDicts"),
                    "NodeFeaturesDicts": hdf5_file.create_group("NodeFeaturesDicts"),
                    "NodeTargetsDict": hdf5_file.create_group("NodeTargetsDict"),
                    "NodeTimestampsDicts": hdf5_file.create_group("NodeTimestampsDicts"),
                    "NodeInfRiskTargetsDict": hdf5_file.create_group("NodeInfRiskTargetsDict"),
                    "NodeTargetsTransitionsDict": hdf5_file.create_group("NodeTargetsTransitionsDict"),
                    "NodesIdsDicts": hdf5_file.create_group("NodesIdsDicts")
                  }
    py_murcia_data = {
                        "EdgeIndexDicts": list_edge_index_dicts,
                        "EdgeFeaturesDicts": list_edge_weight_dicts,
                        "NodeFeaturesDicts": list_node_feature_dicts,
                        "NodeTargetsDict": list_node_target_dicts,
                        "NodeTimestampsDicts": list_node_timestamps_dicts,
                        "NodeInfRiskTargetsDict": list_node_inf_risk_target_dicts,
                        "NodeTargetsTransitionsDict": list_node_transition_target_dicts,
                        "NodesIdsDicts": list_nodes_ids_dicts
                     }
    
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
                step_group.create_dataset(dataset_name, data=data)
                
    # Close file
    hdf5_file.close()

    return h5_fn




#====================================================================================================#
#=============================Estimation of epidemic parameters from data=============================#
#====================================================================================================#
def estimate_epidemic_params_from_data(
                                        patients_data_df,
                                        movement_data,
                                        rates_in_per_day=True
                                      ):
    """
        Estimates the epidemic durations per patient (incubation and infection hours).

        Parameters:
        -----------
        patients_data_df: pandas.DataFrame
            Data frame containing all the static features of the 
            patients.
        movement_data: dict
            Dictionary containing the movements of the patients
            over time, which can be used to compute dynamic 
            features of patients.
        rates_in_per_day: bool
            If True, the rates are given per day instead of per hour.

        Returns:
        --------
        estimated_epidemic_params: dict
            Dictionary where the keys are the estimated epidemic parameters
            and the values are their values.
    """
    
    #======================================================================#
    #======================================================================#
     # Estimating some epidmic durations (incubation and infection hours)
    print("\n\n==========> Estimating some epidmic durations (incubation and infection hours) <==========\n")
    epidemic_durations_per_patient = {}
    for patient_ID in tqdm(patients_data_df['ID']):
        if (patient_ID not in epidemic_durations_per_patient):
            epidemic_durations_per_patient[patient_ID] = {
                                                            'IncubationHours': None,
                    
                                                            'InfectionHours': None,
                                                            'InfectionHoursManual': None,
                
                                                            'Died': None
                                                        }
        # Incubation hours
        inc_duration = patients_data_df.loc[patients_data_df['ID'] == patient_ID, 'IncubationDurationInHours'].values[0]
        epidemic_durations_per_patient[patient_ID]['IncubationHours'] = inc_duration            
                
        # Infection hours
        inf_duration = patients_data_df.loc[patients_data_df['ID'] == patient_ID, 'InfectionDurationInHours'].values[0]
        # IMPORTANT: THE FIRST MAIL FROM DENISSE IS WRONG AND THE INFECTION DURATION IN THE CSV FILE IS IN TIME STEPS, NOT HOURS
        epidemic_durations_per_patient[patient_ID]['InfectionHours'] = inf_duration*8

        # Died or discharged
        epidemic_durations_per_patient[patient_ID]['Died'] = patients_data_df.loc[patients_data_df['ID'] == patient_ID, 'Died'].values[0]

    
    #======================================================================#
    #======================================================================#
    print("\n\n==========> Getting the recovery rates, plus other SEIRD-NS params (A, A_S, A_I, A_NS, ...) <==========\n")
    # Getting the recovery rates, plus other SEIRD-NS params (A, A_S, A_I, A_NS, ...)
    n_steps = len(movement_data)
    #n_steps = 1750 # If using only the first steps (used for training)
    arrivals_per_eight_hours = [0 for _ in range(n_steps)]
    arrivals_S_per_eight_hours = [0 for _ in range(n_steps)]
    arrivals_E_per_eight_hours = [0 for _ in range(n_steps)]
    arrivals_I_per_eight_hours = [0 for _ in range(n_steps)]
    arrivals_R_per_eight_hours = [0 for _ in range(n_steps)]
    arrivals_NS_per_eight_hours = [0 for _ in range(n_steps)]
    deceased_per_eight_hours = [0 for _ in range(n_steps)]
    discharged_per_eight_hours = [0 for _ in range(n_steps)]
    start_end_recovery_per_patient = {patient_ID:[None, None] for patient_ID in list(epidemic_durations_per_patient.keys())}
    new_exposed_ind_per_eight_hours = [0 for _ in range(n_steps)]
    patients_in_previous_steps = set() # Useful to identify new arriving patients
    dead_patients = set()
    for step_ID in tqdm(range(n_steps)):
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ['ContaminatedPlaces', 'PlacesToClean']):
                for patient_ID in movement_data[step_ID][main_place_ID]:
                    current_patient_state = movement_data[step_ID][main_place_ID][patient_ID]['State']
                    if (current_patient_state == MAPPING_STATES['I']):
                        if (start_end_recovery_per_patient[patient_ID][0] is None):
                            start_end_recovery_per_patient[patient_ID][0] = step_ID
                    if (current_patient_state == MAPPING_STATES['R']):
                        if (start_end_recovery_per_patient[patient_ID][1] is None):
                            start_end_recovery_per_patient[patient_ID][1] = step_ID

                    # New exposed patients
                    if (current_patient_state == MAPPING_STATES['E']):
                        if (step_ID == 0):
                            new_exposed_ind_per_eight_hours[step_ID] += 1
                        else:
                            previous_step = step_ID - 1
                            previous_state = None # If it stays None, then we have a new patient
                            for main_place_bis_ID in movement_data[previous_step]:
                                if (main_place_bis_ID not in ['ContaminatedPlaces', 'PlacesToClean']):
                                    for patient_bis_ID in movement_data[previous_step][main_place_bis_ID]:
                                        if (patient_ID == patient_bis_ID):
                                            previous_state = movement_data[previous_step][main_place_bis_ID][patient_bis_ID]['State']
                            if (previous_state is None) or (previous_state == MAPPING_STATES['S']):
                                new_exposed_ind_per_eight_hours[step_ID] += 1
                        
                    # For deceased patients, put recovery rate to 0
                    if (current_patient_state == MAPPING_STATES['D']):
                        if (start_end_recovery_per_patient[patient_ID][0] is not None) and\
                        (start_end_recovery_per_patient[patient_ID][1] is None) and\
                        (patient_ID not in dead_patients): # It means the patient was in the infected state and did not recovered
                            deceased_per_eight_hours[step_ID] += 1
                            dead_patients.add(patient_ID)
                        start_end_recovery_per_patient[patient_ID] = [None, None]

                    # Number of arrivals
                    if (step_ID == 0): # All patients considered as arrivals
                        patients_in_previous_steps.add(patient_ID)
                        arrivals_per_eight_hours[step_ID] += 1
                        if (current_patient_state == MAPPING_STATES['S']):
                            arrivals_S_per_eight_hours[step_ID] += 1
                        if (current_patient_state == MAPPING_STATES['I']):
                            arrivals_I_per_eight_hours[step_ID] += 1
                        if (current_patient_state == MAPPING_STATES['NS']):
                            arrivals_NS_per_eight_hours[step_ID] += 1
                    else:
                        # Identifying if the patient is new
                        new_patient = False
                        if (patient_ID not in patients_in_previous_steps):
                            new_patient = True
                            patients_in_previous_steps.add(patient_ID)

                        # Updating arrivals
                        if (new_patient):
                            arrivals_per_eight_hours[step_ID] += 1
                            if (current_patient_state == MAPPING_STATES['S']):
                                arrivals_S_per_eight_hours[step_ID] += 1
                            if (current_patient_state == MAPPING_STATES['E']):
                                arrivals_E_per_eight_hours[step_ID] += 1
                            if (current_patient_state == MAPPING_STATES['I']):
                                arrivals_I_per_eight_hours[step_ID] += 1
                            if (current_patient_state == MAPPING_STATES['R']):
                                arrivals_R_per_eight_hours[step_ID] += 1
                            if (current_patient_state == MAPPING_STATES['NS']):
                                arrivals_NS_per_eight_hours[step_ID] += 1

                    # For discharged patients, we have to see if they are not present in the next step AND not deceased
                    if (current_patient_state != MAPPING_STATES['D']):
                        patient_discharged = True
                        next_step = step_ID + 1
                        if (step_ID < n_steps-1): # penultimate step
                            for main_place_bis_ID in movement_data[next_step]:
                                if (main_place_bis_ID not in ['ContaminatedPlaces', 'PlacesToClean']):
                                    for patient_bis_ID in movement_data[next_step][main_place_bis_ID]:
                                        if (patient_ID == patient_bis_ID):
                                            patient_discharged = False
                        if (patient_discharged):
                            discharged_per_eight_hours[step_ID] += 1
    new_exposed_ind_per_eight_hours = np.array(new_exposed_ind_per_eight_hours)

    
    # Variable with the global durations
    global_epidemic_durations = {
                                "ArrivalsPerEightHours": arrivals_per_eight_hours,
                                "ArrivalsSPerEightHours": arrivals_S_per_eight_hours,
                                "ArrivalsEPerEightHours": arrivals_E_per_eight_hours,
                                "ArrivalsIPerEightHours": arrivals_I_per_eight_hours,
                                "ArrivalsRPerEightHours": arrivals_R_per_eight_hours,
                                "ArrivalsNSPerEightHours": arrivals_NS_per_eight_hours,
                                "MortalityPerEightHours": deceased_per_eight_hours,
                                "DischargePerEightHours": discharged_per_eight_hours,
                            }
    

    #======================================================================#
    #======================================================================#
    print("\n\n==========> Getting the recovery rates in hours <==========\n")
    # Getting the recovery rates in hours
    for patient_ID in tqdm(start_end_recovery_per_patient):
        #if (start_end_recovery_per_patient[patient_ID][0] is not None):
        if (start_end_recovery_per_patient[patient_ID][0] is not None) and (start_end_recovery_per_patient[patient_ID][1] is not None):
            epidemic_durations_per_patient[patient_ID]['InfectionHoursManual'] = 8*(start_end_recovery_per_patient[patient_ID][1]-start_end_recovery_per_patient[patient_ID][0]) # NO NEED TO ADD OR SUBSTRACT 1
        else:
            epidemic_durations_per_patient[patient_ID]['InfectionHoursManual'] = 0
            
    #======================================================================#
    #======================================================================#
    print("\n\n==========> Computing the mean rates based on the mean hours of incubation, recovery and infection <==========\n")
    # Computing the mean rates based on the mean hours of incubation, recovery and infection
    # Incubation rate
    inc_durations_hours = [epidemic_durations_per_patient[patient_ID]['IncubationHours'] for patient_ID in epidemic_durations_per_patient if  epidemic_durations_per_patient[patient_ID]['IncubationHours'] != 0]
    mean_inc_duration_hours = np.mean(inc_durations_hours)
    inc_rate_per_hour = 1/mean_inc_duration_hours
    inc_rate_per_hour = 1/mean_inc_duration_hours
    if (rates_in_per_day):
        global_inc_rate = 24*inc_rate_per_hour
    else:
        global_inc_rate = inc_rate_per_hour
    # Recovery rate
    # The number of hours that a patient was infectious is the number of hours they needed to recover
    rec_durations_hours = []
    # Considering only patients who recovered (excldue those who died)
    for patient_ID in epidemic_durations_per_patient:
        if (epidemic_durations_per_patient[patient_ID]['InfectionHours'] != 0) and (not epidemic_durations_per_patient[patient_ID]['Died']):
            rec_durations_hours.append(epidemic_durations_per_patient[patient_ID]['InfectionHours'])
    mean_rec_duration_hours = np.mean(rec_durations_hours)
    rec_rate_per_hour = 1/mean_rec_duration_hours
    if (rates_in_per_day):
        global_rec_rate = 24*rec_rate_per_hour
    else:
        global_rec_rate = rec_rate_per_hour
        
    # Print information
    # Incubation rate
    th_inc_rate = 1/2.5
    per_time_unit = "PER DAY" if rates_in_per_day else "PER HOUR"
    print(f"\n=========> Global experimental incubation rate: {global_inc_rate} {per_time_unit}")
    print(f"\t=========> Theoretical incubation rate (used to get the data): {th_inc_rate}")

    # Recovery rate
    th_rec_rate = 1 - 0.027
    print(f"\n=========> Global experimental recovery rate: {global_rec_rate} {per_time_unit}")
    print(f"\t=========> Theoretical recovery rate (used to get the data): {th_rec_rate}")

    #======================================================================#
    #======================================================================#
    print("\n\n==========> Computing the mortality rates <==========\n")
    # Morality rates
    total_mortalities = np.sum(global_epidemic_durations['MortalityPerEightHours'])
    infectious_times = [epidemic_durations_per_patient[patient_ID]['InfectionHours'] for patient_ID in epidemic_durations_per_patient if epidemic_durations_per_patient[patient_ID]['InfectionHours'] != 0]
    total_infectious_person_time = np.sum(infectious_times)
    global_mortality_rate_per_hour = total_mortalities/total_infectious_person_time
    if (rates_in_per_day):
        global_mortality_rate = global_mortality_rate_per_hour
    else:
        global_mortality_rate = 24*global_mortality_rate_per_hour
    print(f"\n=========> Global experimental mortality rate: {global_mortality_rate} {per_time_unit}")
    print(f"\t=========> Theoretical propotion of arrivals in NS state: {0.027}")

    #======================================================================#
    #======================================================================#
    print("\n\n==========> Computing the discharge rates <==========\n")
    # Discharge rates
    mean_length_of_stay_days = patients_data_df['TrueLoS'].values.mean()
    global_discharge_rate_per_day = 1/mean_length_of_stay_days
    if (rates_in_per_day):
        global_discharge_rate = global_discharge_rate_per_day
    else:
        global_discharge_rate = global_discharge_rate_per_day/24
    print(f"\n=========> Global experimental discharge rate: {global_discharge_rate} {per_time_unit}")
    print(f"\t=========> Theoretical propotion of arrivals in NS state: {0.23507287259050308}")


    #======================================================================#
    #======================================================================#
    print("\n\n==========> Computing the arrival rates <==========\n")
    # Arrival rate
    total_arrivals = np.sum(global_epidemic_durations['ArrivalsPerEightHours'])
    total_time = 8*len(global_epidemic_durations['ArrivalsPerEightHours'])
    global_arrival_rate_per_hour = total_arrivals/total_time
    if (rates_in_per_day):
        global_arrival_rate = 24*global_arrival_rate_per_hour
    else:
        global_arrival_rate = global_arrival_rate_per_hour
        
    print(f"\n=========> Global experimental arrival rate: {global_arrival_rate} {per_time_unit}")
    print(f"\t=========> Theoretical arrival rate: {18.603} PER DAY")

    # Arrival S proportion
    total_arrivals_S = np.sum(global_epidemic_durations['ArrivalsSPerEightHours'])
    global_proportion_arrival_S = total_arrivals_S/total_arrivals
    print(f"\n=========> Global experimental propotion of arrivals in S state: {global_proportion_arrival_S}")
    print(f"\t=========> Theoretical propotion of arrivals in S state: {0.997}")

    # Arrival E proportion
    total_arrivals_E = np.sum(global_epidemic_durations['ArrivalsEPerEightHours'])
    global_proportion_arrival_E = total_arrivals_E/total_arrivals
    print(f"\n=========> Global experimental propotion of arrivals in E state: {global_proportion_arrival_E}")
    print(f"\t=========> Theoretical propotion of arrivals in E state: None")

    # Arrival I proportion
    total_arrivals_I = np.sum(global_epidemic_durations['ArrivalsIPerEightHours'])
    global_proportion_arrival_I = total_arrivals_I/total_arrivals
    print(f"\n=========> Global experimental propotion of arrivals in I state: {global_proportion_arrival_I}")
    print(f"\t=========> Theoretical propotion of arrivals in I state: {0.002}")

    # Arrival R proportion
    total_arrivals_R = np.sum(global_epidemic_durations['ArrivalsRPerEightHours'])
    global_proportion_arrival_R = total_arrivals_R/total_arrivals
    print(f"\n=========> Global experimental propotion of arrivals in R state: {global_proportion_arrival_R}")
    print(f"\t=========> Theoretical propotion of arrivals in R state: None")

    # Arrival NS proportion
    total_arrivals_NS = np.sum(global_epidemic_durations['ArrivalsNSPerEightHours'])
    global_proportion_arrival_NS = total_arrivals_NS/total_arrivals
    print(f"\n=========> Global experimental propotion of arrivals in NS state: {global_proportion_arrival_NS}")
    print(f"\t=========> Theoretical propotion of arrivals in NS state: {0.001}")


    #======================================================================#
    #======================================================================#
    print("\n\n==========> Estimating the infection rate <==========\n")
    # Estimating infection rate
    # Getting the S, E, I, R, D, NS values per time step
    patients_per_state_over_time = get_patients_per_state_over_time(movement_data, plot_evolution=False)
    S = np.array(patients_per_state_over_time[MAPPING_STATES['S']])
    E = np.array(patients_per_state_over_time[MAPPING_STATES['E']])
    I = np.array(patients_per_state_over_time[MAPPING_STATES['I']])
    R = np.array(patients_per_state_over_time[MAPPING_STATES['R']])
    D = np.array(patients_per_state_over_time[MAPPING_STATES['D']])
    NS = np.array(patients_per_state_over_time[MAPPING_STATES['NS']])
    times = np.array([tmp_t*8/(24) for tmp_t in movement_data])
    N = S + E + I + R + D + NS
    # Computing the betas (infection rates) for each time step
    betas = new_exposed_ind_per_eight_hours*N/(S*I)
    non_zero_I_and_S_steps = [step_ID for step_ID in range(len(S)) if (S[step_ID] != 0 and I[step_ID] != 0)]
    valid_betas = betas[non_zero_I_and_S_steps]
    global_inf_rate_per_hour = (valid_betas/8).mean()
    if (rates_in_per_day):
        global_inf_rate = 24*global_inf_rate_per_hour
    else:
        global_inf_rate = global_inf_rate_per_hour
    th_rec_rate = 1/(0.435 + 0.24)
    print(f"\n=========> Global experimental infection rate: {global_inf_rate} {per_time_unit}")
    print(f"\t=========> Theoretical infection rate: {th_rec_rate}")

    #======================================================================#
    #======================================================================#
    # Final variable
    estimated_epidemic_params = {
                                    'A': global_arrival_rate,
                                    'A_S': global_proportion_arrival_S,
                                    'A_E': global_proportion_arrival_E,
                                    'A_I': global_proportion_arrival_I,
                                    'A_R': global_proportion_arrival_R,
                                    'A_NS': global_proportion_arrival_NS,
                                    
                                    'BETA': global_inf_rate,
                                    
                                    'MU': global_mortality_rate,
                                    'DIS_RATE': global_discharge_rate,
                                    
                                    'ALPHA': 1/global_inc_rate,
                                    
                                    'GAMMA': global_rec_rate,
                                }

    return estimated_epidemic_params
            

#====================================================================================================#
#===========================Verification of Hazard discretiezd equations=============================#
#====================================================================================================#
def get_state_probs_patient(
                                states_probs_per_patient,
                                step_ID,
                                true_patient_ID,
                                inv_true_patients_ids_mapping_per_timestep
                            ):
    state_probs = None
    try:
        local_patient_ID = inv_true_patients_ids_mapping_per_timestep[step_ID][true_patient_ID]
        state_probs = states_probs_per_patient[step_ID][local_patient_ID]
    except:
        #print(f"\n=========> Patient {true_patient_ID} is not present in time step {step_ID}\n")
        state_probs = np.array([np.nan for _ in range(6)])

    return state_probs

def verification_hazard_discrete_equations(
                                            movement_data,
                                            epidemiological_params,
                                            plot_figures=True
                                          ):
    """
        Verifies the validity of the discretization of the Hazard exponential waiting times 
        equations obtained by integratig factor method.

        Parameters:
        -----------
        movement_data: dict
            Dictionary containing the movements of the patients
            over time, which can be used to compute dynamic 
            features of patients.
        epidemiological_params: dict
            Dictionary where the keys are the names of the epidemiological parameters
            and the values are their values.
        plot_figures: bool
            True if want to plot figures.
    """
    # One-hot encoding of the states of the patients
    n_timesteps = len(movement_data)
    states_probs_per_patient = [[] for _ in range(n_timesteps)]
    true_patients_ids_mapping_per_timestep = [{} for _ in range(n_timesteps)]
    inv_true_patients_ids_mapping_per_timestep = [{} for _ in range(n_timesteps)]
    for step_ID in movement_data:
        local_patient_ID = 0
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):
                for patient_ID in movement_data[step_ID][main_place_ID]:
                    patient_state = movement_data[step_ID][main_place_ID][patient_ID]['State']
                    patient_state_one_hot = one_hot_encoding_np(patient_state, num_classes=6)
                    states_probs_per_patient[step_ID].append(patient_state_one_hot)
                    true_patients_ids_mapping_per_timestep[step_ID][local_patient_ID] = patient_ID
                    inv_true_patients_ids_mapping_per_timestep[step_ID][patient_ID] = local_patient_ID
                    local_patient_ID += 1
        states_probs_per_patient[step_ID] = np.array(states_probs_per_patient[step_ID])

    # Getting the different states per patients
    n_nodes_list = [states_probs_per_patient[step_ID].shape[0] for step_ID in range(n_timesteps)]
    S_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 0]).float() for step_ID in range(n_timesteps)]
    E_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 1]).float() for step_ID in range(n_timesteps)]
    I_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 2]).float() for step_ID in range(n_timesteps)]
    R_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 3]).float() for step_ID in range(n_timesteps)]
    D_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 4]).float() for step_ID in range(n_timesteps)]
    NS_per_patient = [torch.from_numpy(states_probs_per_patient[step_ID][:, 5]).float() for step_ID in range(n_timesteps)]

    # Creating the edges
    edges_types = [('Patient', 'Patient'), ('Patient', 'Place'), ('Place', 'Patient')]
    edge_index_dicts_per_time_step = [{edge_type: [] for edge_type in edges_types} for _ in range(n_timesteps)]
    edge_weight_dicts_per_time_step = [{edge_type: [] for edge_type in edges_types} for _ in range(n_timesteps)]
    for step_ID in tqdm(range(n_timesteps)):
        for main_place_ID in movement_data[step_ID]:
            if (main_place_ID not in ["ContaminatedPlaces", "PlacesToClean"]):                
                patients_in_main_place = [f"Patient-{patient_ID}" for patient_ID in movement_data[step_ID][main_place_ID]]
                for patient_i_ID in patients_in_main_place:
                    # Creating edges between all the patients of the place
                    for patient_j_ID in patients_in_main_place:
                        if (patient_i_ID != patient_j_ID):
                            edge_index_dicts_per_time_step[step_ID][('Patient', 'Patient')].append([patient_i_ID, patient_j_ID])
                            edge_weight_dicts_per_time_step[step_ID][('Patient', 'Patient')].append(1.0)
        
                    # Creating edges between the nodes patients in the place and the node corresponding to the place
                    edge_index_dicts_per_time_step[step_ID][('Patient', 'Place')].append([patient_i_ID, f"Place-{main_place_ID}"])
                    edge_index_dicts_per_time_step[step_ID][('Place', 'Patient')].append([f"Place-{main_place_ID}", patient_i_ID])
                    edge_weight_dicts_per_time_step[step_ID][('Patient', 'Place')].append(np.array([1.0]))
                    edge_weight_dicts_per_time_step[step_ID][('Place', 'Patient')].append(np.array([1.0]))

    # Creating the adjacency matrices
    adj_matrices = [None for _ in range(n_timesteps)]
    for step_ID in tqdm(range(n_timesteps)):
        n_edges = len(edge_index_dicts_per_time_step[step_ID][('Patient', 'Patient')])
        n_nodes = n_nodes_list[step_ID]
        tmp_adj_matrix = torch.zeros(n_nodes, n_nodes)
        for edge_ID in range(n_edges):
            true_node_i_ID = int(edge_index_dicts_per_time_step[step_ID][('Patient', 'Patient')][edge_ID][0].split('-')[-1])
            local_node_i_ID = inv_true_patients_ids_mapping_per_timestep[step_ID][true_node_i_ID]
            true_node_j_ID = int(edge_index_dicts_per_time_step[step_ID][('Patient', 'Patient')][edge_ID][1].split('-')[-1])
            local_node_j_ID = inv_true_patients_ids_mapping_per_timestep[step_ID][true_node_j_ID]
            tmp_adj_matrix[local_node_i_ID, local_node_j_ID] = edge_weight_dicts_per_time_step[step_ID][('Patient', 'Patient')][edge_ID]
            tmp_adj_matrix[local_node_j_ID, local_node_i_ID] = edge_weight_dicts_per_time_step[step_ID][('Patient', 'Patient')][edge_ID]
        adj_matrices[step_ID] = tmp_adj_matrix

    # Getting the local infection hazard (force of infection)
    NORMALIZE_ADJ_MAT = False
    #NORMALIZE_ADJ_MAT = True
    lambdas = [epidemiological_params["BETA"]*(adj_matrices[step_ID] @ I_per_patient[step_ID]) for step_ID in range(n_timesteps)]
    if (NORMALIZE_ADJ_MAT):
        lambdas = [lambdas[step_ID]/adj_matrices[step_ID].sum(dim=1) for step_ID in range(n_timesteps)]

    # Getting the fraction of arrivals per compartment in the current step
    w_t = [1/n_nodes_list[step_ID] for step_ID in range(n_timesteps)]

    # Computing the next-step probabilities 
    delta_t = 0.33
    p_S_to_E_per_node = [1 - torch.exp(-lambdas[step_ID]*delta_t) for step_ID in range(n_timesteps)]
    p_E_to_I = [1 - torch.exp(-torch.tensor(epidemiological_params["ALPHA"])*delta_t) for step_ID in range(n_timesteps)]
    p_leave_I = [1 - torch.exp(-(torch.tensor(epidemiological_params["GAMMA"]) + torch.tensor(epidemiological_params["MU"]) + torch.tensor(epidemiological_params["DIS_RATE"]))*delta_t) for step_ID in range(n_timesteps)]
    p_I_to_R = [torch.tensor(epidemiological_params["GAMMA"])/(torch.tensor(epidemiological_params["GAMMA"]) + torch.tensor(epidemiological_params["MU"]) + torch.tensor(epidemiological_params["DIS_RATE"]))*p_leave_I[step_ID] for step_ID in range(n_timesteps)]
    p_I_to_D = [torch.tensor(epidemiological_params["MU"])/(torch.tensor(epidemiological_params["GAMMA"]) + torch.tensor(epidemiological_params["MU"]) + torch.tensor(epidemiological_params["DIS_RATE"]))*p_leave_I[step_ID] for step_ID in range(n_timesteps)]
            
    # Next state using the discrete-time Hazard formulation waiting time
    S_next_per_patient = [None for step_ID in range(n_timesteps)]
    E_next_per_patient = [None for step_ID in range(n_timesteps)]
    I_next_per_patient = [None for step_ID in range(n_timesteps)]
    R_next_per_patient = [None for step_ID in range(n_timesteps)]
    D_next_per_patient = [None for step_ID in range(n_timesteps)]
    NS_next_per_patient = [None for step_ID in range(n_timesteps)]
    all_comparts_next_per_patient = [None for step_ID in range(n_timesteps)]
    all_comparts_current_GT_per_patient = [None for step_ID in range(n_timesteps)]


    # Computing the next states
    #USE_NEW_EQUATIONS = False
    USE_NEW_EQUATIONS = True
    A = epidemiological_params["A"] 
    A_S = epidemiological_params["A_S"] 
    A_E = epidemiological_params["A_E"] 
    A_I = epidemiological_params["A_I"] 
    A_R = epidemiological_params["A_R"] 
    A_NS = epidemiological_params["A_NS"] 
    BETA = epidemiological_params["BETA"] 
    MU = epidemiological_params["MU"] 
    DIS_RATE = epidemiological_params["DIS_RATE"]
    ALPHA = epidemiological_params["ALPHA"] 
    GAMMA = epidemiological_params["GAMMA"]
    for step_ID in tqdm(range(n_timesteps)):
        if (not USE_NEW_EQUATIONS):
            # OLD VERSIONS
            S_next_per_patient[step_ID] = S_per_patient[step_ID] * torch.exp(-(lambdas[step_ID]+torch.tensor(DIS_RATE))*delta_t)
            
            E_next_per_patient[step_ID] = E_per_patient[step_ID] * torch.exp(-(torch.tensor(ALPHA) + torch.tensor(DIS_RATE))*delta_t) + S_per_patient[step_ID] * p_S_to_E_per_node[step_ID]
            
            I_next_per_patient[step_ID] = I_per_patient[step_ID] * torch.exp(-(torch.tensor(GAMMA) + torch.tensor(MU) + torch.tensor(DIS_RATE))*delta_t) + E_per_patient[step_ID] * (1-torch.exp(-(torch.tensor(ALPHA) + torch.tensor(DIS_RATE))*delta_t))
            
            R_next_per_patient[step_ID] = R_per_patient[step_ID] * torch.exp(-torch.tensor(DIS_RATE)*delta_t) + I_per_patient[step_ID] * p_I_to_R[step_ID]
            
            D_next_per_patient[step_ID] = D_per_patient[step_ID] * torch.exp(-torch.tensor(delta_t)) + I_per_patient[step_ID] * p_I_to_D[step_ID]
            
            NS_next_per_patient[step_ID] = NS_per_patient[step_ID] * torch.exp(-torch.tensor(DIS_RATE)*delta_t)

        else:
            alpha, beta, gamma, mu, dis_rate = torch.tensor(ALPHA), torch.tensor(BETA), torch.tensor(GAMMA), torch.tensor(MU), torch.tensor(DIS_RATE)
            delta_t = torch.tensor(delta_t)
            S_next_per_patient[step_ID] = S_per_patient[step_ID] * torch.exp(-(lambdas[step_ID] + dis_rate)*delta_t)
        
            E_next_per_patient[step_ID] = E_per_patient[step_ID] * torch.exp(-(alpha + dis_rate)*delta_t) +\
                    (lambdas[step_ID] * S_per_patient[step_ID])/(alpha + dis_rate)  * (1 - torch.exp(-(alpha + dis_rate)*delta_t))
        
            I_next_per_patient[step_ID] = I_per_patient[step_ID] * torch.exp(-(gamma + mu + dis_rate)*delta_t) +\
                    (alpha * E_per_patient[step_ID])/(gamma + mu + dis_rate) * (1 - torch.exp(-(gamma + mu + dis_rate)*delta_t))
        
            R_next_per_patient[step_ID] = R_per_patient[step_ID] * torch.exp(-dis_rate*delta_t) +\
                    (gamma * I_per_patient[step_ID] * torch.exp(-dis_rate * delta_t))/(gamma + mu) * (1 - torch.exp(-(gamma + mu)*delta_t)) +\
                    gamma * (alpha * E_per_patient[step_ID])/(gamma + mu + dis_rate) * ( (1 - torch.exp(-dis_rate*delta_t))/dis_rate - (torch.exp(-dis_rate*delta_t) - torch.exp(-(gamma + mu + dis_rate)*delta_t))/(gamma + mu) )
        
            D_next_per_patient[step_ID] = D_per_patient[step_ID] * torch.exp(-torch.tensor(delta_t)) +\
                    (gamma * I_per_patient[step_ID] * torch.exp(-delta_t))/(gamma + mu + dis_rate - 1) * (1 - torch.exp(-(gamma + mu + dis_rate - 1)*delta_t)) +\
                    (mu*(alpha * E_per_patient[step_ID]))/(gamma + mu + dis_rate) * (1 - torch.exp(-delta_t) - (torch.exp(-delta_t) - torch.exp(-(gamma + mu + dis_rate)*delta_t))/(gamma + mu + dis_rate - 1))
        
            NS_next_per_patient[step_ID] = NS_per_patient[step_ID] * torch.exp(-dis_rate*delta_t)

        # All compartments
        all_comparts_next_per_patient[step_ID] = torch.stack([S_next_per_patient[step_ID], E_next_per_patient[step_ID], I_next_per_patient[step_ID], R_next_per_patient[step_ID], D_next_per_patient[step_ID], NS_next_per_patient[step_ID]], dim=1)
        all_comparts_current_GT_per_patient[step_ID] = torch.stack([S_per_patient[step_ID], E_per_patient[step_ID], I_per_patient[step_ID], R_per_patient[step_ID], D_per_patient[step_ID], NS_per_patient[step_ID]], dim=1)
                    
    # Getting the final state of each patient at each time step with the computed next steps
    future_states_per_patients_per_timestep = [{} for _ in range(n_timesteps)] # Between times 1 and n_timesteps + 1
    current_states_per_patients_per_timestep = [{} for _ in range(n_timesteps)] # Between times 0 and n_timesteps
    true_next_state_probs_all = [None for _ in range(n_timesteps)]
    predicted_state_probs_all = [None for _ in range(n_timesteps)]
    for step_ID in tqdm(range(n_timesteps)):
        tmp_n_nodes = all_comparts_next_per_patient[step_ID].shape[0]

        predicted_state_probs_all[step_ID] = all_comparts_next_per_patient[step_ID].numpy()
        true_next_state_probs_all[step_ID] = [None for _ in range(tmp_n_nodes)]
        
        next_states = torch.argmax(all_comparts_next_per_patient[step_ID], dim=1)
        current_states = torch.argmax(all_comparts_current_GT_per_patient[step_ID], dim=1)
        for local_node_ID in range(tmp_n_nodes):
            true_node_ID = true_patients_ids_mapping_per_timestep[step_ID][local_node_ID]

            next_state = get_state_probs_patient(states_probs_per_patient, step_ID+1, true_node_ID, inv_true_patients_ids_mapping_per_timestep)
            true_next_state_probs_all[step_ID][local_node_ID] = next_state
            
            future_states_per_patients_per_timestep[step_ID][true_node_ID] = next_states[local_node_ID]
            current_states_per_patients_per_timestep[step_ID][true_node_ID] = current_states[local_node_ID]
        true_next_state_probs_all[step_ID] = np.array(true_next_state_probs_all[step_ID])

    # Removing last step for next states predictions and GT as we do not have the GT (so true_next_state_probs_all[-1] is all np.nan)
    true_next_state_probs_all = true_next_state_probs_all[:-1]
    predicted_state_probs_all = predicted_state_probs_all[:-1]

    # Getting only the next states probs where the GT is non nan (so we have it)
    filtered_true_next_state_probs_all = []
    filtered_predicted_state_probs_all = []
    for step_ID in tqdm(range(len(true_next_state_probs_all))):
        # IMPORTANT: the "~" allows to have True if values are not nan when using np.isnan
        non_nan_values_idx = ~np.isnan(true_next_state_probs_all[step_ID]).any(axis=1)
        filtered_true_next_state_probs_all.append(true_next_state_probs_all[step_ID][non_nan_values_idx, :])
        filtered_predicted_state_probs_all.append(predicted_state_probs_all[step_ID][non_nan_values_idx, :])


    # Comparing the MSE per step for all the sample, and per sample
    step_MSEs = []
    step_MSEs_per_sample = []
    for step_ID in tqdm(range(len(filtered_true_next_state_probs_all))):
        # Global MSE of the step
        step_mse = np.mean((filtered_true_next_state_probs_all[step_ID] - filtered_predicted_state_probs_all[step_ID]) ** 2)
        step_MSEs.append(step_mse)
        # Per sample MSE of the step
        step_per_sample_mse = np.mean((filtered_true_next_state_probs_all[step_ID] - filtered_predicted_state_probs_all[step_ID]) ** 2, axis=1)
        step_MSEs_per_sample.append(step_per_sample_mse)


    # Plot evolution of the MSE per step
    if (plot_figures):
        plt.figure()
        plt.plot(list(range(len(step_MSEs))), step_MSEs)
        plt.xlabel("Time step")
        plt.ylabel("MSE")
        plt.title("Global MSE per time step")
        plt.legend()
        plt.show()

    # Printing useful information
    print(f"\n=========> Mean global MSE over all the steps: {np.mean(step_MSEs)}\n")
    #print(f"\n\t=========> Global MSE per step: {step_MSEs}\n")

    # Counting the total number of individuals in each compartment at each time step
    n_patient_per_compartment_predicted = [{'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0, 'NS': 0, } for _ in range(n_timesteps)] # Between times 1 and n_timesteps + 1
    n_patient_per_compartment_GT = [{'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0, 'NS': 0, } for _ in range(n_timesteps)] # Between times 1 and n_timesteps + 1
    for step_ID in tqdm(range(n_timesteps)):
        for patient_ID in future_states_per_patients_per_timestep[step_ID]:
            # Current state
            current_state = int(current_states_per_patients_per_timestep[step_ID][patient_ID]) 
            str_current_state = INV_MAPPING_STATES[current_state]
            n_patient_per_compartment_GT[step_ID][str_current_state] += 1
            
            # Future state
            future_state = int(future_states_per_patients_per_timestep[step_ID][patient_ID]) 
            str_future_state = INV_MAPPING_STATES[future_state]
            n_patient_per_compartment_predicted[step_ID][str_future_state] += 1

    # For comparison, we are going to get the data of both predicted and current states between 1 and n_timesteps + 1 
    # As we have a shift of one timestamp between the future predicted states probs and the current true ones, we have to correct it
    n_patient_per_compartment_predicted = n_patient_per_compartment_predicted[:-1]
    times_compartment_predicted = np.array([tmp_t*8/(24) for tmp_t in range(1, len(n_patient_per_compartment_predicted))])
    n_patient_per_compartment_GT = n_patient_per_compartment_GT[1:]
    times_compartment_GT = np.array([tmp_t*8/(24) for tmp_t in range(0, len(n_patient_per_compartment_predicted) - 1)])


    # Test if there are some wrong values (predicted probabilities greater than 1 or smaller than 0)
    wrong_values_per_state = {'S': {}, 'E': {}, 'I': {}, 'R': {}, 'D': {}, 'NS': {}}
    for step_ID in tqdm(range(len(all_comparts_next_per_patient))):
        for local_node_ID in range(all_comparts_next_per_patient[step_ID].shape[0]):
            for state_ID in range(all_comparts_next_per_patient[step_ID].shape[1]):
                str_state = INV_MAPPING_STATES[state_ID]
                state_prob = all_comparts_next_per_patient[step_ID][local_node_ID, state_ID]
                if (state_prob < 0) or (state_prob > 1):
                    #print(f"\n=========>PROBLEM: At time step {step_ID} we have node {local_node_ID} with a {str_state} compartment probability < 0 or > 1 ({state_prob})")
                    if (step_ID not in wrong_values_per_state[str_state]):
                        wrong_values_per_state[str_state][step_ID] = []
                    wrong_values_per_state[str_state][step_ID].append(state_prob)

    # Verifying if wrong values
    problem_predicted_probs_states = False
    for state_str in wrong_values_per_state:
        if (len(wrong_values_per_state[state_str]) > 0):
            print(f"\n=========>PROBLEM: For state {state_str} there are {len(wrong_values_per_state[state_str])} predicted probabilities < 0 or > 1")
            problem_predicted_probs_states = True

    if (not problem_predicted_probs_states):
        print("\n\n=========> Test passed <=========\n\n")

    # Verifying if wrong values
    problem_predicted_probs_states = False
    for state_str in wrong_values_per_state:
        if (len(wrong_values_per_state[state_str]) > 0):
            print(f"\n=========>PROBLEM: For state {state_str} there are {len(wrong_values_per_state[state_str])} predicted probabilities < 0 or > 1")
            problem_predicted_probs_states = True

    if (not problem_predicted_probs_states):
        print("\n\n=========> Test passed <=========\n\n")
    else:
        print("\n\n=========> Test FAILED <=========\n\n")
        print(wrong_values_per_state)

    # Comparing the simulated SEIRD-NS model with the one obtained by predicting the next state of each node using the current 
    # state and the Hazard waiting time model
    # Plotting boths results
    if (plot_figures):
        for state in list(n_patient_per_compartment_predicted[0].keys()):
            fig = plt.figure()
            data_true = np.array([n_patient_per_compartment_GT[step_ID][state] for step_ID in range(len(times_compartment_GT))])
            data_predicted = np.array([n_patient_per_compartment_predicted[step_ID][state] for step_ID in range(len(times_compartment_predicted))])
            plt.plot(times_compartment_GT, data_true, label=f"{state} True")
            plt.plot(times_compartment_predicted, data_predicted, label=f"{state} Predicted", linestyle='--')
            plt.legend()
            plt.xlabel("Time (hours)")
            plt.ylabel("Number of individuals")
            plt.show()
    


#====================================================================================================#
#=======================================Verification of ODEs=========================================#
#====================================================================================================#
def verify_ODEs(
                    times,
                    epidemiological_params,
                    patients_per_state_over_time,
                    plot_figures=True
                ):
    """
        Verify the validity of the ODEs by computing the right and left hand side terms of the
        equations and plotting them.

        Parameters:
        -----------
        times: np.array
            Array containing the time steps (in PER DAY unit) of the different measurments.
        epidemiological_params: dict
            Dictionary where the keys are the names of the epidemiological parameters
            and the values are their values.
        patients_per_state_over_time: dict
            Dictionary where the keys are the possible states (S, E, I, R, D, NS) and the values
            are list containing the total quantity of of patients in that state over time.
        plot_figures: bool  
            True if want to plot figures.
    """
    #======================================================================#
    #=========================Separating the states=========================#
    #======================================================================#
    S = np.array(patients_per_state_over_time[MAPPING_STATES['S']])
    E = np.array(patients_per_state_over_time[MAPPING_STATES['E']])
    I = np.array(patients_per_state_over_time[MAPPING_STATES['I']])
    R = np.array(patients_per_state_over_time[MAPPING_STATES['R']])
    D = np.array(patients_per_state_over_time[MAPPING_STATES['D']])
    NS = np.array(patients_per_state_over_time[MAPPING_STATES['NS']])
    N = S + E + I + R + D + NS

    #======================================================================#
    #===========================WITHOUT FILTERING===========================#
    #======================================================================#
    # Computation of numerical derivatives
    #dt = np.diff(times, n=1, prepend=np.nan)[0:]
    dt = np.diff(times, n=1)[0:]
    dt = np.where(np.isnan(dt), 0, dt)
    #dS = np.diff(S, n=1, prepend=np.nan)[0:] / dt
    dS = np.diff(S, n=1)[0:] / dt
    dS = np.where(np.isnan(dS), 0, dS)
    #dE = np.diff(E, n=1, prepend=np.nan)[0:] / dt
    dE = np.diff(E, n=1)[0:] / dt
    dE = np.where(np.isnan(dE), 0, dE)
    #dI = np.diff(I, n=1, prepend=np.nan)[0:] / dt
    dI = np.diff(I, n=1)[0:] / dt
    dI = np.where(np.isnan(dI), 0, dI)
    #dR = np.diff(R, n=1, prepend=np.nan)[0:] / dt
    dR = np.diff(R, n=1)[0:] / dt
    dR = np.where(np.isnan(dR), 0, dR)
    #dD = np.diff(D, n=1, prepend=np.nan)[0:] / dt
    dD = np.diff(D, n=1)[0:] / dt
    dD = np.where(np.isnan(dD), 0, dD)
    #dNS = np.diff(NS, n=1, prepend=np.nan)[0:] / dt
    dNS = np.diff(NS, n=1)[0:] / dt
    dNS = np.where(np.isnan(dNS), 0, dNS)

    # Epidemiological params
    A = epidemiological_params["A"] 
    A_S = epidemiological_params["A_S"] 
    A_E = epidemiological_params["A_E"] 
    A_I = epidemiological_params["A_I"] 
    A_R = epidemiological_params["A_R"] 
    A_NS = epidemiological_params["A_NS"] 
    BETA = epidemiological_params["BETA"] 
    MU = epidemiological_params["MU"] 
    DIS_RATE = epidemiological_params["DIS_RATE"]
    ALPHA = epidemiological_params["ALPHA"] 
    GAMMA = epidemiological_params["GAMMA"]


    # Computing theoretical derivates based on the SEIR model
    dS_model = -BETA * S * I / N + A_S*A - DIS_RATE*S
    dE_model = BETA * S * I / N + A_E*A - (ALPHA + DIS_RATE) * E
    dI_model = ALPHA * E + A_I*A - (GAMMA + MU + DIS_RATE) * I
    dR_model = GAMMA * I + A_R*A - DIS_RATE*R
    dD_model = MU * I - D
    dNS_model = A_NS*A - D*NS

    # Comparison with numerical derivatives
    mse_S = mean_squared_error(dS, dS_model[1:])
    mse_E = mean_squared_error(dE, dE_model[1:])
    mse_I = mean_squared_error(dI, dI_model[1:])
    mse_R = mean_squared_error(dR, dR_model[1:])
    print("\n=======>MSE between the numerical derivates (LHS) and the theoretical ones (RHS):")
    print("\t For S: {}".format(mse_S))
    print("\t For E: {}".format(mse_E))
    print("\t For I: {}".format(mse_I))
    print("\t For R: {}".format(mse_R))

    # Plot both to do a visual comparison
    if (plot_figures):
        INIT_SAMPLE_PLOT, LAST_SAMPLE_PLOT = 0, -1
        # For S
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dS[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dS (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dS_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dS (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of S: numerical vs model WITHOUT FILTERING")
        plt.show()

        # For E
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dE[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dE (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dE_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dE (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of E: numerical vs model WITHOUT FILTERING")
        plt.show()

        # For I
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dI[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dI (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dI_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dI (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of I: numerical vs model WITHOUT FILTERING")
        plt.show()

        # For R
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dR[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dR (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dR_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dR (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of R: numerical vs model WITHOUT FILTERING")
        plt.show()

        # For D
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dD[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dD (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dD_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dD (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of D: numerical vs model WITHOUT FILTERING")
        plt.show()

        # For NS
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dNS[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dNS (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dNS_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dNS (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of NS: numerical vs model WITHOUT FILTERING")
        plt.show()


    #======================================================================#
    #============================WITH FILTERING============================#
    #======================================================================#
    # Estimation of time derivaties
    states_over_time = np.stack([S, E, I, R, D, NS], axis=1)
    #dt = 1
    dt = 0.33
    SMOOTHING_OBS_DERIVATIVES = True
    #SMOOTHING_OBS_DERIVATIVES = False
    if (SMOOTHING_OBS_DERIVATIVES):
        dX_dt_obs = np.array([
                                savgol_filter(states_over_time[:, c], window_length=7, polyorder=2, deriv=1, delta=dt, axis=0)
                                for c in range(6)
                            ]).transpose(1, 0)
        dX_dt_obs = dX_dt_obs[1:, :]
    else:
        dX_dt_obs = (states_over_time[1:, :] - states_over_time[:-1, :]) / dt 
    dS = dX_dt_obs[:, MAPPING_STATES['S']]
    dE = dX_dt_obs[:, MAPPING_STATES['E']]
    dI = dX_dt_obs[:, MAPPING_STATES['I']]
    dR = dX_dt_obs[:, MAPPING_STATES['R']]
    dD = dX_dt_obs[:, MAPPING_STATES['D']]
    dNS = dX_dt_obs[:, MAPPING_STATES['NS']]

    # Plot both to do a visual comparison
    if (plot_figures):
        INIT_SAMPLE_PLOT, LAST_SAMPLE_PLOT = 0, -1
        #INIT_SAMPLE_PLOT, LAST_SAMPLE_PLOT = 500, 1000
        # For S
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dS[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dS (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dS_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dS (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of S: numerical vs model WITH FILTERING")
        plt.show()

        # For E
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dE[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dE (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dE_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dE (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of E: numerical vs model WITH FILTERING")
        plt.show()

        # For I
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dI[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dI (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dI_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dI (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of I: numerical vs model WITH FILTERING")
        plt.show()

        # For R
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dR[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dR (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dR_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dR (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of R: numerical vs model WITH FILTERING")
        plt.show()

        # For D
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dD[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dD (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dD_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dD (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of D: numerical vs model WITH FILTERING")
        plt.show()

        # For NS
        plt.plot(times[:-1][INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dNS[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dNS (numerical)')
        plt.plot(times[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], dNS_model[INIT_SAMPLE_PLOT:LAST_SAMPLE_PLOT], label='dNS (model)', linestyle='--')
        plt.legend()
        plt.title("Derivative of NS: numerical vs model WITH FILTERING")
        plt.show()


#====================================================================================================#
#============================================Main function============================================#
#====================================================================================================#
def main():
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser patients_data_path
    ap.add_argument('--folder_to_store_data', default='./data/murcia/preprocessed/', help="Folder where the generated graph data structure will be saved.")
    ap.add_argument('--movement_data_path', default='./data/murcia/raw_data/movements_0.csv', help="Path to the CSV file containing the movements of the patients over time.")
    ap.add_argument('--locations_data_path', default='./data/murcia/raw_data/locations_0.csv', help="Path to the CSV file describing the hospital environment and locations.")
    ap.add_argument('--patients_data_path', default='./data/murcia/raw_data/patients_0.csv', help="Path to the CSV file describing the different patients in the simulation.")
    ap.add_argument('--dataset_store_name', default='Dataset_0', help="Name to use to store the dataset (name of the folder where all the generated files are going to be stored).", type=str)
    ap.add_argument('--create_list_nodes_edges', required=False, help="True if want to create list of nodes and edges information for Pytorch Geomtric datasets", action='store_true')
    ap.add_argument('--fully_connect_patients_same_place', required=False, help="If True, all the patients that are in the same places are connected with a weight of 1 or 1/n_patient_in_place.", action='store_true')
    ap.add_argument('--uniform_weights_patients_to_patient_edge', required=False, help="If true, the weights between two patients is 1/n_patient_in_place with n_patient_in_place is the number of patients that are in the sample place as those patients.", action='store_true')
    ap.add_argument('--forecast_horizon', default=1, required=False, help="Forecast horizon used to define the targets of each node.", type=int)
    ap.add_argument('--create_H5_file', required=False, help="True if want to create HDF5 data file.", action='store_true')
    ap.add_argument('--use_estimated_values_ODEs_verification', required=False, help="True if want to use the estimated epidemic parameters from data for the ODEs verification.", action='store_true')
    ap.add_argument('--plot_figures', required=False, help="True if want to plot figures about the states evolution, differential equations, etc.", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    folder_to_store_data = args['folder_to_store_data']
    movement_data_path = args['movement_data_path']
    locations_data_path = args['locations_data_path']
    patients_data_path = args['patients_data_path']
    dataset_store_name = args['dataset_store_name']
    create_list_nodes_edges = args['create_list_nodes_edges']
    fully_connect_patients_same_place = args['fully_connect_patients_same_place']
    uniform_weights_patients_to_patient_edge = args['uniform_weights_patients_to_patient_edge']
    forecast_horizon = args['forecast_horizon']
    create_H5_file = args['create_H5_file']
    use_estimated_values_ODEs_verification = args['use_estimated_values_ODEs_verification']
    plot_figures = args['plot_figures']
    

    #======================================================================#
    #======================Load locations information======================#
    #======================================================================#
    # Loading the data
    locations_info, main_parent_places = load_locations_data(locations_data_path=locations_data_path)

    #======================================================================#
    #======================Load movement information======================#
    #======================================================================#
    # Loading the data
    movement_data_restructured,\
    places_list,\
    specific_places_list,\
    rooms_ward_mapping,\
    one_hot_enc_main_places,\
    inv_one_hot_enc_main_places,\
    one_hot_enc_specific_places,\
    inv_one_hot_enc_specific_places = load_movement_data(movement_data_path, locations_info)

    # Get some information about the simulation
    simulation_information(movement_data_restructured, locations_info)

    # Get the total number of patients per state over the simulation
    patients_per_state_over_time = get_patients_per_state_over_time(movement_data_restructured, plot_evolution=plot_figures)

    # Number of contaminated places per step
    n_contaminated_places_per_step = get_number_contaminated_places_per_step(movement_data_restructured, plot_evolution=plot_figures)

    # Number of to clean places per step
    n_places_to_clean_per_step = get_number_places_to_clean_per_step(movement_data_restructured, plot_evolution=plot_figures)

    #======================================================================#
    #=======================Load patient information=======================#
    #======================================================================#
    # Loading as a Pandas DataFrame
    patients_data_df = get_patients_data(patients_data_path)

    # Number of patients
    print(f"\n========> Total number of patients during the whole period of the simulation: {len(patients_data_df)}\n")

    # See some of the data
    print("\nExample of patient data: ", patients_data_df.head(10))


    #======================================================================#
    #=============================Plot graphs=============================#
    #======================================================================#
    if (plot_figures):
        # Graph without the places
        plot_nx_graphs(
                        movement_data_restructured,
                        places_as_nodes=False,
                        fully_connect_patients_same_place=True,
                        n_graphs_plot=2
                    )

        # Graph with the places and fully connected patients in the same place
        plot_nx_graphs(
                        movement_data_restructured,
                        places_as_nodes=True,
                        fully_connect_patients_same_place=True,
                        n_graphs_plot=2
                    )

        # Graph with the places and NOT fully connected patients in the same place
        plot_nx_graphs(
                        movement_data_restructured,
                        places_as_nodes=True,
                        fully_connect_patients_same_place=False,
                        n_graphs_plot=2
                    )


    #======================================================================#
    #===============Data re-structuring for Pytorch Geometric===============#
    #======================================================================#
    # Creating folder to store the new dataset
    if (create_list_nodes_edges):
        if (os.path.exists(folder_to_store_data + f"/{dataset_store_name}/")):
            i = 0
            while os.path.exists(folder_to_store_data + f"/{dataset_store_name}_{i}/"):
                i += 1
            folder_to_store_data = folder_to_store_data + f"/{dataset_store_name}_{i}/"
        else:
            folder_to_store_data = folder_to_store_data + f"/{dataset_store_name}/"
        os.mkdir(folder_to_store_data)
    else:
        if (not os.path.exists(folder_to_store_data + f"/{dataset_store_name}/")):
            raise RuntimeError(f"\n If you want to use pre-computed lists of nodes and edges, you should give a valid combination of folder_to_store_data and dataset_store_name.\n")
        else:
            folder_to_store_data = folder_to_store_data + f"/{dataset_store_name}/"
        

    # Creating the graph structure
    FORECAST_WINDOW_LENGTH = forecast_horizon
    #FORECAST_WINDOW_LENGTH = 1
    #FORECAST_WINDOW_LENGTH = 7
    if (create_list_nodes_edges):
        # Getting the lists
        list_edge_index_dicts,\
        list_edge_weight_dicts,\
        list_node_feature_dicts,\
        list_node_target_dicts,\
        list_node_timestamps_dicts,\
        list_node_inf_risk_target_dicts,\
        list_node_transition_target_dicts,\
        list_nodes_ids_dicts,\
        patient_features_names_idx,\
        places_features_names_idx = create_lists_nodes_edges(    
                                                                movement_data=movement_data_restructured,
                                                                locations_info=locations_info,
                                                                patients_data_df=patients_data_df,
                                                                one_hot_enc_main_places=one_hot_enc_main_places,
                                                                one_hot_enc_specific_places=one_hot_enc_specific_places,
                                                                forecast_window_length=FORECAST_WINDOW_LENGTH,
                                                                fully_connect_patients_same_place=fully_connect_patients_same_place,
                                                                uniform_weights_patients_to_patient_edge=uniform_weights_patients_to_patient_edge
                                                            )
        # Saving the created data for the heterogeneous graph
        i = 0
        fn_list_edge_index_dicts = f"{folder_to_store_data}/list_edge_index_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_"
        while (os.path.exists(f"{fn_list_edge_index_dicts}_{i}.pkl")):
            i += 1
        
        fn_list_edge_index_dicts = f"{folder_to_store_data}/list_edge_index_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_" + str(i) + ".pkl"
        with open(fn_list_edge_index_dicts, "wb") as fp:
            pickle.dump(list_edge_index_dicts, fp)
        
        fn_list_edge_weight_dicts = f"{folder_to_store_data}/list_edge_weight_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_edge_weight_dicts, "wb") as fp:
            pickle.dump(list_edge_weight_dicts, fp)
        
        fn_list_node_feature_dicts = f"{folder_to_store_data}/list_node_feature_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_node_feature_dicts, "wb") as fp:
            pickle.dump(list_node_feature_dicts, fp)
        
        fn_list_node_target_dicts = f"{folder_to_store_data}/list_node_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_node_target_dicts, "wb") as fp:
            pickle.dump(list_node_target_dicts, fp)
        
        fn_list_node_timestamps_dicts = f"{folder_to_store_data}/list_node_timestamps_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_node_timestamps_dicts, "wb") as fp:
            pickle.dump(list_node_timestamps_dicts, fp)

        fn_list_node_inf_risk_target_dicts = f"{folder_to_store_data}/list_node_inf_risk_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_node_inf_risk_target_dicts, "wb") as fp:
            pickle.dump(list_node_inf_risk_target_dicts, fp)

        fn_list_node_transition_target_dicts = f"{folder_to_store_data}/list_node_transition_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_node_transition_target_dicts, "wb") as fp:
            pickle.dump(list_node_transition_target_dicts, fp)
        
        fn_list_nodes_ids_dicts = f"{folder_to_store_data}/list_nodes_ids_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_{i}.pkl"
        with open(fn_list_nodes_ids_dicts, "wb") as fp:
            pickle.dump(list_nodes_ids_dicts, fp)

        # Patients features mapping dict
        i = 0 
        fn_patient_features_names_idx = f"{folder_to_store_data}/patient_features_names_idx_"
        while (os.path.exists(fn_patient_features_names_idx + str(i) + '.pkl')):
            i += 1
        fn_patient_features_names_idx = fn_patient_features_names_idx + str(i) + '.pkl'
        with open(fn_patient_features_names_idx, "wb") as fp:
            pickle.dump(patient_features_names_idx, fp)
        
        # Places features mapping dict
        i = 0 
        fn_places_features_names_idx = f"{folder_to_store_data}/place_features_names_idx_"
        while (os.path.exists(fn_places_features_names_idx + str(i) + '.pkl')):
            i += 1
        fn_places_features_names_idx = fn_places_features_names_idx + str(i) + '.pkl'
        with open(fn_places_features_names_idx, "wb") as fp:
            pickle.dump(places_features_names_idx, fp)

        print(f"\n\n==========> List nodes and edges files created in {folder_to_store_data}\n")
            
    else:
        # Load data
        fn_list_edge_index_dicts = f"{folder_to_store_data}/list_edge_index_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_edge_index_dicts, mode='rb') as pf:
            list_edge_index_dicts = pickle.load(pf)
        fn_list_edge_weight_dicts = f"{folder_to_store_data}/list_edge_weight_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_edge_weight_dicts, mode='rb') as pf:
            list_edge_weight_dicts = pickle.load(pf)
        fn_list_node_feature_dicts = f"{folder_to_store_data}/list_node_feature_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_node_feature_dicts, mode='rb') as pf:
            list_node_feature_dicts = pickle.load(pf)
        fn_list_node_target_dicts = f"{folder_to_store_data}/list_node_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_node_target_dicts, mode='rb') as pf:
            list_node_target_dicts = pickle.load(pf)
        fn_list_node_timestamps_dicts = f"{folder_to_store_data}/list_node_timestamps_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_node_timestamps_dicts, mode='rb') as pf:
            list_node_timestamps_dicts = pickle.load(pf)
        fn_list_node_inf_risk_target_dicts = f"{folder_to_store_data}/list_node_inf_risk_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_node_inf_risk_target_dicts, mode='rb') as pf:
            list_node_inf_risk_target_dicts = pickle.load(pf)
        fn_list_node_transition_target_dicts = f"{folder_to_store_data}/list_node_transition_target_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_node_transition_target_dicts, mode='rb') as pf:
            list_node_transition_target_dicts = pickle.load(pf)
        fn_list_nodes_ids_dicts = f"{folder_to_store_data}/list_nodes_ids_dicts_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_0.pkl"
        with open(fn_list_nodes_ids_dicts, mode='rb') as pf:
            list_nodes_ids_dicts = pickle.load(pf)
        fn_patient_features_names_idx = f"{folder_to_store_data}/patient_features_names_idx_0.pkl"
        with open(fn_patient_features_names_idx, mode='rb') as pf:
            patient_features_names_idx = pickle.load(pf)
        fn_places_features_names_idx = f"{folder_to_store_data}/place_features_names_idx_0.pkl"
        with open(fn_places_features_names_idx, mode='rb') as pf:
            places_features_names_idx = pickle.load(pf)
        
    #======================================================================#
    #================Estimate epidemic parameters from data================#
    #======================================================================#
    # Estimation of epidemic parameters from data
    #RATES_IN_PER_DAY = False
    RATES_IN_PER_DAY = True
    epidemiological_params_for_csv = estimate_epidemic_params_from_data(
                                                                            patients_data_df=patients_data_df,
                                                                            movement_data=movement_data_restructured,
                                                                            rates_in_per_day=RATES_IN_PER_DAY
                                                                        )
    

    #======================================================================#
    #======================Creating HDF5 file if asked======================#
    #======================================================================#
    # Create HDF5 file
    # File name
    h5_fn = folder_to_store_data + f"/MurciaGraphData_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_0.hdf5"
    if (create_H5_file):
        # HDF5 creation
        h5_fn = create_HDF5_file(
                                    h5_fn,
                                    list_edge_index_dicts,
                                    list_edge_weight_dicts,
                                    list_node_feature_dicts,
                                    list_node_target_dicts,
                                    list_node_timestamps_dicts,
                                    list_node_inf_risk_target_dicts,
                                    list_node_transition_target_dicts,
                                    list_nodes_ids_dicts
                                )

        print(f"\n\n==========> HDF5 file created in {h5_fn}\n")
    else:
        h5_fn = folder_to_store_data + f"/MurciaGraphData_FullyConnected-{fully_connect_patients_same_place}_UniformPatToPatEdges-{uniform_weights_patients_to_patient_edge}_ForecastHorizon-{FORECAST_WINDOW_LENGTH}_0_0.hdf5"
    
    # Load HDF5 file dataset
    h5_murcia_dataset = h5py.File(h5_fn, 'r')

    # Get the lists of dicts to create the Murcia Heterogeneous Dataset
    n_steps = len(h5_murcia_dataset["EdgeFeaturesDicts"])
    py_murcia_data_from_h5 = {
                                "EdgeIndexDicts": [None for _ in range(n_steps)],
                                "EdgeFeaturesDicts": [None for _ in range(n_steps)],
                                "NodeFeaturesDicts": [None for _ in range(n_steps)],
                                "NodeTargetsDict": [None for _ in range(n_steps)],
                                "NodeTimestampsDicts": [None for _ in range(n_steps)],
                                "NodeInfRiskTargetsDict": [None for _ in range(n_steps)],
                                "NodeTargetsTransitionsDict": [None for _ in range(n_steps)],
                                "NodesIdsDicts": [None for _ in range(n_steps)]
                            }
    for main_group in py_murcia_data_from_h5:
        for str_step_ID in tqdm(h5_murcia_dataset[main_group]):
            step_ID = int(str_step_ID)
            keys = list(h5_murcia_dataset[main_group][str_step_ID])
            py_murcia_data_from_h5[main_group][step_ID] = {}
            for key in keys:
                if (len(key.split('-')) == 2):
                    new_key = tuple(key.split('-'))
                else:
                    new_key = key
                py_murcia_data_from_h5[main_group][step_ID][new_key] = h5_murcia_dataset[main_group][str_step_ID][key][:]

    # Close HDF5 file
    h5_murcia_dataset.close()

    # Creating heterogeneous Pytorch Geometric dataset
    other_attributes = {
                            "inf_risk_targets_dicts": py_murcia_data_from_h5["NodeInfRiskTargetsDict"],
                            "transitions_targets": py_murcia_data_from_h5["NodeTargetsTransitionsDict"]
                       }
    whole_dataset = DynamicHeteroGraphTemporalSignal(
                                                edge_index_dicts=py_murcia_data_from_h5["EdgeIndexDicts"],
                                                edge_weight_dicts=py_murcia_data_from_h5["EdgeFeaturesDicts"],
                                                feature_dicts=py_murcia_data_from_h5["NodeFeaturesDicts"],
                                                target_dicts=py_murcia_data_from_h5["NodeTargetsDict"],
                                                timestamps=py_murcia_data_from_h5["NodeTimestampsDicts"],
                                                ids=py_murcia_data_from_h5["NodesIdsDicts"],
                                                **other_attributes
                                             )
    times = np.array([tmp_t*8/(24) for tmp_t in range(len(whole_dataset.feature_dicts))])

    
    #======================================================================#
    #=============Verification of Hazard discretiezd equations=============#
    #======================================================================#
    # Parameters of the differential equations
    if (use_estimated_values_ODEs_verification):
        epidemiological_params = epidemiological_params_for_csv
    else: # Using values in the article
        A = 18.603
        A_S = 0.997
        A_E = 0
        A_I = 0.002
        A_R = 0
        A_NS = 0.001
        
        BETA = 1/(0.435 + 0.24)
        
        MU = 0.027
        
        LOS = 4.254
        DIS_RATE = 1/LOS # d IN THE DIFFERENTIAL EQUATIONS
        
        ALPHA = 1/2.5
        
        GAMMA = 1 - MU

        epidemiological_params = {
                                    "A": A, 
                                    "A_S": A_S, 
                                    "A_E": A_E, 
                                    "A_I": A_I, 
                                    "A_R": A_R, 
                                    "A_NS": A_NS, 
                                    
                                    "BETA": BETA, 
                                    
                                    "MU": MU, 
                                    
                                    "DIS_RATE": DIS_RATE,
                                    
                                    "ALPHA": ALPHA, 
                                    
                                    "GAMMA": GAMMA
                                }
    verification_hazard_discrete_equations(
                                            movement_data=movement_data_restructured,
                                            epidemiological_params=epidemiological_params,
                                            plot_figures=plot_figures
                                          )
    
    #======================================================================#
    #=========================Verification of ODEs=========================#
    #======================================================================#
    # Getting the data per time step
    # USING THE DATA IN THE HETEROGENOUS PYTORCH GEOMETRIC DATASET
    S = [0 for _ in range(len(whole_dataset.feature_dicts))]
    E = [0 for _ in range(len(whole_dataset.feature_dicts))]
    I = [0 for _ in range(len(whole_dataset.feature_dicts))]
    R = [0 for _ in range(len(whole_dataset.feature_dicts))]
    D = [0 for _ in range(len(whole_dataset.feature_dicts))]
    NS = [0 for _ in range(len(whole_dataset.feature_dicts))]
    times = np.array([tmp_t*8/(24) for tmp_t in range(len(whole_dataset.feature_dicts))])
    n_steps = len(whole_dataset.feature_dicts)
    init_state_idx, end_state_idx = patient_features_names_idx['current_state'][0], patient_features_names_idx['current_state'][1]
    for step_ID in range(n_steps):
        #patients_states = [int(whole_dataset.feature_dicts[step_ID]['Patient'][pat_idx][-1]) for pat_idx in range(len(whole_dataset.feature_dicts[step_ID]['Patient']))]
        patients_states = [None for _ in range(len(whole_dataset.feature_dicts[step_ID]['Patient']))]
        for pat_idx in range(len(whole_dataset.feature_dicts[step_ID]['Patient'])):
            patients_states[pat_idx] = MAPPING_STATES[INV_ONE_HOT_PATIENT_STATES[tuple(whole_dataset.feature_dicts[step_ID]['Patient'][pat_idx][init_state_idx: end_state_idx])]]
        counter_states = Counter(patients_states)




        for state_idx in counter_states:
            if (INV_MAPPING_STATES[state_idx] == 'S'):
                S[step_ID] += counter_states[state_idx]
            elif (INV_MAPPING_STATES[state_idx] == 'E'):
                E[step_ID] += counter_states[state_idx]
            elif (INV_MAPPING_STATES[state_idx] == 'I'):
                I[step_ID] += counter_states[state_idx]
            elif (INV_MAPPING_STATES[state_idx] == 'R'):
                R[step_ID] += counter_states[state_idx]
            elif (INV_MAPPING_STATES[state_idx] == 'D'):
                D[step_ID] += counter_states[state_idx]
            elif (INV_MAPPING_STATES[state_idx] == 'NS'):
                NS[step_ID] += counter_states[state_idx]
    # Transform to numpy arrays
    S = np.array(S)
    E = np.array(E)
    I = np.array(I)
    R = np.array(R)
    D = np.array(D)
    NS = np.array(NS)
    N = S + E + I + R + D + NS

    # Verification ODEs
    new_patients_per_state_over_time = {0: S, 1: E, 2: I, 3: R, 4: D, 5: NS}
    verify_ODEs(
                    times=times,
                    epidemiological_params=epidemiological_params,
                    patients_per_state_over_time=new_patients_per_state_over_time,
                    plot_figures=plot_figures
                )





if __name__=="__main__":
    main()