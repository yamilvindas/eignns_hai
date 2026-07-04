import sys
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
sns.set_style("darkgrid")


def get_feature_in_feature_vector(feature_vector, feature_name, feature_names_idx_dict):
    """
        Gets the value of a given feature within a feature vector.

        Parameters:
        -----------
        feature_vector: np.array
            Feature vector of shape (n_nodes, all_features_dim) if it corresponds
            to the feature of a set of nodes, or of shape (all_features_dim) if
            it is the feature vector of a single node.
        feature_name: str
            Name of the feature from which we want to get its value.
        feature_names_idx_dict: 
            Dictionary where the keys are the names of the features, and the
            values are the starting and ending index of that feature in the
            feature vector.
        

        Returns:
        --------
        feature_value: np.array
            Value of the selected feature. If feature vector is the
            feature vector for a set of nodes (so of shape (n_nodes, all_features_dim))
            then its shape is (n_nodes, feature_dim), if not its shape is (feature_dim)
    """
    # Get starting and ending indices
    start_idx = feature_names_idx_dict[feature_name][0]
    end_idx = feature_names_idx_dict[feature_name][1]

    # Getting the feature
    if (len(feature_vector.shape) == 2):
        feature_value = feature_vector[:, start_idx:end_idx]
    else:
        feature_value = feature_vector[start_idx:end_idx]

    return feature_value

def get_statistics_train_dataset_murcia(py_murcia_data_from_h5,
                                        train_steps_IDs,
                                        patient_features_names_idx,
                                        place_features_names_idx):
    """
        Computes some statistics necessary for normalizing a Murcia dataset,
        based on the training set.

        Parameters:
        -----------
        py_murcia_data_from_h5: dict
            Dictionary containing the following keys: EdgeIndexDicts,
            EdgeFeaturesDicts, NodeFeaturesDicts, NodeTargetsDict, 
            NodeTimestampsDicts, NodesIdsDicts. Each value is a list
            containing the data for each time step in the simulation.
        train_steps_IDs: list
            List with the IDs of the training steps.
        patient_features_names_idx: dict
            Dictionary indicating the position of each feature on the
            feature vector for patients nods.
        place_features_names_idx: dict
            Dictionary indicating the position of each feature on the
            feature vector for places nods.
    """
    # Age
    static_features_per_patient = {
                                        "Age": {},
                                        'AdmissionDay': {}
                                  }
    dynamic_features_per_patient = {
                                        "los_in_place": {},
                                        "n_patients_same_place": {}
                                   }
    dynamic_features_per_place = {'nPatients': {}}
    #for step_ID in tqdm(range(N_STEPS_TRAIN)):
    for step_ID in tqdm(train_steps_IDs):
        #======================================================================#
        # IDs of the patients and places in the current step
        patients_IDs_current_step = py_murcia_data_from_h5["NodesIdsDicts"][step_ID]['Patient']
        palces_IDs_current_step = py_murcia_data_from_h5["NodesIdsDicts"][step_ID]['Place']
    
        #======================================================================#
        # Train ages
        train_ages_current_step = get_feature_in_feature_vector(
                                                                    feature_vector=py_murcia_data_from_h5["NodeFeaturesDicts"][step_ID]['Patient'],
                                                                    feature_name="Age",
                                                                    feature_names_idx_dict=patient_features_names_idx
                                                               )
        for i in range(patients_IDs_current_step.shape[0]):
            patient_ID = patients_IDs_current_step[i]
            if (patient_ID not in static_features_per_patient['Age']):
                static_features_per_patient['Age'][patient_ID] = train_ages_current_step[i]
        
        #======================================================================#
        # Train AdmissionDay
        train_adim_days_current_step = get_feature_in_feature_vector(
                                                                        feature_vector=py_murcia_data_from_h5["NodeFeaturesDicts"][step_ID]['Patient'],
                                                                        feature_name="AdmissionDay",
                                                                        feature_names_idx_dict=patient_features_names_idx
                                                                    )
        for i in range(patients_IDs_current_step.shape[0]):
            patient_ID = patients_IDs_current_step[i]
            if (patient_ID not in static_features_per_patient['AdmissionDay']):
                static_features_per_patient['AdmissionDay'][patient_ID] = train_adim_days_current_step[i]
    
        #======================================================================#
        # Train los_in_place
        train_los_in_place_current_step = get_feature_in_feature_vector(
                                                                    feature_vector=py_murcia_data_from_h5["NodeFeaturesDicts"][step_ID]['Patient'],
                                                                    feature_name="los_in_place",
                                                                    feature_names_idx_dict=patient_features_names_idx
                                                               )
        for i in range(patients_IDs_current_step.shape[0]):
            patient_ID = patients_IDs_current_step[i]
            if (patient_ID not in dynamic_features_per_patient['los_in_place']):
                dynamic_features_per_patient['los_in_place'][patient_ID] = train_los_in_place_current_step[i]
    
        #======================================================================#
        # Train n_patients_same_place
        train_n_patients_same_place_current_step = get_feature_in_feature_vector(
                                                                    feature_vector=py_murcia_data_from_h5["NodeFeaturesDicts"][step_ID]['Patient'],
                                                                    feature_name="n_patients_same_place",
                                                                    feature_names_idx_dict=patient_features_names_idx
                                                               )
        for i in range(patients_IDs_current_step.shape[0]):
            patient_ID = patients_IDs_current_step[i]
            if (patient_ID not in dynamic_features_per_patient['n_patients_same_place']):
                dynamic_features_per_patient['n_patients_same_place'][patient_ID] = train_n_patients_same_place_current_step[i]
    
    
        #======================================================================#
        # Train nPatients for a place
        train_nPatients_current_step = get_feature_in_feature_vector(
                                                                            feature_vector=py_murcia_data_from_h5["NodeFeaturesDicts"][step_ID]['Place'],
                                                                            feature_name="nPatients",
                                                                            feature_names_idx_dict=place_features_names_idx
                                                                       )
        for i in range(palces_IDs_current_step.shape[0]):
            place_ID = palces_IDs_current_step[i]
            if (place_ID not in dynamic_features_per_place['nPatients']):
                dynamic_features_per_place['nPatients'][place_ID] = train_nPatients_current_step[i]

    return static_features_per_patient, dynamic_features_per_patient, dynamic_features_per_place