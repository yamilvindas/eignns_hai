#!/usr/bin/env python3
"""
    Code to generate synthetic SocioPatterns datasets
"""
import os
import argparse
import h5py
import pickle
from tqdm import tqdm

from src.DataManipulation.socio_patterns.generate_single_dataset import raw_data_loading,\
                                                                        get_extended_data,\
                                                                        simulate_per_individual_SEIR,\
                                                                        plot_SEIR_evolution,\
                                                                        analyze_parameters_distribution,\
                                                                        analyze_ODEs
       

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
def main():
    #==========================================================================#
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--raw_data_folder', required=True, help="Path to the SocioPatterns  folder containing the SFHH and Hospital Ward raw data", type=str)
    # nargs='+' allows to specify that at least one value is expected
    ap.add_argument('--list_n_days_use', nargs='+', default=[50, 100, 200], help="List of at least 3 number of days to use for the datasets (extend the days to that number of days). Each value on the list generates one DS", type=int)
    ap.add_argument('--n_datasets_generate', required=True, help="Number of sets of datasets to generate (one set correspond to all the datasets associated to the values given in list_n_days_use)", type=int)
    ap.add_argument('--do_plots', help="Use it if want to get some basic plots", action='store_true')
    ap.add_argument('--analyze_generated_data', help="Use it if want to analyze the generated data", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    raw_data_folder = args['raw_data_folder']
    list_n_days_use = sorted(args['list_n_days_use'])
    if (len(list_n_days_use) != 1) and (len(list_n_days_use) != 3):
        raise ValueError("\n--list_n_days_use should contain exactly 3 values to define the train, val, and test datasets. If only one value is given, it will be repeated 3 times, so the three splits will have the same number of days\n")
    if (len(list_n_days_use) == 1):
        list_n_days_use = list_n_days_use*3
    print(f"\nWe are going to generate SocioPatterns datasets with the following number of days: {list_n_days_use}\n")
    n_datasets_generate = args['n_datasets_generate']
    do_plots = args['do_plots']
    analyze_generated_data = args['analyze_generated_data']


    #==========================================================================#
    #==========================================================================#
    # Loading raw data
    raw_data_path_sfhh = raw_data_folder + "/SFHH/tij_SFHH.dat"
    sfhh_data = raw_data_loading(raw_data_path=raw_data_path_sfhh, do_plots=do_plots)
    raw_data_path_hospital_ward = raw_data_folder + "/HospitalWard/detailed_list_of_contacts_Hospital.dat"
    hospital_ward_data = raw_data_loading(raw_data_path=raw_data_path_hospital_ward, do_plots=do_plots)

    #==========================================================================#
    #==========================================================================#
    for dataset_ID in tqdm(range(n_datasets_generate)):
        print(f"\n\n======================================================================")
        print(f"=====================Generating dataset {dataset_ID}=====================")
        print(f"======================================================================\n\n")
        train_days_ID = 2
        train_data_file_config_1 = None
        train_data_file_config_2 = None
        val_days_ID = 0
        val_data_file_config_1 = None
        val_data_file_config_2 = None
        test_days_ID = 1
        test_data_file_config_1 = None
        test_data_file_config_2 = None
        for days_ID in range(3):
            for dataset_type in ['SFHH', 'HospitalWard']:
                n_days_use = list_n_days_use[days_ID]
                print(f"\t==========Generating subdataset for {dataset_type} with {n_days_use} days to use==========\n")
                #==========================================================================#
                # Extend days in the data
                if (dataset_type.lower() == 'sfhh'):
                    data = sfhh_data
                    n_inital_infected = 100
                else:
                    data = hospital_ward_data
                    n_inital_infected = 20
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
                                                                                    # n_simulations=1,
                                                                                    n_simulations=2,
                                                                                    #n_simulations=10,
                                                                                    fix_features=True,
                                                                                    #fix_features=False,
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
                if (days_ID == train_days_ID):
                    if (dataset_type.lower() == 'sfhh'):
                        train_data_file_config_1 = dataset_filename
                    else:
                        train_data_file_config_2 = dataset_filename
                elif (days_ID == val_days_ID):
                    if (dataset_type.lower() == 'sfhh'):
                        val_data_file_config_1 = dataset_filename
                    else:
                        val_data_file_config_2 = dataset_filename
                elif (days_ID == test_days_ID):
                    # IMPORTANT: WE TEST IN ANOTHER INTERACTIONS DISTRIBUTION DATASET
                    if (dataset_type.lower() == 'hospitalward'): # IMPORTANT: WE TEST IN ANOTHER INTERACTIONS DISTRIBUTION DATASET
                        test_data_file_config_1 = dataset_filename
                    else:
                        test_data_file_config_2 = dataset_filename


        # Writing HDF5 files
        i = 0
        # Configuration 1
        hdf5_file_path_config_1 = "./data/SocioPatterns/socio_patterns_config_1_"
        while (os.path.exists(hdf5_file_path_config_1 + str(i) + ".hdf5") ):
            i += 1
        hdf5_file_path_config_1 = hdf5_file_path_config_1 + str(i) + ".hdf5"
        # Creating the HDF5 file
        with h5py.File(hdf5_file_path_config_1, "w") as hdf5_file_config_1:
            # Creating group for each data split
            train = hdf5_file_config_1.create_group(f"train")
            train.attrs['Path'] = train_data_file_config_1
            val = hdf5_file_config_1.create_group(f"val")
            val.attrs['Path'] = val_data_file_config_1
            test = hdf5_file_config_1.create_group(f"test")
            test.attrs['Path'] = test_data_file_config_1

        # Configuration 2
        hdf5_file_path_config_2 = f"./data/SocioPatterns/socio_patterns_config_2_{i}.hdf5"
        with h5py.File(hdf5_file_path_config_2, "w") as hdf5_file_config_2:
            # Creating group for each data split
            train = hdf5_file_config_2.create_group(f"train")
            train.attrs['Path'] = train_data_file_config_2
            val = hdf5_file_config_2.create_group(f"val")
            val.attrs['Path'] = val_data_file_config_2
            test = hdf5_file_config_2.create_group(f"test")
            test.attrs['Path'] = test_data_file_config_2

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