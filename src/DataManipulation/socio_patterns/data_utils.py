import os
import sys
from math import ceil
import numpy as np
import pandas as pd
import networkx as nx
import datetime
from tqdm import tqdm
import pickle
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from src.DataManipulation.temporal_graphs import TemporalGraphDataset, TemporalGraph, DynamicGraphTemporalSignal


def reindex(data):
    """ Reindex the indices of the nodes to be consecutive integers
    """
    mapping = {node: i for i, node in enumerate(np.unique(data[['i', 'j']].values.ravel()))}
    data.i = data.i.map(mapping)
    data.j = data.j.map(mapping)
    return data, mapping

# VERSION WORKING FOR SFHH DATASET 
def extension_SFHH(day1, day2, strategy='REP', strategy_kwargs={}, verbose=False):
    """ Extend the graph of day1 to day2 following one of the strategies :
        - REP : simply return the two days
        - RAND-SH : shuffle the participant identities randomly 
        - CONSTR-SH : constrained shuffling of the participant identities such that the average (participant met at day 1 AND 2)/(participant met) over all the participant is the same as the empirical one
    """

    # strategies
    def REP(day1, day2):
        """ Simply return the two days
        """
        return day1, day2

    def RAND_SH(day1, day2):
        """ Shuffle the participant identities randomly
        """
        day1.i = np.random.permutation(day1.i)
        day1.j = np.random.permutation(day1.j)
        day2.i = np.random.permutation(day2.i)
        day2.j = np.random.permutation(day2.j)
        return day1, day2

    def CONSTR_SH(day1, day2, b, n_shuffle, verbose=False,):
        """ Constrained shuffling of participant identities :
            1. compute f_emp, the average of (participant met by i at day 1 AND 2)/(participant met by i on day 1) over all participant i
            2. Choose two identies at random
            3. Swap the identities in the two days
            4. compute the new average f_new and (f_new - f_emp)²
            5. accept the exchange with probability b*(f_new - f_emp)²
            6. repeat step 2 to 5 n_shuffle times 
        """

        def overlap(day1, day2, ids):
            # interactions at day 1 and 2 as dictionaries of sets
            interactions1 = day1.groupby('i')['j'].apply(set).to_dict()
            interactions2 = day2.groupby('i')['j'].apply(set).to_dict()

            f = np.mean([
                len(interactions1.get(i, set()).intersection(interactions2.get(i, set()))) / max(1, len(interactions1.get(i, set())))
                for i in ids
            ])

            return f
        
        def contrained_shuffling(day_shuffled, day_ref, ids):
            # choose two identities at random
            i1, i2 = np.random.choice(ids, 2, replace=False)

            # swap the identities
            mask1_i = day_shuffled.i == i1
            mask2_i = day_shuffled.i == i2
            mask1_j = day_shuffled.j == i1
            mask2_j = day_shuffled.j == i2

            day_shuffled.loc[mask1_i, 'i'], day_shuffled.loc[mask2_i, 'i'] = i2, i1
            day_shuffled.loc[mask1_j, 'j'], day_shuffled.loc[mask2_j, 'j'] = i2, i1

            f_new = overlap(day_ref, day_shuffled, ids)
            prob = np.exp(-b*(f_new - f_emp)**2)

            if verbose :
                print(f'{i1} <-> {i2} : {f_emp} -> {f_new} : {(f_new - f_emp)**2} : {prob}')

            if np.random.rand() <= prob:
                # accept changes
                accepted = True
            else :
                # revert changes if not accepted
                day_shuffled.loc[mask1_i, 'i'], day_shuffled.loc[mask2_i, 'i'] = i1, i2
                day_shuffled.loc[mask1_j, 'j'], day_shuffled.loc[mask2_j, 'j'] = i1, i2
                accepted = False

            return day_shuffled, accepted, f_new

        # compute the empirical average
        ids = pd.concat([day1[['i', 'j']], day2[['i', 'j']]]).melt().value.unique()
        f_emp = overlap(day1, day2, ids)

        # constrained shuffling on each day
        n_accepted = 0
        for _ in range(n_shuffle):
            
            day1, accepted1, _     = contrained_shuffling(day1, day2, ids)
            day2, accepted2, f_new = contrained_shuffling(day2, day1, ids)

            n_accepted += accepted1 + accepted2

        if verbose :
            print(f'Acceptance rate : {n_accepted/(2*n_shuffle)} ({n_accepted}/{(2*n_shuffle)})')
            print(f'Variation of f : {f_new - f_emp}')

        return day1, day2

    # extend the days
    delta_t =  datetime.timedelta(days=1)
    day1_extended = day1.copy()
    day1_extended.t = day1.t + delta_t
    day2_extended = day2.copy()
    day2_extended.t = day2.t + delta_t

    if strategy == 'REP':
        return REP(day1_extended, day2_extended)
    elif strategy == 'RAND-SH':
        return RAND_SH(day1_extended, day2_extended)
    elif strategy == 'CONSTR-SH':
        return CONSTR_SH(day1_extended, day2_extended, verbose=verbose, **strategy_kwargs)
    else :
        raise NotImplementedError(f'Strategy {strategy} not implemented')

# New version designed for SFHH AND Hospital Ward datasets
def extension_more_two_days(days, n_days_use, strategy='REP', strategy_kwargs={}, verbose=False):
    """ Extend the graph of the first day to the last day (2 for SFHH and 5 for Hospital Ward) following one of the strategies:
        - REP : simply return the days that we have
        - RAND-SH : shuffle the participant identities randomly 
        - CONSTR-SH : constrained shuffling of the participant identities such that the average (participant met at day 1 AND day 2 AND ... AND day N)/(participant met) over all the participant is the same as the empirical one
    """
    # strategies
    def CONSTR_SH(days, b_in, b_out, last_day_previous_block, n_shuffle, verbose=False,):
        """ Constrained shuffling of participant identities :
            1. Start from the empirical N-day contact dataset 
            2. Fix the target repeated contact fraction f_emp, computed from real data, as before.
            3. During reshuffling of block m:
                i. Randomly choose two tags IDs, i and j.
                ii. Swap identities of tags i and j across the entire dataset. For every contact involving i, replace i with j, and vice versa.
                iii. For each swap (e.g., of tags i and j):
                    a. Reshuffle all N days of the block (step 3.ii).
                    b. Aggregate the data into daily contact sets (i.e. group all contacts events by day and compute for each day and individual the individuals they had contact with on that day).
                    c. Compute:
                        - f_block: average fraction of repeated contacts within the block (i.e., day d vs d+1, for d=1 to N-1).
                        - f_boundary: average fraction of repeated contacts between the last day of block m-1 and the first day of block mm.
                ii. Combine both into a global cost function: cost=b_in*(f_block-f_emp)^2+b_out*(f_boundary-f_emp)^2
                iii. Accept or reject the tag swap with probability: p = exp(-cost)

        """
        def set_contacts_day_d(day_ID, days):
            """
                Creates a dict where the keys are the individuals IDs and
                the values are the individuals that where in contact
                with the particular individual at that day. For example:
                {
                    0: {5, 7, 10, 13, 14, 15, 16, 18, 19, 21, 22, 23, 35},
                    2: {4, 5, 7, 9, 16, 18, 22, 30, 36},
                    4: {6, 9, 12, 16, 22, 30, 36},
                    5: {7, 9, 10, 13, 18, 21, 22, 36}
                }
            """
            return days[day_ID].groupby('i')['j'].apply(set).to_dict()

        def compute_f_rep_contacts(days):
            """
                Computes the average fraction of repeated contacts across the days,
                considering consecutive day pairs
            """
            frac_rep_contacts_over_days = []
            for day_ID in range(len(days)-1):
                contacts_in_curr_day = set_contacts_day_d(day_ID, days)
                contacts_in_next_day = set_contacts_day_d(day_ID+1, days)
                ids = set(list(contacts_in_curr_day.keys()) + list(contacts_in_next_day.keys()))
                frac_rep_contacts = 0
                for i in ids:
                    # Current day
                    if (i not in contacts_in_curr_day):
                        v_i_d = None
                    else:
                        v_i_d = contacts_in_curr_day[i]
                    # Next day
                    if (i not in contacts_in_next_day):
                        v_i_d_next = None
                    else:
                        v_i_d_next = contacts_in_next_day[i]
                    # Sum
                    if (v_i_d is None):
                        pass # Nothing to do as we cannot divide by zero, so we ignore this individual 
                    else:
                        if (v_i_d_next is None):
                            pass # Nothing to do as the intersection between v_i_d and v_i_d_next will be empty so its length would be 0
                        else:
                            frac_rep_contacts += (len(v_i_d.intersection(v_i_d_next)))/len(v_i_d)
                frac_rep_contacts = frac_rep_contacts/len(ids)
                frac_rep_contacts_over_days.append(frac_rep_contacts)

            return np.mean(frac_rep_contacts_over_days)
                    

        #==================================================#
        # Fix the target repeated contact fraction f_emp computed from the real data
        f_emp = compute_f_rep_contacts(days)

        # Getting all the ids
        ids = pd.concat( [day[['i', 'j']] for day in days ] ).melt().value.unique()
        
        # Getting the roles of all individuals
        roles = {}
        for day_ID in range(len(days)):
            # Roles of the individuals i on that day
            roles_indiv_days_i = days[day_ID].groupby('i')['Si'].apply(set).to_dict()
            for indiv_ID in roles_indiv_days_i:
                if (indiv_ID) not in roles:
                    roles[indiv_ID] = roles_indiv_days_i[indiv_ID].pop()
            # Roles of the individuals i on that day
            roles_indiv_days_j = days[day_ID].groupby('j')['Sj'].apply(set).to_dict()
            for indiv_ID in roles_indiv_days_j:
                if (indiv_ID) not in roles:
                    roles[indiv_ID] = roles_indiv_days_j[indiv_ID].pop()


        # Computing M=n_shuffle new blocks of M days
        n_accepted = 0
        days_reshuffled = deepcopy(days)
        for _ in range(n_shuffle):
            # Swap identities of two randomly chosen tags HAVING THE SAME ROLE (patients
            # swapped between patients, nurses between nurses, etc.)
            i1, i2 = np.random.choice(ids, 2, replace=False)
            i1, i2 = int(i1), int(i2)
            tmp_c = 0
            while (roles[i1] != roles[i2]) or (i1 == i2):
                i2 = int(np.random.choice(ids, 1, replace=False))
                tmp_c += 1
                #print(f"\n\n While iter {tmp_c}\n\n")
            for day_ID in range(len(days)):
                mask1_i = days_reshuffled[day_ID].i == i1
                mask2_i = days_reshuffled[day_ID].i == i2
                days_reshuffled[day_ID].loc[mask1_i, 'i'], days_reshuffled[day_ID].loc[mask2_i, 'i'] = i2, i1
                days_reshuffled[day_ID].loc[mask1_i, 'Si'], days_reshuffled[day_ID].loc[mask2_i, 'Si'] = roles[i2], roles[i1]
                mask1_j = days_reshuffled[day_ID].j == i1
                mask2_j = days_reshuffled[day_ID].j == i2
                days_reshuffled[day_ID].loc[mask1_j, 'j'], days_reshuffled[day_ID].loc[mask2_j, 'j'] = i2, i1
                days_reshuffled[day_ID].loc[mask1_j, 'Sj'], days_reshuffled[day_ID].loc[mask2_j, 'Sj'] = roles[i2], roles[i1]
            
            # Compute f_block: average fraction of repeated contacts within the new created block of contact days (i.e., day d vs d+1, for d=1 to N−1).
            f_block = compute_f_rep_contacts(days_reshuffled)

            # Compute f_boundary: average fraction of repeated contacts between the last day of block m−1 and the first day of block m.
            days_boundary_blocks = [last_day_previous_block, days_reshuffled[0]]
            f_boundary = compute_f_rep_contacts(days_boundary_blocks)

            # Comput global cost
            cost = b_in*(f_block-f_emp)**2 + b_out*(f_boundary-f_emp)**2
            prob = np.exp(-cost)

            # Accept or reject the tag swap
            if verbose :
                print(f'{i1} <-> {i2} : {prob}')

            if np.random.rand() <= prob:
                # accept changes
                accepted = True
            else :
                accepted = False

            # Update number of accepted blocks
            if (accepted):
                n_accepted += 1

        return days_reshuffled


    # Adding n_days_use
    n_blocks_days = ceil(n_days_use/len(days))
    new_days = [deepcopy(day) for day in days]
    last_day_previous_block = new_days[-1]
    delta_t =  datetime.timedelta(days=len(days))
    for m in tqdm(range(n_blocks_days)):
        # Getting the new block
        new_block_days = []
        for day in days: # Here we use the ORIGINAL days to create the block
            new_day = deepcopy(day)
            new_day.t = new_day.t + delta_t*(m+1)
            new_block_days.append(new_day)
        # Using the desired strategy
        if (strategy == 'CONSTR-SH'):
            new_block_days = CONSTR_SH(
                                            days=new_block_days,
                                            last_day_previous_block=last_day_previous_block,
                                            verbose=verbose,
                                            **strategy_kwargs
                                      )
            new_days.extend(new_block_days)
            last_day_previous_block = new_days[-1]
        else :
            raise NotImplementedError(f'Strategy {strategy} not implemented')

    return new_days[:n_days_use]

    
def extension(days, strategy='REP', strategy_kwargs={}, n_days_use=None, dataset_type='SFHH', verbose=False):
    """ Extend the graph of the first day to the last day (2 for SFHH and 5 for Hospital Ward) following one of the strategies:
        - REP : simply return the days that we have
        - RAND-SH : shuffle the participant identities randomly 
        - CONSTR-SH : constrained shuffling of the participant identities such that the average (participant met at day 1 AND day 2 AND ... AND day N)/(participant met) over all the participant is the same as the empirical one
    """
    if (dataset_type.lower() == 'sfhh'):
        return extension_SFHH(days[-2], days[-1], strategy=strategy, strategy_kwargs=strategy_kwargs, verbose=verbose)
    else:
        return extension_more_two_days(days, n_days_use=n_days_use, strategy=strategy, strategy_kwargs=strategy_kwargs, verbose=verbose)
    

def collapsed(data, setting='HET'):
    """ Create collapsed graphs for each day in the dataset :
        - HET : add 20 seconds to the edge weight if it already exists, proxy for the total time spent together
        - HOM : constant edge weight of 1
    """
    collapsed = data.copy()
    collapsed['day'] = collapsed.t.dt.date
    if ('Si' in collapsed.keys() and 'Sj' in collapsed):
        collapsed = collapsed.groupby(['day', 'i', 'j', 'Si', 'Sj']).count().reset_index().rename(columns={'t': 'weight'})
    else:
        collapsed = collapsed.groupby(['day', 'i', 'j']).count().reset_index().rename(columns={'t': 'weight'})
    if setting == 'HET':
        collapsed['weight'] = collapsed['weight'] * 20 # each edge is the total time spent together
    elif setting == 'HOM':
        collapsed['weight'] = 1 # all edges are the same
    else :
        raise NotImplementedError(f'{setting} is not implemented')

    return collapsed


def to_graphs(data, time_column='day'):
    """ Convert the data to a list of daily graphs of interactions
    """
    data.sort_index(inplace=True)
    graphs = []
    for day, df in data.groupby(time_column):
        g = nx.Graph()
        g.add_nodes_from(df.i.unique())
        g.add_nodes_from(df.j.unique())
        g.add_edges_from([(row.i, row.j, {'weight': row.weight}) for row in df.itertuples()])
        graphs.append(g)
    return graphs


class SFHHDataset(TemporalGraphDataset):
    """
    Collection of temporal graphs as a collection of :
        - Nodes events : dataframe (node_events_id, trial_id, timestamp, node_id) 
        - Nodes features : tensor (node_events_id, node_features) 
        - Nodes targets : tensor (node_events_id, node_targets)
        - Edges events : dataframe (edge_events_id, trial_id, timestamp, src_id, dst_id)
        - Edges features : tensor (edge_events_id, d_edge_features)
        - Edges targets : tensor (edge_events_id, edge_targets)
    """
    def __init__(self, datapath : str, *args, **kwargs):
        self._download(datapath, *args, **kwargs)


    def _download(self, datapath : str, *args, **kwargs) -> None:
        self.nodes_events = []
        self.nodes_features = []
        self.nodes_targets = []

        self.edges_events = []
        self.edges_features = []
        self.edges_targets = np.empty((0,0))

        self.states_histories = []
        self.infection_rates = []
        self.incubation_durations = []
        self.recovery_durations = []

        for i, f in enumerate(sorted([f for f in os.listdir(datapath) if f.endswith('.pkl')])):
            with open(os.path.join(datapath, f), 'rb') as h:
                for nodes_ids,\
                    edges_indices,\
                    edges_weights,\
                    timestamps,\
                    features,\
                    targets,\
                    states_history,\
                    infection_rates,\
                    incubation_durations,\
                    recovery_durations in zip(*pickle.load(h)):
                        self.nodes_events.append(np.concatenate([np.ones((nodes_ids.shape[0], 1)) * i, timestamps.reshape(-1, 1), nodes_ids.reshape(-1, 1)], axis=1))
                        self.nodes_features.append(features)
                        self.nodes_targets.append(targets)
                        self.edges_events.append(np.concatenate([np.ones((edges_indices.shape[1], 1)) * i, np.zeros((edges_indices.shape[1], 1))+timestamps[0], edges_indices.T], axis=1))
                        self.edges_features.append(edges_weights)
                        self.states_histories.append(states_history)
                        self.infection_rates.append(infection_rates)
                        self.incubation_durations.append(incubation_durations)
                        self.recovery_durations.append(recovery_durations)
    
        self.nodes_events = pd.DataFrame(np.concatenate(self.nodes_events, axis=0), columns=['trial_id', 'timestamp', 'node_id'])
        self.nodes_features = np.concatenate(self.nodes_features, axis=0)
        self.nodes_targets = np.concatenate(self.nodes_targets, axis=0)

        self.states_histories = np.concatenate(self.states_histories, axis=0)
        self.infection_rates = np.concatenate(self.infection_rates, axis=0)
        self.incubation_durations = np.concatenate(self.incubation_durations, axis=0)
        self.recovery_durations = np.concatenate(self.recovery_durations, axis=0)
        
        self.edges_events = pd.DataFrame(np.concatenate(self.edges_events, axis=0), columns=['trial_id', 'timestamp', 'src_id', 'dst_id'])
        self.edges_features = np.concatenate(self.edges_features, axis=0)

        
    def _snapshots(self, trial_idx : int = 0, time_idxs : np.array = None, verbose : bool = False) -> TemporalGraph:
        mask_nodes = self.nodes_events['trial_id'] == trial_idx
        mask_egdes = self.edges_events['trial_id'] == trial_idx

        nodes_events = self.nodes_events[mask_nodes].groupby('timestamp')
        edges_events = self.edges_events[mask_egdes].groupby('timestamp')

        timestamps = self.timestamps(trial_idx)

        if time_idxs is not None: 
            timestamps = timestamps[time_idxs]

        edges_indices = []
        edges_weights = []
        features = []
        targets = []
        times = []
        ids = []

        states_histories = []
        infection_rates = []
        incubation_durations = []
        recovery_durations = []

        bar = tqdm(timestamps, desc=f'trial {trial_idx} to temporal graph', leave=False) if verbose else timestamps
        for t in bar:
            nodes_events_ids = nodes_events.get_group(t).index
            edges_events_ids = edges_events.get_group(t).index
            edges_indices.append(self.edges_events.loc[edges_events_ids][['src_id', 'dst_id']].values.T)
            edges_weights.append(self.edges_features[edges_events_ids])
            features.append(self.nodes_features[nodes_events_ids])
            targets.append(self.nodes_targets[nodes_events_ids])
            times.append(np.ones((len(nodes_events_ids))) * t)
            ids.append(self.nodes_events.loc[nodes_events_ids].node_id.values)
            states_histories.append(self.states_histories[nodes_events_ids])
            infection_rates.append(self.infection_rates[nodes_events_ids])
            incubation_durations.append(self.incubation_durations[nodes_events_ids])
            recovery_durations.append(self.recovery_durations[nodes_events_ids])

        
        # Temporal graph
        temporalGraph = DynamicGraphTemporalSignal(
            edge_indices=edges_indices,
            edge_weights=edges_weights,
            features=features,
            targets=targets,
            timestamps=times,
            ids=ids
        )

        # SEIR params
        SEIR_params = {
                            "infection_rates": infection_rates,
                            "incubation_durations": incubation_durations,
                            "recovery_durations": recovery_durations 
                    }

        return {"TemporalGraph": temporalGraph, "SEIRParams": SEIR_params}
