import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

class GraphSEIR():
    """
    Implementation of the Susceptible-Exposed-Infectious-Recovered model on time-series of graphs.
    Given a graph (V, E, W) where V is the set of nodes, E the set of edges and W the set of weights compute the evolution of the SEIR model on the graph for each node i :

        Pr(E -> I) = SIGMA * dt
        Pr(I -> R) = GAMMA * dt
        Pr(S -> E) = BETA * sum_{j in N(i)} W_{ij} * dt
    """
    
    SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
    MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}


    def __init__(self, incubation_period, recovery_period, infection_rate):
        """
        Initialize the SEIR model parameters :
            - incubation_period : average period for an exposed individual to become infectious
            - infectuous_perecovery_periodriod : average period for an infectious individual to recover
            - infection_rate : average frequency of transmission from an infectious individual to a susceptible individual per unit of time
        """
        self.SIGMA = 1/incubation_period
        self.GAMMA = 1/recovery_period
        self.BETA = infection_rate
        self.DEBUG = False


    def __repr__(self):
        txt = f'SEIR model with parameters : \n'
        txt += f'Incubation rate : {1/self.SIGMA}\n'
        txt += f'Recovery rate : {1/self.GAMMA}\n'
        txt += f'Infection rate : {self.BETA}\n'
        return txt
    

    def step(self, state, src, dst, weights, dt, scale=1):    
        """
        Update vector of states of the nodes given a list of interactions between nodes i,j at with weights W_{ij} :

        Parameters :
        ------------
            - state : vector of states of the nodes
            - src : list of source nodes i
            - dst : list of destination nodes j
            - weights : list of weights W_{ij}
            - dt : time step
            - scale : scaling factor for the transmission rate

        Returns :
        ---------
            - state_next : vector of states of the nodes at time
        """
        state_next = state.copy() # encode the initial state

        # transitions susceptible -> exposed
        mask_S = state == self.SUSCEPTIBLE
        mask_I = state == self.INFECTIOUS
        for i, j, w in zip(src, dst, weights):
            if mask_S[i] and mask_I[j]:
                state_next[i] = self.EXPOSED if np.random.rand() < self.BETA * w * dt * scale else self.SUSCEPTIBLE
            if mask_I[i] and mask_S[j]:
                state_next[j] = self.EXPOSED if np.random.rand() < self.BETA * w * dt * scale else self.SUSCEPTIBLE

        # transitions exposed -> infectious
        mask_E = state == self.EXPOSED
        if self.DEBUG : print(mask_E.sum(), self.SIGMA * dt)
        state_next[mask_E] = np.where(np.random.rand(mask_E.sum()) < self.SIGMA * dt, self.INFECTIOUS, self.EXPOSED)

        # transitions infectious -> recovered
        if self.DEBUG : print(mask_I.sum(), self.GAMMA * dt)
        state_next[mask_I] = np.where(np.random.rand(mask_I.sum()) < self.GAMMA * dt, self.RECOVERED, self.INFECTIOUS)

        return state_next
    
    def to_snapshot(self, src, dst, weight, time, features, state):
        """ 
            Convert the data to the format expected by one instance of DynamicGraphTemporalSignal
        """
        nodes_ids = np.arange(len(state))
        edges_indices = np.array([src, dst])
        edges_weights = weight.values
        features = np.array(features)
        # Multi-class version (Threshold classification is not possible)
        targets = np.expand_dims(state, axis=1)
        timestamps = np.zeros(len(features)) + time

        return edges_indices, edges_weights, timestamps, features, targets, nodes_ids

    def _simulate_age(self, n_individuals, mean_age=41.9, std_age=10):
        
        """
            Simulates the age of the different individuals, between 0 and 92
            (0 and 92 are the min and max values of the IPC dataset).

            Parameters:
            -----------
            n_individuals: int
                Number of individuals in the population to simulate.
            mean_age: float
                Mean age of the population.
            std_age: float
                Standard deviation of the age of the population.

            Returns:
            --------
            ages_raw: np.array
                Array of shape (n_individuals,) corresponding to the age (between 0
                and 92) of the different individuals.
            ages: np.array
                Standard/robust scaled ages (to use as features)
        """
        # Raw ages
        # Normal-like distribution centered at MEAN_AGE, std dev STD_AGE, clipped to [0, 92]
        MIN_AGE = 0
        MAX_AGE = 92
        ages_raw = np.clip(np.random.normal(loc=mean_age, scale=std_age, size=n_individuals), MIN_AGE, MAX_AGE)

        
        # Pre-processed ages
        # Robust scaling
        scaler = StandardScaler()
        ages = scaler.fit_transform(ages_raw.reshape(-1, 1) ).squeeze()

        return ages_raw, ages

    def _simulate_origin(self, n_individuals, p_acu=0.57, p_residence=0.35, p_rsop=0.08):
        
        """
            Simulates the origin of the different individuals. Three different
            origins based on the IPC dataset: acute care unit, residence, RSOP.

            Parameters:
            -----------
            n_individuals: int
                Number of individuals in the population to simulate.
            p_acu: float
                "Probability" or proportion of being from the acute care
                unit. Default value based on statistics of the IPC dataset.
            p_residence: float
                "Probability" or proportion of being "residence". Default 
                value based on statistics of the IPC dataset.
            p_rsop: float
                "Probability" or proportion of from RSOP. Default 
                value based on statistics of the IPC dataset.

            Returns:
            --------
            origins_raw: np.array
                Simulated raw string origins.
            origins: np.array
                One-hot encoded origins.
            origins_corr: dict
                Dictionary making the correspondence between the str name
                of the origin, and the idx in the one-hot encoding.

            
        """
        # ORIGIN (acute care unit, Residence, RSOP)
        origins_raw = np.random.choice(
                                        ['Acute Care Unit', 'Residence', 'RSOP'],
                                        size=n_individuals,
                                        p=[p_acu, p_residence, p_rsop]
                                    )

        # One hot encoding
        possible_origins = np.unique(origins_raw)
        oh_enc_origins = LabelBinarizer() # one-hot encoder
        oh_enc_origins.fit_transform(possible_origins)
        origins_corr = {oh_enc_origins.classes_[i].lower():i for i in range(len(oh_enc_origins.classes_))}
        origins = oh_enc_origins.transform(origins_raw)

        return origins_raw, origins, origins_corr

    def _simulate_gender(self, n_individuals, p_male=0.52, p_female=0.48):
        """
            Simulates the gender of the different individuals. Two possible
            values based on the IPC dataset: Male and Female.

            Parameters:
            -----------
            n_individuals: int
                Number of individuals in the population to simulate.
            p_male: float
                "Probability" or proportion of being male. Default 
                value based on statistics of the IPC dataset.
            p_female: float
                "Probability" or proportion of being female. Default 
                value based on statistics of the IPC dataset.

            Returns:
            --------
            genders_raw: np.array
                Simulated raw string genders.
            genders: np.array
                Binary encoded genders (0: male, 1: female).
            genders_corr: dict
                Dictionary making the correspondence between the str name
                of the origin, and the idx in the one-hot encoding.
            
        """
        # GENDER (male (1), female (2))
        genders_corr = {
                        "Male": 0,
                        "Female": 1,
        }
        genders_raw = np.random.choice(["Male", "Female"],
                                size=n_individuals,
                                p=[p_male, p_female])

        # Binary encoding
        genders = genders_raw
        genders[genders == "Male"] = 0
        genders[genders == "Female"] = 1
        

        return genders_raw, genders, genders_corr

    def _simulate_comorbidities(self, n_individuals, p_diabetes=6.2/100, p_stroke=296.3/100000, p_cancer=377/100000, p_obesity=12/100, p_smoke=24/100):
        """
            Simulates different comorbidities for the different individuals.
            Five possible comorbidities (one individual can have several 
            comorbidities, going from none to all): Diabetes, Stroke, Cancer,
            Obesity, Smoke.

            Parameters:
            -----------
            n_individuals: int
                Number of individuals in the population to simulate.
            p_diabetes: float
                "Probability" or proportion of having diabetes. Default 
                value based on statistics of Switzerland.
            p_stroke: float
                "Probability" or proportion of having had a stroke. Default 
                value based on statistics of Switzerland
            p_cancer: float
                "Probability" or proportion of having cancer. Default 
                value based on statistics of Switzerland
            p_obesity: float
                "Probability" or proportion of being obese. Default 
                value based on statistics of Switzerland
            p_smoke: float
                "Probability" or proportion of being a smoker. Default 
                value based on statistics of Switzerland

            Returns:
            --------
            comorbidities: dict
                Dictionary of simulated comorbidities, where the keys are the
                comorbidities names, and the values are binary np.ararys
                of shape (n_individuals, ) indicating if each individual
                has or not the comorbidity (0 if not and 1 if yes).
        """
        # COMORBIDITIES (multi-hot encoding: Diabetes, Stroke, Cancer, Obesity, Smoke)
        comorbidities_corr = {
                                'diabetes': 0,
                                'stroke': 1,
                                'cancer': 2,
                                'obesity': 3,
                                'smoke': 4
                            }
        comorbidities = {
            'diabetes': np.random.binomial(1, p_diabetes, n_individuals),
            'stroke': np.random.binomial(1, p_stroke, n_individuals),
            'cancer': np.random.binomial(1, p_cancer, n_individuals),
            'obesity': np.random.binomial(1, p_obesity, n_individuals),
            'smoke': np.random.binomial(1, p_smoke, n_individuals)
        }


        return comorbidities

    def _simulate_static_features(self, nodes_ids, **kwargs):

        """
           Simulate different features for the different individuals.

           Parameters:
           -----------
           nodes_ids: list
               List of np.arrays of lenght the number of simulated days.
               Each array correspond to the list of nodes at each given
               day.

           Returns:
           --------
           features: np.array
               A numpy array of shape (n_individuals, n_total_features) where
               for each individual we have one feature vector regrouping all 
               the simulated features.
           corr_features: dict
                Dictionary indicating, for each named feature of an individual, its index in the
                np.array.
            raw_features: dict
                Dictionary with the raw features (useful to plot histograms).
        """
        # Number of individuals
        n_individuals = nodes_ids[0].shape[0]

        # Age
        ages_raw, ages = self._simulate_age(n_individuals, kwargs["mean_age"], kwargs["std_age"])

        # Origin
        origins_raw, origins, origins_corr = self._simulate_origin(n_individuals, kwargs["p_acu"], kwargs["p_residence"], kwargs["p_rsop"])

        # Gender
        genders_raw, genders, genders_corr = self._simulate_gender(n_individuals, kwargs["p_male"], kwargs["p_female"])

        # Comorbidities
        comorbidities = self._simulate_comorbidities(n_individuals, kwargs["p_diabetes"],  kwargs["p_stroke"], kwargs["p_cancer"], kwargs["p_obesity"], kwargs["p_smoke"])

        # Combining all the features in one single feature vector
        n_days = len(nodes_ids)
        n_origins = len(np.unique(origins_raw))
        n_comorbidities = len(comorbidities)
        n_features = 2 + n_origins + len(comorbidities) # Age, 3 origins, gender, 5 comorbidities
        corr_features = {
                            "age": 0,
                            "acute care unit": 1,
                            "rsop": 2,
                            "residence": 3,
                            "gender": 4,
                            "diabetes": 5,
                            "stroke": 6,
                            "cancer": 7,
                            "obesity": 8,
                            "smoke": 9
                            
                        }
        features = np.zeros((n_days, n_individuals, n_features))
        for tmp_day in range(n_days):
            # Handling STATIC FEATURES (that do not change over time)
            features[tmp_day, :, corr_features["age"]] = ages
            features[tmp_day, :, corr_features["acute care unit"]] = origins[:, origins_corr["acute care unit"]]
            features[tmp_day, :, corr_features["rsop"]] = origins[:, origins_corr["rsop"]]
            features[tmp_day, :, corr_features["residence"]] = origins[:, origins_corr["residence"]]
            features[tmp_day, :, corr_features["gender"]] = genders
            features[tmp_day, :, corr_features["diabetes"]] = comorbidities["diabetes"]
            features[tmp_day, :, corr_features["stroke"]] = comorbidities["stroke"]
            features[tmp_day, :, corr_features["cancer"]] = comorbidities["cancer"]
            features[tmp_day, :, corr_features["obesity"]] = comorbidities["obesity"]
            features[tmp_day, :, corr_features["smoke"]] = comorbidities["smoke"]

        # Raw features (useful for histograms)
        raw_features = {
                            "age": ages_raw,
                            "origin": origins_raw,
                            "gender": genders_raw,
                            "comorbidities": comorbidities,
                       }

        return features, corr_features, raw_features 
    

    def simulate(self, init_state, interactions, dt, scale, precomputed_features=None):
        """
            Simulate the evolution of the SEIR model on a time-series of graphs.
            If "precomputed_features" is None, it simulates them from scractch, if not it
            takes them as features (useful if want to simulate several times
            without changing the features, for instance if just want to 
            modifiy the global transmission risk)
        """
        # Simulating some "realistic" features
        if (precomputed_features is None):
            tmp_nodes_ids = np.array([[i for i in range(init_state.shape[0])] for _ in range(len(interactions))])
            features, corr_features, raw_features = self._simulate_static_features(
                                                                                    tmp_nodes_ids,
                                                                                    mean_age=41.9,
                                                                                    std_age=10, 
                                                                                    p_acu=0.57,
                                                                                    p_residence=0.35,
                                                                                    p_rsop=0.08,
                                                                                    p_male=0.52,
                                                                                    p_female=0.48,
                                                                                    p_diabetes=6.2/100,
                                                                                    p_stroke=296.3/100000,
                                                                                    p_cancer=377/100000,
                                                                                    p_obesity=12/100,
                                                                                    p_smoke=24/100
                                                                                )
        else:
            features = precomputed_features
            corr_features = None
            raw_features = None

        # initialize the states
        initial_states = np.vectorize(self.MAPPING.get)(init_state.copy()) # encode the initial state
        states0 = np.eye(4)[initial_states]
        d0 = min([d for d, _ in interactions])

        # initialize the data
        nodes_ids = []
        edges_indices = []
        edges_weights = []
        timestamps = []
        #features = []
        targets = []
        stats = []
        # roles = []
        roles = {}
        self._states_history = []


        for i, (d, df) in enumerate(interactions):
            if (d == d0):
                states = self.step(initial_states, df.i, df.j, df.weight, dt=dt, scale=scale)
            else:
                states = self.step(states, df.i, df.j, df.weight, dt=dt, scale=scale)
            stats.append([d, np.sum(states == self.SUSCEPTIBLE), np.sum(states == self.EXPOSED), np.sum(states == self.INFECTIOUS), np.sum(states == self.RECOVERED)])
            self._states_history.append(np.vectorize(self.INV_MAPPING.get)(states.copy()))
            snapshot = self.to_snapshot(src=df.i, dst=df.j, weight=df.weight, time=i * dt, features=states0, state=states)
            edges_indices.append(snapshot[0])
            edges_weights.append(snapshot[1])
            timestamps.append(snapshot[2])
            targets.append(snapshot[4])
            nodes_ids.append(snapshot[5])
            # Getting the roles of the individuals for the current snapshot
            if ('Si' in df.keys() and 'Sj' in df.keys()):
                roles_i = df.groupby('i')['Si'].apply(set).to_dict()
                roles_j = df.groupby('j')['Sj'].apply(set).to_dict()
                for tmp_ind_i in roles_i:
                    tmp_role = roles_i[tmp_ind_i].pop()
                    if (tmp_ind_i in roles):
                        if (roles[tmp_ind_i] != tmp_role):
                            print(f"\nPROBLEM: Individual {tmp_ind_i} has two different roles: {roles[tmp_ind_i]} and {tmp_role}")
                            raise RuntimeError(f"\nPROBLEM: Individual {tmp_ind_i} has two different roles: {curr_roles[tmp_ind_i]} and {tmp_role}")
                    else:
                        roles[tmp_ind_i] = tmp_role
                for tmp_ind_j in roles_j:
                    tmp_role = roles_j[tmp_ind_j].pop()
                    if (tmp_ind_j in roles):
                        if (roles[tmp_ind_j] != tmp_role):
                            print(f"\nPROBLEM: Individual {tmp_ind_j} has two different roles: {roles[tmp_ind_j]} and {tmp_role}")
                            raise RuntimeError(f"\nPROBLEM: Individual {tmp_ind_i} has two different roles: {curr_roles[tmp_ind_i]} and {tmp_role}")
                    else:
                        roles[tmp_ind_j] = tmp_role

        states = np.vectorize(self.INV_MAPPING.get)(states.copy()) # decode the updated state
        stats = pd.DataFrame(stats, columns=['step', 'S', 'E', 'I', 'R'])
        
        return nodes_ids, edges_indices, edges_weights, timestamps, features, targets, states, stats, roles, corr_features, raw_features


class GraphSEIRPerIndividual(GraphSEIR):
    """
    Implementation of the Susceptible-Exposed-Infectious-Recovered model on time-series of graphs
    where the individuals have transmission, recovery and incubation parameters that depends on their
    features.
    Given a graph (V, E, W) where V is the set of nodes, E the set of edges and W the set of weights compute the evolution of the SEIR model on the graph for each node i :
    """
    SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
    MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}


    def __init__(self, base_incubation=5, base_recovery=7, base_infection_factor=0.02):
        """
            SEIR model with infection, incubation and recovery parameters per individual
        """
        # Base incubation period in days
        self.base_incubation = base_incubation
        # Base number of recovery days
        self.base_recovery = base_recovery
        # Base infection factor per contact (bot exactly the same as the infection risk)
        self.base_infection_factor = base_infection_factor

        self.DEBUG = False  


    def __repr__(self):
        txt = "SEIR Model with parameters per individual \n"
        return txt

    def step(self, states, src, dst, weights, current_day, features, scale):    
        """
        Update vector of states of the nodes given a list of interactions between nodes i,j at with weights W_{ij} :

        Parameters :
        ------------
            - states : vector of states of the nodes
            - src : list of source nodes i
            - dst : list of destination nodes j
            - weights : list of weights W_{ij}
            - current_day : Day that is being simulated
            - features : features of the different individuals
            - scale : scaling factor for the transmission rate

        Returns :
        ---------
            - states_next : vector of states of the nodes at time
        """
        # Copy initial states
        states_next = states.copy() # encode the initial states

        # Transition S --> E
        # NOTE: we suppose that, for individual A, each interaction
        # NOTE: may cause infection, but they are independent. This
        # NOTE: means that, if one of the persons that interacted
        # NOTE: with individual A gave him the infection, then he is 
        # NOTE: infected from that moment (so even if other future 
        # NOTE: interactions that day give they the infection, it does
        # NOTE: change anything).
        # NOTE: Because of the incubation rates, an individual that
        # NOTE: got infected in day X cannot spread the virus/bacteria
        # NOTE: that day (unless we select an incubation period < 1).
        for i, j, w in zip(src, dst, weights):
            # Case where i is susceptible and j is infected
            if (states[i] == self.SUSCEPTIBLE and states[j] == self.INFECTIOUS):
                ind_a = features[current_day, j]
                ind_b = features[current_day, i]
                interaction_duration = w
                inf_prob = infection_probability(ind_a, ind_b, interaction_duration, self._corr_features, self.base_infection_factor)
                self.infection_rates[current_day, j, i] = inf_prob
                if (np.random.rand() <  inf_prob*scale):
                    states_next[i] = self.EXPOSED
                    self._n_exposed[current_day, i] = 0
                    if (current_day+1 < len(self._n_exposed)):
                        self._n_susceptible[current_day+1, i] = -1
                        self._n_exposed[current_day+1, i] = self._n_exposed[current_day, i] + 1
                else:
                    if (current_day+1 < len(self._n_susceptible)):
                        self._n_susceptible[current_day+1, i] = self._n_susceptible[current_day, i] + 1
                    
            # Case where j susceptible and i is infected
            if (states[j] == self.SUSCEPTIBLE and states[i] == self.INFECTIOUS):
                ind_a = features[current_day, i]
                ind_b = features[current_day, j]
                interaction_duration = w               
                inf_prob = infection_probability(ind_a, ind_b, interaction_duration, self._corr_features, self.base_infection_factor)
                self.infection_rates[current_day, i, j] = inf_prob
                if (np.random.rand() <  inf_prob*scale):
                    states_next[j] = self.EXPOSED
                    self._n_exposed[current_day, j] = 0
                    if (current_day+1 < len(self._n_exposed)):
                        self._n_susceptible[current_day+1, j] = -1
                        self._n_exposed[current_day+1, j] = self._n_exposed[current_day, j] + 1
                else:
                    if (current_day+1 < len(self._n_susceptible)):
                        self._n_susceptible[current_day+1, j] = self._n_susceptible[current_day, j] + 1
        
        # Transition E --> I
        E_idxs = np.argwhere(states == self.EXPOSED)
        for exposed_id in E_idxs:
            # Transition?
            inc_dur = self.incubation_durations[current_day, exposed_id]
            exposed_duration = self._n_exposed[current_day, exposed_id]
            if (exposed_duration) < 0: # Initializing the number of days that the individual has been exposed if it has not been done yet
                self._n_exposed[current_day, exposed_id] = 0
            if (exposed_duration >= inc_dur):
                # Updating states
                states_next[exposed_id] = self.INFECTIOUS
                # Update counter
                self._n_infectious[current_day, exposed_id] = 0
                if (current_day+1 < len(self._n_exposed)-1):
                    self._n_exposed[current_day+1, exposed_id] = -1
                    self._n_infectious[current_day+1, exposed_id] = self._n_infectious[current_day, exposed_id] + 1
            else:
                if (current_day+1 < len(self._n_exposed)-1):
                    self._n_exposed[current_day+1, exposed_id] = self._n_exposed[current_day, exposed_id] + 1      

        # Transition I --> R
        I_idxs = np.argwhere(states == self.INFECTIOUS)
        for infectious_id in I_idxs:
            # Transition?
            rec_dur = self.recovery_durations[current_day, infectious_id]
            infectious_duration = self._n_infectious[current_day, infectious_id]
            if (infectious_duration) < 0: # Initializing the number of days that the individual has been infectious if it has not been done yet
                self._n_infectious[current_day, infectious_id] = 0
            if (infectious_duration >= rec_dur):
                # Updating states
                states_next[infectious_id] = self.RECOVERED
                # Update counter
                self._n_recovered[current_day, infectious_id] = 0
                if (current_day+1 < len(self._n_recovered)-1):
                    self._n_infectious[current_day+1, infectious_id] = -1
                    self._n_recovered[current_day+1, infectious_id] = self._n_recovered[current_day, infectious_id] + 1
            else:
                if (current_day+1 < len(self._n_infectious)-1):
                    self._n_infectious[current_day+1, infectious_id] = self._n_infectious[current_day, infectious_id] + 1      

        return states_next


    def simulate(self, init_state, interactions, dt, scale, precomputed_features=None):
        """
            Simulate the evolution of the SEIR model on a time-series of graphs.
            If "precomputed_features" is None, it simulates them from scractch, if not it
            takes them as features (useful if want to simulate several times
            without changing the features, for instance if just want to 
            modifiy the global transmission risk)
        """
        # Simulating some "realistic" features
        n_days = len(interactions)
        n_individuals = init_state.shape[0]
        if (precomputed_features is None):
            tmp_nodes_ids = np.array([[i for i in range(n_individuals)] for _ in range(n_days)])
            features, corr_features, raw_features = self._simulate_static_features(
                                                                                        tmp_nodes_ids,
                                                                                        mean_age=41.9,
                                                                                        std_age=10, 
                                                                                        p_acu=0.57,
                                                                                        p_residence=0.35,
                                                                                        p_rsop=0.08,
                                                                                        p_male=0.52,
                                                                                        p_female=0.48,
                                                                                        p_diabetes=6.2/100,
                                                                                        p_stroke=296.3/100000,
                                                                                        p_cancer=377/100000,
                                                                                        p_obesity=12/100,
                                                                                        p_smoke=24/100
                                                                                    )
            self._corr_features = corr_features
            corr_features = self._corr_features 
        else:
            features = precomputed_features
            corr_features = None 
            raw_features = None

        # initialize the states
        initial_states = np.vectorize(self.MAPPING.get)(init_state.copy()) # encode the initial state
        states0 = np.eye(4)[initial_states]
        d0 = min([d for d, _ in interactions])

        # Attribute for the infection rates
        # IT WILL BE FILLED OVER THE STEPS
        self.infection_rates = np.zeros((n_days, n_individuals, n_individuals))
        
        # Getting the incubation period per individuals
        self.incubation_durations = np.empty((n_days, n_individuals))
        for day in range(n_days):
        #for day in tqdm(range(n_days)):
            for ind_id in range(n_individuals):
            #for ind_id in tqdm(range(n_individuals)):
                ind = features[day, ind_id]
                rec_dur = incubation_duration(ind, self._corr_features, self.base_incubation)
                self.incubation_durations[day, ind_id] = rec_dur

        # Getting the recovery rate per individuals
        self.recovery_durations = np.empty((n_days, n_individuals))
        for day in range(n_days):
        #for day in tqdm(range(n_days)):
            for ind_id in range(n_individuals):
            #for ind_id in tqdm(range(n_individuals)):
                ind = features[day, ind_id]
                rec_dur = recovery_duration(ind, self._corr_features, self.base_recovery)
                self.recovery_durations[day, ind_id] = rec_dur

        
        # Initialize variables to follow the states over time (shape n_day x n_total_individuals
        # If an individual is not in the state at the given time, then the value is -1
        self._n_susceptible = -1*np.ones((n_days, n_individuals))
        self._n_susceptible[0, initial_states == self.SUSCEPTIBLE] = 0
        self._n_exposed = -1*np.ones((n_days, n_individuals))
        self._n_exposed[0, initial_states == self.EXPOSED] = 0
        self._n_infectious = -1*np.ones((n_days, n_individuals))
        self._n_infectious[0, initial_states == self.INFECTIOUS] = 0
        self._n_recovered = -1*np.ones((n_days, n_individuals))
        self._n_recovered[0, initial_states == self.RECOVERED] = 0
        
        # initialize the data
        nodes_ids = []
        edges_indices = []
        edges_weights = []
        timestamps = []
        #features = []
        targets = []
        stats = []
        #roles = []
        roles = {}
        self._states_history = []


        for i, (d, df) in enumerate(interactions):
            current_day = i
            if (d == d0):
                states = self.step(initial_states, df.i, df.j, df.weight, current_day=current_day, features=features, scale=scale)
            else:
                states = self.step(states, df.i, df.j, df.weight, current_day=current_day, features=features, scale=scale)
            stats.append([d, np.sum(states == self.SUSCEPTIBLE), np.sum(states == self.EXPOSED), np.sum(states == self.INFECTIOUS), np.sum(states == self.RECOVERED)])
            self._states_history.append(np.vectorize(self.INV_MAPPING.get)(states.copy()))
            snapshot = self.to_snapshot(src=df.i, dst=df.j, weight=df.weight, time=i * dt, features=states0, state=states)
            edges_indices.append(snapshot[0])
            edges_weights.append(snapshot[1])
            timestamps.append(snapshot[2])
            targets.append(snapshot[4])
            nodes_ids.append(snapshot[5])
            # Getting the roles of the individuals for the current snapshot
            if ('Si' in df.keys() and 'Sj' in df.keys()):
                roles_i = df.groupby('i')['Si'].apply(set).to_dict()
                roles_j = df.groupby('j')['Sj'].apply(set).to_dict()
                for tmp_ind_i in roles_i:
                    tmp_role = roles_i[tmp_ind_i].pop()
                    if (tmp_ind_i in roles):
                        if (roles[tmp_ind_i] != tmp_role):
                            print(f"\nPROBLEM: Individual {tmp_ind_i} has two different roles: {roles[tmp_ind_i]} and {tmp_role}")
                    else:
                        roles[tmp_ind_i] = tmp_role
                for tmp_ind_j in roles_j:
                    tmp_role = roles_j[tmp_ind_j].pop()
                    if (tmp_ind_j in roles):
                        if (roles[tmp_ind_j] != tmp_role):
                            print(f"\nPROBLEM: Individual {tmp_ind_j} has two different roles: {roles[tmp_ind_j]} and {tmp_role}")
                    else:
                        roles[tmp_ind_j] = tmp_role

        states = np.vectorize(self.INV_MAPPING.get)(states.copy()) # decode the updated state
        stats = pd.DataFrame(stats, columns=['step', 'S', 'E', 'I', 'R'])
        
        return nodes_ids, edges_indices, edges_weights, timestamps, features, targets, states, stats, roles, corr_features, raw_features



def infection_probability(ind_a, ind_b, interaction_duration, corr_features, base_infection_factor):
    """
        Computes the probability of individual A infecting individual B.
        
        Each individual has the following features:
            - age (int) (scaled)
            - gender ('male' or 'female', 0 or 1)
            - hospital_origin ('Acute Care Unit', 'Residence', 'RSOP')
            - comorbidities ('diabetes', 'stroke', 'cancer', 'obesity', 'smoke')

        Parameters:
        -----------
        ind_a: np.array
            Features of individual a.
        ind_b: np.array
            Features of individual b.
        interaction_duration: int
            Duration of the interaction between the two individuals, in seconds
        corr_features: dict
            Dictionary indicating, for each named feature, its index in the
            np.array
        base_infection_factor: float
            Base infection factor per contact (not exactly the same as the infection risk).

        Returns:
        --------
        infection_prob: bool
            Float between 0 an 1, indicating the probability that individual
            A infected individual B.
    """
    # The following scores can be modified for a more realistic modelisation
    # Age (scaled, between -1 and 1)
    age_avg = (ind_a[corr_features["age"]]+ind_b[corr_features["age"]])/2 + 1 # +1 to bring the factor between 0 and 1
    age_factor = 1 + 0.5*age_avg
    # Gender
    if (ind_b[corr_features["gender"]] == 0):
        gender_factor = 1.1
    else:
        gender_factor = 1.0
    # Origin
    if (ind_b[corr_features["acute care unit"]] == 1):
        origin_factor = 1.5
    else:
        if (ind_b[corr_features["residence"]] == 1):
            origin_factor = 1.2
        else:
            if (ind_b[corr_features["rsop"]] == 1):
                origin_factor = 1.0
    # Comorbidity
    n_comorbidities_b = ind_b[corr_features["diabetes"]] +\
                        ind_b[corr_features["stroke"]] +\
                        ind_b[corr_features["cancer"]] +\
                        ind_b[corr_features["obesity"]] +\
                        ind_b[corr_features["smoke"]]
    comorbidity_factor = 1 + 0.1 * n_comorbidities_b
    smoking_boost = 0.2 if (ind_b[corr_features["smoke"]] == 1) else 0

    # Contact factor
    time_constant_rate_risk_saturation = 1200 # IN SECONDS
    interaction_factor = 1 - np.exp(-interaction_duration / time_constant_rate_risk_saturation)

    # Computing the final risk
    risk = (base_infection_factor * age_factor * gender_factor * origin_factor * comorbidity_factor * interaction_factor) + smoking_boost
    risk = np.clip(risk, a_min=0, a_max=1)
    
    return risk


def recovery_duration(ind, corr_features, base_recovery=7):
    """
        Estimates the recovery time in days for an infected individual.

        Parameters:
        -----------
        ind: np.array
            Features of the individual to consider.
        corr_features: dict
            Dictionary indicating, for each named feature, its index in the
            np.array.
        base_recovery: int
            Base recovery duration

        Returns:
        --------
        duration: int
            Number of days necessary for the individual to recover
    """
    # Age penalty
    age_shifted = (ind[corr_features['age']] + 1) / 2  # range [0, 1]
    age_penalty = 7 * age_shifted  # adds 0–10 days based on age
    # Origin penalty
    if (ind[corr_features["acute care unit"]] == 1):
        origin_penalty = 5
    else:
        if (ind[corr_features["residence"]] == 1):
            origin_penalty = 2
        else:
            if (ind[corr_features["rsop"]] == 1):
                origin_penalty = 1
    # Comorbitiy penalty
    n_comorbidities = ind[corr_features["diabetes"]] +\
                      ind[corr_features["stroke"]] +\
                      ind[corr_features["cancer"]] +\
                      ind[corr_features["obesity"]] +\
                      ind[corr_features["smoke"]]
    comorbidity_penalty = 2 * n_comorbidities 
    
    # Computing the duration
    duration = int(base_recovery + age_penalty + origin_penalty + comorbidity_penalty)
    max_recovery_duration = 30
    min_recovery_duration = 5
    duration = np.clip(duration, a_min=min_recovery_duration, a_max=max_recovery_duration)
        
    return duration


def incubation_duration(ind, corr_features, base_incubation=5):
    """
        Estimates the incubation period (time between exposure and symptom onset).
        
        Parameters:
        -----------
        ind: np.array
            Features of the individual to consider.
        corr_features: dict
            Dictionary indicating, for each named feature, its index in the
            np.array.
        base_incubation: int
            Base incubation duration.

        Returns:
        --------
        incubation: int
            Incubation time (in day) corresponding to the time between exposure
            and symtoms onset.
    """
    
    # Age penalty
    age_shifted = (ind[corr_features["age"]] + 1) / 2  # [0, 1]
    age_penalty = 2 * age_shifted  # up to +2 days for older people
    # Origin penalty
    # IMPORTANT: We make the assumption that the origin of the person does not affect the incubation time
    origin_penalty = 0
    # Comorbidities penalty
    n_comorbidities = ind[corr_features["diabetes"]] +\
                      ind[corr_features["stroke"]] +\
                      ind[corr_features["cancer"]] +\
                      ind[corr_features["obesity"]] +\
                      ind[corr_features["smoke"]]
    comorbidity_penalty = 0.2 * n_comorbidities
    smoker_penalty = 1 if (ind[corr_features["smoke"]] == 1) else 0

    # Final incubation time
    incubation = int(base_incubation + age_penalty + origin_penalty + comorbidity_penalty + smoker_penalty)
    min_incubation = 2
    max_incubation = 100000
    incubation = np.clip(incubation, a_min=min_incubation, a_max=max_incubation)
    
    return incubation
