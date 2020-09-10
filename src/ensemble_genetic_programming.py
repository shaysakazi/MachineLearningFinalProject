import random
import re

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class EnsembleGeneticProgramming:
    def __init__(self, num_trees=100, num_forest=10, weighted=False, max_generations=10, forest_selection=5, k=5):
        self.num_trees = num_trees
        self.num_forest = num_forest
        self.weighted = weighted
        self.max_generations = max_generations
        self.forest_selection = forest_selection
        self.k = k
        self.voting_forests = None

    def _egp(self, X, y):
        forest_list = self._generate_forests(X)  # list of (str, estimator) tuples
        g = 0
        while g < self.max_generations or len(forest_list) == 1:
            forest_parents = self._select_forests(forest_list, X, y)  # choose N forests
            forest_off_springs = self._breeding(forest_parents)  # create N off springs
            forest_list = self._prune(forest_off_springs, X, y)  # prune 2N (N from forest selection and N from off
            # springs) into N forests
            g += 1

        return VotingRegressor(forest_list)

    def _generate_forests(self, X):
        forest_list = []
        for forest_index in range(self.num_forest):
            new_forest = self._generate_forest(X)
            generation_index = 0
            while new_forest in forest_list and generation_index < 100:
                generation_index += 1
                new_forest = self._generate_forest(X)
            forest_list.append(new_forest)
        return forest_list

    def _generate_forest(self, X):
        dataset_features = X.columns
        forest_features = {'max_depth': range(2, 10),
                           'min_samples_split': range(2, 10),
                           'min_weight_fraction_leaf': np.linspace(0.05, 0.5, 10),
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'random_state': range(2, 100)}

        # choose randomly random forest features
        max_depth = random.choice(forest_features['max_depth'])
        min_samples_split = random.choice(forest_features['min_samples_split'])
        min_weight_fraction_leaf = random.choice(forest_features['min_weight_fraction_leaf'])
        max_features = random.choice(forest_features['max_features'])
        random_state = random.choice(forest_features['random_state'])

        # choose randomly dataset features
        num_of_dataset_features = random.randrange(len(dataset_features)) + 1
        used_features = []
        for _ in range(num_of_dataset_features):
            choose_feature = random.choice(dataset_features)
            if len(used_features) < len(dataset_features):
                while choose_feature in used_features:
                    choose_feature = random.choice(dataset_features)
                used_features.append(choose_feature)

        rf_str = 'START, ' + ', '.join(used_features) + ', END, ' + \
                 f'{max_depth}, {min_samples_split}, {min_weight_fraction_leaf}, {max_features}, {random_state}'
        rf_reg = RandomForestRegressor(n_estimators=self.num_trees, max_depth=max_depth, random_state=random_state,
                                       min_samples_split=min_samples_split, max_features=max_features,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, n_jobs=-1)

        return rf_str, rf_reg

    def _select_forests(self, forest_list, X, y):
        forest_parents = []
        for i in range(self.num_forest):
            selected_forest = self._tournament_selection(forest_list, X, y)
            forest_parents.append(selected_forest)

        return forest_parents

    def _tournament_selection(self, forest_list, X, y):
        best = None
        best_mse = 0
        for i in range(self.k):
            ind = random.choice(forest_list)
            ind_mse = self._eval_forest(ind, X, y)
            if best is None or ind_mse > best_mse:
                best = ind
                best_mse = ind_mse
        return best

    def _breeding(self, forest_parents):
        for i in range(self.num_forest):
            off_spring = self._breed(forest_parents)
            forest_parents.append(off_spring)
        return forest_parents

    def _breed(self, forest_parents):
        first_parent = random.choice(forest_parents)
        second_parent = None
        return []

    def _prune(self, forest_off_springs, X, y):
        forest_list = []
        # choose randomly 5% of the forests
        # the rest take the best forests

        return self.num_trees

    def _eval_forest(self, ind, X, y):
        dataset_features = re.search('START, (.*), END', ind[0]).group(1).split(', ')
        forest = ind[1]

        #  ***CHECK***
        X_train, X_test, y_train, y_test = train_test_split(X[dataset_features], y, test_size=0.2, shuffle=True)
        #  ***CHECK***
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        return mean_squared_error(y_test, y_pred)

    def fit(self, X, y):
        self.voting_forests = self._egp(X, y)

    def predict(self, X):
        if self.voting_forests is None:
            raise NotFittedError("This EnsembleGeneticProgramming instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")
        else:
            pass

    def score(self, X, y):
        pass

