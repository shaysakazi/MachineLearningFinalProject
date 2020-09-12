import random
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
import copy


class EnsembleGeneticProgramming:
    def __init__(self, num_trees=100, num_forest=100, max_generations=10, k=3):
        self.num_trees = num_trees
        self.num_forest = num_forest
        self.max_generations = max_generations
        self.k = k
        self.voting_forests = None

    def _egp(self, X, y):
        forest_list = self._generate_forests(X)  # list of (str, estimator) tuples
        g = 0
        while g < self.max_generations:
            # print(f"Generation index: {g}, number of forests: {len(forest_list)},"
            #       f" max_generations: {self.max_generations}")
            time = datetime.now()
            forest_parents = self._select_forests(forest_list, X, y)  # choose N forests
            time = datetime.now() - time
            # print(f"Finish selection on Generation index: {g} after {time.seconds} seconds")
            time = datetime.now()
            forest_offsprings = self._breeding(forest_parents)  # create N off springs
            time = datetime.now() - time
            # print(f"Finish breeding on Generation index: {g} after {time.seconds} seconds")
            time = datetime.now()
            forest_list = self._prune(forest_offsprings, X, y)  # prune 2N (N from forest selection and N from off
            # springs) into N forests
            time = datetime.now() - time
            # print(f"Finish pruning on Generation index: {g} after {time.seconds} seconds")
            g += 1

        return forest_list

    def _best_model(self, forest_list, X, y):
        forests_scores = {forest_offspring: self._eval_forest(forest_offspring, X, y)
                                   for forest_offspring in forest_list}

        count_dict = Counter(forests_scores)
        best_forest = count_dict.most_common(1)

        dataset_features = best_forest[0][7:best_forest[0].find('END')-2].split(', ')
        return best_forest[0], best_forest[1].fit(X[dataset_features], y)

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
        used_features = self._choose_features(dataset_features)

        rf_str = 'START, ' + ', '.join(used_features) + ', END, ' + \
                 f'{max_depth}, {min_samples_split}, {min_weight_fraction_leaf}, {max_features}, {random_state}'
        rf_reg = RandomForestRegressor(n_estimators=self.num_trees, max_depth=max_depth, random_state=random_state,
                                       min_samples_split=min_samples_split, max_features=max_features,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf)

        return rf_str, rf_reg

    def _choose_features(self, features):
        num_of_features = random.randrange(len(features)) + 1
        used_features = []
        for _ in range(num_of_features):
            choose_feature = random.choice(features)
            if len(used_features) < len(features):
                while choose_feature in used_features:
                    choose_feature = random.choice(features)
                used_features.append(choose_feature)
        return used_features

    def _select_forests(self, forest_list, X, y):
        forest_parents = []
        for _ in range(self.num_forest):
            # time = datetime.now()
            selected_forest = self._tournament_selection(forest_list, X, y)
            # time = datetime.now() - time
            # print(f"_tournament_selection after {time.seconds} seconds")
            forest_parents.append(selected_forest)

        return forest_parents

    def _tournament_selection(self, forest_list, X, y):
        best = None
        best_mse = 0
        for i in range(self.k):
            ind = random.choice(forest_list)
            # time = datetime.now()
            ind_mse = self._eval_forest(ind, X, y)
            # time = datetime.now() - time
            # print(f"_eval_forest after {time.seconds} seconds")
            if best is None or ind_mse > best_mse:
                best = ind
                best_mse = ind_mse
        return best

    def _breeding(self, forest_parents):
        offsprings = copy.deepcopy(forest_parents)
        for i in range(self.num_forest):
            offspring = self._breed(forest_parents)
            offsprings.append(offspring)
        return offsprings

    def _breed(self, forest_parents):
        parent_1 = random.choice(forest_parents)
        parent_2 = random.choice(forest_parents)

        offspring = self._uniform_crossover(parent_1, parent_2)
        mutation_prob = random.random()
        if mutation_prob < 0.5:
            offspring = self._mutate(offspring)
        return offspring

    def _uniform_crossover(self, parent_1, parent_2):
        offspring_forest_features = {'max_depth': None, 'min_samples_split': None, 'min_weight_fraction_leaf': None,
                                     'max_features': None, 'random_state': None}
        parent_1_forest_features = {'max_depth': parent_1[1].max_depth,
                                    'min_samples_split': parent_1[1].min_samples_split,
                                    'min_weight_fraction_leaf': parent_1[1].min_weight_fraction_leaf,
                                    'max_features': parent_1[1].max_features, 'random_state': parent_1[1].random_state}
        parent_2_forest_features = {'max_depth': parent_2[1].max_depth,
                                    'min_samples_split': parent_2[1].min_samples_split,
                                    'min_weight_fraction_leaf': parent_2[1].min_weight_fraction_leaf,
                                    'max_features': parent_2[1].max_features, 'random_state': parent_2[1].random_state}

        for key in offspring_forest_features:
            p = random.random()
            if p < 0.5:
                offspring_forest_features[key] = parent_1_forest_features[key]
            else:
                offspring_forest_features[key] = parent_2_forest_features[key]

        selected_parent_1_dataset_features = parent_1[0][7:parent_1[0].find('END')-2].split(', ')
        selected_parent_2_dataset_features = parent_2[0][7:parent_2[0].find('END')-2].split(', ')

        offspring_dataset_features = list(set(selected_parent_1_dataset_features + selected_parent_2_dataset_features))

        rf_str = 'START, ' + ', '.join(offspring_dataset_features) + ', END, ' + \
                 f"{offspring_forest_features['max_depth']}, {offspring_forest_features['min_samples_split']}," \
                 f" {offspring_forest_features['min_weight_fraction_leaf']}," \
                 f" {offspring_forest_features['max_features']}, {offspring_forest_features['random_state']}"

        rf_reg = RandomForestRegressor(n_estimators=self.num_trees, max_depth=offspring_forest_features['max_depth'],
                                       random_state=offspring_forest_features['random_state'],
                                       min_samples_split=offspring_forest_features['min_samples_split'],
                                       max_features=offspring_forest_features['max_features'],
                                       min_weight_fraction_leaf=offspring_forest_features['min_weight_fraction_leaf'])

        return rf_str, rf_reg

    def _mutate(self, offspring):
        mutate_feature = random.choice(['max_depth', 'min_samples_split', 'min_weight_fraction_leaf',
                                        'max_features', 'random_state'])
        forest_features = {'max_depth': range(2, 10),
                           'min_samples_split': range(2, 10),
                           'min_weight_fraction_leaf': np.linspace(0.05, 0.5, 10),
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'random_state': range(2, 100)}

        offspring_forest_features = dict(max_depth=offspring[1].max_depth,
                                         min_samples_split=offspring[1].min_samples_split,
                                         min_weight_fraction_leaf=offspring[1].min_weight_fraction_leaf,
                                         max_features=offspring[1].max_features, random_state=offspring[1].random_state)

        offspring_forest_features[mutate_feature] = random.choice(forest_features[mutate_feature])
        offspring_str = offspring[0][:offspring[0].find('END') + 5] + \
                       f"{offspring_forest_features['max_depth']}, {offspring_forest_features['min_samples_split']}," \
                       f" {offspring_forest_features['min_weight_fraction_leaf']}," \
                       f" {offspring_forest_features['max_features']}, {offspring_forest_features['random_state']}"

        offspring[1].__setattr__(mutate_feature, offspring_forest_features[mutate_feature])

        return offspring_str, offspring[1]

    def _prune(self, forest_offsprings, X, y):
        forest_offsprings_score = {forest_offspring: self._eval_forest(forest_offspring, X, y)
                                   for forest_offspring in forest_offsprings}

        count_dict = Counter(forest_offsprings_score)
        prune_forest = count_dict.most_common(self.num_forest)
        # print(f"minimum mse forest is: {min(count_dict, key=count_dict.get)[0]}")
        # print(f"minimum mse forest: {count_dict[min(count_dict, key=count_dict.get)]:.4f}")
        # print(f"maximum mse forest is: {max(count_dict, key=count_dict.get)[0]}")
        # print(f"maximum mse forest: {count_dict[max(count_dict, key=count_dict.get)]:.4f}\n")

        return [item[0] for item in prune_forest]

    def _eval_forest(self, ind, X, y):
        dataset_features = ind[0][7:ind[0].find('END')-2].split(', ')
        forest = ind[1]
        mse_fold = cross_val_score(forest, X[dataset_features], y, scoring='neg_mean_squared_error', cv=2)
        # kf = KFold(n_splits=3, shuffle=True)
        # mse_fold = []
        # for train_index, val_index in kf.split(X[dataset_features]):
        #     X_train, X_val = X[dataset_features].iloc[train_index], X[dataset_features].iloc[val_index]
        #     y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        #
        #     forest.fit(X_train, y_train)
        #     y_pred = forest.predict(X_val)
        #
        #     mse_fold.append(mean_squared_error(y_val, y_pred))

        # print(f"forest str: {ind[0]}")
        # print(f"forest mse: {np.mean(mse_fold):.4f} +/- {np.std(mse_fold):.4f}")

        return np.mean(mse_fold) + np.std(mse_fold)

    def fit(self, X, y):
        # print("===START===")
        self.voting_forests = self._egp(X, y)
        for forest in self.voting_forests:
            dataset_features = forest[0][7:forest[0].find('END')-2].split(', ')
            forest[1].fit(X[dataset_features], y)
        # print("===END===")

    def predict(self, X):
        if self.voting_forests is None:
            raise NotFittedError("This EnsembleGeneticProgramming instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")

        predict_values = []
        for forest in self.voting_forests:
            dataset_features = forest[0][7:forest[0].find('END')-2].split(', ')
            predict_values.append(forest[1].predict(X[dataset_features]))

        columns = zip(*predict_values)
        return np.array([np.mean([x for x in c]) for c in columns])

    def score(self, X, y):
        if self.voting_forests is None:
            raise NotFittedError("This EnsembleGeneticProgramming instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")
        predict_scores = []
        for forest in self.voting_forests:
            dataset_features = forest[0][7:forest[0].find('END')-2].split(', ')
            predict_scores.append(forest[1].score(X[dataset_features], y))
        return np.mean(predict_scores)

    def get_params(self, deep=True):
        return {"num_trees": self.num_trees, "num_forest": self.num_forest, "max_generations": self.max_generations,
                "k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
