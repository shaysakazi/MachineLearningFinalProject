import random
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
import copy


class EnsembleGeneticProgramming:
    """
    Ensemble Genetic Programming class implemented with a scikit-learn regressor API
    """
    def __init__(self, num_trees=100, num_forest=100, max_generations=10, k=2):
        """
        Constructor of the eGP class
        :param num_trees: number of trees which forests are composed of.
        :param num_forest: number of forest to construct
        :param max_generations: number of generation to run
        :param k: number of forest to compare in the

        max_time: stop algorithm session if it surpass its max time limit
        voting_forests: final voting forests
        """
        self.num_trees = num_trees
        self.num_forest = num_forest
        self.max_generations = max_generations
        self.k = k
        self.voting_forests = None

        # self.illustrate_forest = []
        # self.gn = 0
        # self.s = 0
        # self.X = None
        # self.y = None
        # pd.DataFrame(self.illustrate_forest).to_csv('ill.csv',index=False)

    def _egp(self, X, y):
        """
        This function is the main controller of the algorihm, this is where the stages are taking place
        :param X: dataset features
        :param y: dataset target value
        :return: forest list which will be used for predicting
        """
        forest_list = self._generate_forests(X)  # list of (str, estimator) tuples
        # self.gn = 'Before generation'
        # self.s = 'Generate forests'
        # self.X = X
        # self.y = y
        # for forest in forest_list:
        #     self._add_illustrate(forest, X, y, self.gn, self.s)
        g = 0
        start_time = datetime.now()
        time_delta = datetime.now() - start_time
        # self.gn = g
        while g < self.max_generations and time_delta.seconds < 5:
            # print(f"Generation index: {g}, number of forests: {len(forest_list)},"
            #       f" max_generations: {self.max_generations}")

            forest_parents = self._select_forests(forest_list, X, y)  # choose N forests
            # self.s = f"Select forests => best selected forests"
            # for forest in forest_parents:
            #     self._add_illustrate(forest, X, y, self.gn, self.s)
            # time = datetime.now() - time
            # print(f"Finish selection on Generation index: {g} after {time.seconds} seconds")
            # time = datetime.now()
            forest_offsprings = self._breeding(forest_parents)  # create N off springs
            # time = datetime.now() - time
            # print(f"Finish breeding on Generation index: {g} after {time.seconds} seconds")
            # time = datetime.now()
            # self.s = 'all forest before pruning'
            # for forest in forest_offsprings:
            #     self._add_illustrate(forest, X, y, self.gn, self.s)
            forest_list = self._prune(forest_offsprings, X, y)  # prune 2N (N from forest selection and N from off
            # springs) into N forests
            # time = datetime.now() - time
            # print(f"Finish pruning on Generation index: {g} after {time.seconds} seconds")
            # self.s = 'best forests after pruning'
            # for forest in forest_list:
            #     self._add_illustrate(forest, X, y, self.gn, self.s)
            g += 1
            time_delta = datetime.now() - start_time
            print(time_delta.seconds)
            # self.gn = g

        # pd.DataFrame(self.illustrate_forest).to_csv('illustrate.csv', index=False)
        return forest_list

    # def _best_model(self, forest_list, X, y):
    #     forests_scores = {forest_offspring: self._eval_forest(forest_offspring, X, y)
    #                                for forest_offspring in forest_list}
    #
    #     count_dict = Counter(forests_scores)
    #     best_forest = count_dict.most_common(1)
    #
    #     dataset_features = best_forest[0][7:best_forest[0].find('END')-2].split(', ')
    #     return best_forest[0], best_forest[1].fit(X[dataset_features], y)

    def _generate_forests(self, X):
        """
        This function iterate over number of desired forest to generated
        :param X: dataset features
        :return: generated forests list
        """
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
        """
        This function generate forest randomly, where takes in a random propability the forest features
        and choose randomly the dataset features.
        :param X: dataset features
        :return: generated forest => tuple consist forest features and dataset features
        """
        dataset_features = X.columns
        forest_features = {'max_depth': range(2, 10),
                           'min_samples_split': range(2, 10),
                           'min_weight_fraction_leaf': np.linspace(0.05, 0.5, 10),
                           'max_features': ['auto', 'sqrt', 'log2']}

        # choose randomly random forest features
        max_depth = random.choice(forest_features['max_depth'])
        min_samples_split = random.choice(forest_features['min_samples_split'])
        min_weight_fraction_leaf = random.choice(forest_features['min_weight_fraction_leaf'])
        max_features = random.choice(forest_features['max_features'])

        # choose randomly dataset features
        used_features = self._choose_features(dataset_features)

        rf_features = {'dataset_features': list(used_features),
                       'forest_features': {'max_depth': max_depth,
                                         'min_samples_split': min_samples_split,
                                         'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                         'max_features': max_features}}

        # rf_str = 'START, ' + ', '.join(used_features) + ', END, ' + \
        #          f'{max_depth}, {min_samples_split}, {min_weight_fraction_leaf}, {max_features}'
        rf_reg = RandomForestRegressor(n_estimators=self.num_trees, max_depth=max_depth,
                                       min_samples_split=min_samples_split, max_features=max_features,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf)

        return rf_features, rf_reg

    def _choose_features(self, features):
        """
        This fucntion decide which dataset features a generated forest will have
        :param features: dataset features
        :return: the chosen dataset features
        """
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
        """
        This function yields the best forests that wins the tournament selection
        The amount of forest depends on the num_forest parameter
        :param forest_list:
        :param X: dataset features
        :param y: dataset target value 
        :return: forest tournament selection winners
        """
        forest_parents = []
        for _ in range(self.num_forest):
            # time = datetime.now()
            # self.s = f"Select forests => tournament selection index {_}"
            selected_forest = self._tournament_selection(forest_list, X, y)
            # time = datetime.now() - time
            # print(f"_tournament_selection after {time.seconds} seconds")
            forest_parents.append(selected_forest)

        return forest_parents

    def _tournament_selection(self, forest_list, X, y):
        """
        This function create a tournament between k randomly chosen forest from the generated forest
        it yields the best forest between the randomly selected
        :param forest_list: this is the list which randomly choose forest from
        :param X: dataset features
        :param y: dataset target value 
        :return: forest tournament selection winner
        """
        best = None
        best_mse = 0
        for i in range(self.k):
            ind = random.choice(forest_list)
            # self._add_illustrate(ind, X, y, self.gn, self.s)
            # time = datetime.now()
            ind_mse = self._eval_forest(ind, X, y)
            # time = datetime.now() - time
            # print(f"_eval_forest after {time.seconds} seconds")
            if best is None or ind_mse > best_mse:
                best = ind
                best_mse = ind_mse
        return best

    def _breeding(self, forest_parents):
        """
        This function create offspring form a given list of parents
        :param forest_parents: list of parents, from this list the offspring will created
        :return: created offsprings
        """
        offsprings = copy.deepcopy(forest_parents)
        for i in range(self.num_forest):
            # self.s = f"breeding index {i}"
            offspring = self._breed(forest_parents)
            offsprings.append(offspring)
        return offsprings

    def _breed(self, forest_parents):
        """
        This function created a new offspring
        :param forest_parents: a list of parents candidates
        :return: created offspring
        """
        parent_1 = random.choice(forest_parents)
        parent_2 = random.choice(forest_parents)

        # self._add_illustrate(parent_1, self.X, self.y, self.gn, self.s + ' parent 1')
        # self._add_illustrate(parent_2, self.X, self.y, self.gn, self.s + ' parent 2')

        offspring = self._uniform_crossover(parent_1, parent_2)
        mutation_prob = random.random()
        if mutation_prob < 0.5:
            offspring = self._mutate(offspring)
        # else:
        #     self.s = self.s + ' offspring'
        # self._add_illustrate(offspring, self.X, self.y, self.gn, self.s)
        return offspring

    def _uniform_crossover(self, parent_1, parent_2):
        """
        This function generated a new offspring from a given two parents using uniform cross over
        :param parent_1: first parent
        :param parent_2: second parent
        :return: new offspring
        """
        offspring_forest_features = {'max_depth': None, 'min_samples_split': None, 'min_weight_fraction_leaf': None,
                                     'max_features': None}
        parent_1_forest_features = {'max_depth': parent_1[1].max_depth,
                                    'min_samples_split': parent_1[1].min_samples_split,
                                    'min_weight_fraction_leaf': parent_1[1].min_weight_fraction_leaf,
                                    'max_features': parent_1[1].max_features}
        parent_2_forest_features = {'max_depth': parent_2[1].max_depth,
                                    'min_samples_split': parent_2[1].min_samples_split,
                                    'min_weight_fraction_leaf': parent_2[1].min_weight_fraction_leaf,
                                    'max_features': parent_2[1].max_features}

        for key in offspring_forest_features:
            p = random.random()
            if p < 0.5:
                offspring_forest_features[key] = parent_1_forest_features[key]
            else:
                offspring_forest_features[key] = parent_2_forest_features[key]

        selected_parent_1_dataset_features = parent_1[0]['dataset_features']
        selected_parent_2_dataset_features = parent_2[0]['dataset_features']

        offspring_dataset_features = list(set(selected_parent_1_dataset_features + selected_parent_2_dataset_features))

        rf_features = {'dataset_features': list(offspring_dataset_features),
                       'forest_features': {'max_depth': offspring_forest_features['max_depth'],
                                           'min_samples_split': offspring_forest_features['min_samples_split'],
                                           'min_weight_fraction_leaf':
                                               offspring_forest_features['min_weight_fraction_leaf'],
                                           'max_features': offspring_forest_features['max_features']}}

        rf_reg = RandomForestRegressor(n_estimators=self.num_trees, max_depth=offspring_forest_features['max_depth'],
                                       min_samples_split=offspring_forest_features['min_samples_split'],
                                       max_features=offspring_forest_features['max_features'],
                                       min_weight_fraction_leaf=offspring_forest_features['min_weight_fraction_leaf'])

        return rf_features, rf_reg

    def _mutate(self, offspring):
        """
        This function applied a mutation for an offspring means the offspring can have a feature that is parents
        did not have.
        :param offspring: the new generated offspring
        :return: offspring after mutation has applied
        """
        mutate_feature = random.choice(['max_depth', 'min_samples_split', 'min_weight_fraction_leaf', 'max_features'])
        forest_features = {'max_depth': range(2, 10),
                           'min_samples_split': range(2, 10),
                           'min_weight_fraction_leaf': np.linspace(0.05, 0.5, 10),
                           'max_features': ['auto', 'sqrt', 'log2']}

        offspring_forest_features = dict(max_depth=offspring[1].max_depth,
                                         min_samples_split=offspring[1].min_samples_split,
                                         min_weight_fraction_leaf=offspring[1].min_weight_fraction_leaf,
                                         max_features=offspring[1].max_features)

        offspring_forest_features[mutate_feature] = random.choice(forest_features[mutate_feature])
        # self.s += f' offspring mutate feature: {mutate_feature}'
        offspring[1].__setattr__(mutate_feature, offspring_forest_features[mutate_feature])
        offspring_features = {'dataset_features': offspring[0]['dataset_features'],
                              'forest_features': {'max_depth': offspring_forest_features['max_depth'],
                                                  'min_samples_split': offspring_forest_features['min_samples_split'],
                                                  'min_weight_fraction_leaf':
                                                      offspring_forest_features['min_weight_fraction_leaf'],
                                                  'max_features': offspring_forest_features['max_features']}}
        return offspring_features, offspring[1]

    def _prune(self, forest_offsprings, X, y):
        """
        This function takes the best forests from the forest_offsprings list
        :param forest_offsprings: list of forest candidates
        :param X: dataset feature
        :param y: dataset target value 
        :return: The best forests from the list of forest candidates
        """
        forest_offsprings_score = {i: self._eval_forest(forest_offsprings[i], X, y)
                                   for i in range(len(forest_offsprings))}

        count_dict = Counter(forest_offsprings_score)
        prune_forest = count_dict.most_common(self.num_forest)
        # print(f"minimum mse forest is: {min(count_dict, key=count_dict.get)[0]}")
        # print(f"minimum mse forest: {count_dict[min(count_dict, key=count_dict.get)]:.4f}")
        # print(f"maximum mse forest is: {max(count_dict, key=count_dict.get)[0]}")
        # print(f"maximum mse forest: {count_dict[max(count_dict, key=count_dict.get)]:.4f}\n")

        return [forest_offsprings[forest_index[0]] for forest_index in prune_forest]

    def _eval_forest(self, ind, X, y):
        """
        evaluate forest
        :param ind: forest to evaluate
        :param X: dataset features
        :param y: dataset target value 
        :return: the forest mean negative mean squared error in a 3 cross validation
        """
        dataset_features = ind[0]['dataset_features']
        forest = ind[1]
        mse_fold = cross_val_score(forest, X[dataset_features], y, scoring='neg_mean_squared_error', cv=3)

        return np.mean(mse_fold) + np.std(mse_fold)

    def fit(self, X, y):
        """
        This function is similar to scikit-learn API which use X as data features and y for labels
        :param X: dataset features
        :param y: dataset target value 

        fit the voting forest list
        """
        # print("===START===")
        self.voting_forests = self._egp(X, y)
        for forest in self.voting_forests:
            dataset_features = forest[0]['dataset_features']
            forest[1].fit(X[dataset_features], y)
        # print("===END===")

    def predict(self, X):
        """
        This function is similar to scikit-learn API which predict the target value from a given dataset features
        :param X: dataset features
        :return: target value
        """
        if self.voting_forests is None:
            raise NotFittedError("This EnsembleGeneticProgramming instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")

        predict_values = []
        for forest in self.voting_forests:
            dataset_features = forest[0]['dataset_features']
            predict_values.append(forest[1].predict(X[dataset_features]))

        columns = zip(*predict_values)
        return np.array([np.mean([x for x in c]) for c in columns])

    def score(self, X, y):
        """
        This function is similar to scikit-learn API which yields the model score
        :param X: dataset features       
        :param y: dataset target value   
        :return: model score
        """
        if self.voting_forests is None:
            raise NotFittedError("This EnsembleGeneticProgramming instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")
        predict_scores = []
        for forest in self.voting_forests:
            dataset_features = forest[0]['dataset_features']
            predict_scores.append(forest[1].score(X[dataset_features], y))
        return np.mean(predict_scores)

    def get_params(self, deep=True):
        """
        This function is similar to scikit-learn API return the model params
        :param deep: scikit-learn API parameter
        :return: return the model params
        """
        return {"num_trees": self.num_trees, "num_forest": self.num_forest, "max_generations": self.max_generations,
                "k": self.k}

    def set_params(self, **parameters):
        """
        This function is similar to scikit-learn API sets the model params
        :param parameters: params to set
        :return: model after setting the params
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # def _add_illustrate(self, forest, X, y, gn, s):
    #     forest_illustrate_features = {'Generation number': gn, 'Stage': s,
    #                                   'Forest hyper parameters': str(forest[0]['forest_features']).replace('\'', '')
    #                                       .replace('{', '').replace('}', ''),
    #                                   'Dataset features': str(forest[0]['dataset_features']).replace('\'', '')
    #                                       .replace('{', '').replace('}', ''),
    #                                   '3 cross-fold validation negative Mean Squared Error':
    #                                       self._eval_forest(forest, X, y)}
    #     self.illustrate_forest.append(forest_illustrate_features)
