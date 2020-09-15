import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, \
    explained_variance_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import ExtraTreeRegressor
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(1, '/home/sakazis/workspace/MachineLearningFinalProject/')
from src.ensemble_genetic_programming import EnsembleGeneticProgramming
from src.meta_model import meta_learning

home_dir = "/Users/shaysakazi/PycharmProjects/MachineLearningFinalProject"
server_dir = '/home/sakazis/workspace/MachineLearningFinalProject/'
working_dir = home_dir


def eval_dataset(dataset_path, model_name, egp_hyper_parameters):
    dataset = pd.read_csv(f'{working_dir}/regressionDatasets/{dataset_path}')
    test_data = []
    X = dataset.iloc[:, :-1]
    X = X.fillna(0)
    X = pd.get_dummies(X)
    y = dataset.iloc[:, -1]

    outer_kf = KFold(n_splits=10, shuffle=True)
    outer_fold_index = 1

    for train_index, test_index in outer_kf.split(X):
        time = datetime.now()
        print(f'{dataset_path}, model: {model_name}, fold_number: {outer_fold_index}')
        test_cross_data = {'Dataset Name': dataset_path[:dataset_path.find('.')], 'Algorithm Name': model_name,
                           'Cross Validation': outer_fold_index, 'Hyper-Parameters Values': None,
                           'Mean Squared Error': None, 'Mean Absolute Error': None, 'Median Absolute Error': None,
                           'R2 Score': None, 'Explained Variance Score': None, 'Training Time': None,
                           'Inference Time': None}

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == 'Ensemble Genetic Programming':
            model = EnsembleGeneticProgramming()
            hyper_parameters = egp_hyper_parameters
        else:
            model = ExtraTreeRegressor()
            hyper_parameters = dict(max_depth=range(1, 9), min_samples_split=range(2, 8), splitter=['random', 'best'])
        random_search_cv = RandomizedSearchCV(model, hyper_parameters, n_iter=50, cv=3).fit(X_train, y_train)

        test_cross_data['Hyper-Parameters Values'] = str(random_search_cv.best_params_)\
            .replace('\'', '').replace('{', '').replace('}', '')

        fold_model = random_search_cv.best_estimator_

        time_before_train = datetime.now()
        fold_model.fit(X_train, y_train)
        train_time = datetime.now() - time_before_train
        test_cross_data['Training Time'] = f"{train_time.microseconds} microseconds"

        y_pred = fold_model.predict(X_test)
        test_cross_data['Mean Squared Error'] = mean_squared_error(y_test, y_pred)
        test_cross_data['Mean Absolute Error'] = mean_absolute_error(y_test, y_pred)
        test_cross_data['Median Absolute Error'] = median_absolute_error(y_test, y_pred)
        test_cross_data['R2 Score'] = r2_score(y_test, y_pred)
        test_cross_data['Explained Variance Score'] = explained_variance_score(y_test, y_pred)

        time_before_predict = datetime.now()
        if len(X) < 1000:
            fold_model.predict(X)
        else:
            fold_model.predict(X.iloc[:1000])
        predict_time = datetime.now() - time_before_predict
        test_cross_data['Inference Time'] = f"{predict_time.microseconds} microseconds"

        # Putting all together
        test_data.append(test_cross_data)
        time = datetime.now() - time
        print(f"fold {outer_fold_index} has ended after {time.seconds / 60:.4f} minutes\n")
        outer_fold_index += 1
    return test_data


def eval_datasets():
    dataset_index = 1
    egp_hyper_parameters = dict(num_trees=range(50, 500, 50), num_forest=range(10, 40, 5),
                                max_generations=range(10, 90, 10))
    egp_str = str(egp_hyper_parameters).replace('\'', '').replace('{', '').replace('}', '')
    print("EGP hyper-parameters are: " + egp_str)
    for dataset_path in os.listdir(f'{working_dir}/regressionDatasets/'):
        time = datetime.now()
        print(f"dataset index: {dataset_index}, dataset name: {dataset_path}")
        dataset_results = eval_dataset(dataset_path, 'Ensemble Genetic Programming', egp_hyper_parameters)
        pd.DataFrame(dataset_results).to_csv(f'results_{dataset_path}', index=False)
        time = datetime.now() - time
        print(f"Finish {dataset_path} dataset after {time.seconds / 60:.4f} minutes\n")
        dataset_index += 1


def main():
    # eval_datasets()
    meta_learning(working_dir)


if __name__ == '__main__':
    print('****** START MAIN ******')
    main()
    print('****** END MAIN ******')
