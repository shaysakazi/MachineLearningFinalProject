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


def eval_dataset(dataset_path, model_name, hyper_parameters):
    dataset = pd.read_csv(f'{working_dir}/regressionDatasets/{dataset_path}')
    test_data = []
    X = dataset.iloc[:, :-1]
    X = X.fillna(0)
    X = pd.get_dummies(X)
    y = dataset.iloc[:, -1]

    outer_kf = KFold(n_splits=10, shuffle=True)
    outer_fold_index = 1
    model = EnsembleGeneticProgramming() if model_name == 'Ensemble Genetic Programming' else ExtraTreeRegressor()

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

        random_search_cv = RandomizedSearchCV(model, hyper_parameters, n_iter=50, cv=3).fit(X_train, y_train)
        best_params = random_search_cv.best_params_

        test_cross_data['Hyper-Parameters Values'] = str(best_params)\
            .replace('\'', '').replace('{', '').replace('}', '')

        if model_name == 'Ensemble Genetic Programming':
            model.set_params(num_trees=best_params['num_trees'], num_forest=best_params['num_forest'],
                             max_generations=best_params['max_generations'])
        else:
            model.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
                             splitter=best_params['splitter'])

        time_before_train = datetime.now()
        model.fit(X_train, y_train)
        train_time = datetime.now() - time_before_train
        test_cross_data['Training Time'] = f"{train_time.microseconds} microseconds"

        y_pred = model.predict(X_test)
        test_cross_data['Mean Squared Error'] = mean_squared_error(y_test, y_pred)
        test_cross_data['Mean Absolute Error'] = mean_absolute_error(y_test, y_pred)
        test_cross_data['Median Absolute Error'] = median_absolute_error(y_test, y_pred)
        test_cross_data['R2 Score'] = r2_score(y_test, y_pred)
        test_cross_data['Explained Variance Score'] = explained_variance_score(y_test, y_pred)

        time_before_predict = datetime.now()
        if len(X) < 1000:
            model.predict(X)
            X_inference_time = X.__deepcopy__()
            while len(X_inference_time) < 1000:
                X_inference_time = X_inference_time.append(X_inference_time)
            model.predict(X.iloc[:1000])
        else:
            model.predict(X.iloc[:1000])
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
    egp_hyper_parameters = dict(num_trees=range(20, 110, 10), num_forest=range(2, 11, 2),
                                max_generations=range(5, 25, 5))
    etr_hyper_parameters = dict(max_depth=range(1, 9), min_samples_split=range(2, 8), splitter=['random', 'best'])
    dataset_results = []
    egp_str = str(egp_hyper_parameters).replace('\'', '').replace('{', '').replace('}', '')
    print("EGP hyper-parameters are: " + egp_str)

    for dataset_path in os.listdir(f'{working_dir}/regressionDatasets/'):
        time = datetime.now()
        print(f"dataset index: {dataset_index}, dataset name: {dataset_path}")
        dataset_results.extend(eval_dataset(dataset_path, 'Ensemble Genetic Programming', egp_hyper_parameters))
        dataset_results.extend(eval_dataset(dataset_path, 'Extra Tree Regressor', etr_hyper_parameters))
        time = datetime.now() - time
        print(f"Finish {dataset_path} dataset after {time.seconds / 60:.4f} minutes\n")
        dataset_index += 1
        break
    pd.DataFrame(dataset_results).to_csv(f'{working_dir}/Outputs/final_results.csv', index=False)


# def illustrate():
#     df = pd.read_csv(f"{working_dir}/regressionDatasets/Admission_Predict_kaggle.csv")
#     egp = EnsembleGeneticProgramming(num_trees=50, num_forest=5, max_generations=2)
#     X = df.iloc[:, :-1]
#     X = X.fillna(0)
#     X = pd.get_dummies(X)
#     y = df.iloc[:, -1]
#     egp.fit(X, y)


def main():
    eval_datasets()
    # meta_learning(working_dir)
    # illustrate()


if __name__ == '__main__':
    print('****** START MAIN ******')
    main()
    print('****** END MAIN ******')
