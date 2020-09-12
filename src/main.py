import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, \
    explained_variance_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.tree import ExtraTreeRegressor
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(1, '/home/sakazis/workspace/MachineLearningFinalProject/')
from src.ensemble_genetic_programming import EnsembleGeneticProgramming

home_dir = "/Users/shaysakazi/PycharmProjects/MachineLearningFinalProject"
server_dir = '/home/sakazis/workspace/MachineLearningFinalProject/'
working_dir = home_dir

# def test():
#     df = pd.read_csv(f'{working_dir}/regressionDatasets/Admission_Predict_kaggle.csv')
#     X = df.iloc[:, :-1]
#     y = df.iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
#     egp = EnsembleGeneticProgramming()
#     egp.fit(X_train, y_train)
#     egp_y_pred = egp.predict(X_test)
#     print(f"egp prediction mse: {mean_squared_error(y_test, egp_y_pred):.4f}")
#     print(f"egp prediction score: {egp.score(X_test, y_test):.4f}")
#
#     etr = ExtraTreeRegressor()
#     etr.fit(X_train, y_train)
#     etr_y_pred = etr.predict(X_test)
#     print(f"etr prediction mse: {mean_squared_error(y_test, etr_y_pred):.4f}")
#     print(f"etr prediction score: {etr.score(X_test, y_test):.4f}")
#
#     rfr = RandomForestRegressor()
#     rfr.fit(X_train, y_train)
#     rfr_y_pred = rfr.predict(X_test)
#     print(f"rfr prediction mse: {mean_squared_error(y_test, rfr_y_pred):.4f}")
#     print(f"rfr prediction score: {rfr.score(X_test, y_test):.4f}")


def eval_dataset(dataset_path, model_name, egp_hyper_parameters, egp_str):
    dataset = pd.read_csv(f'{working_dir}/regressionDatasets/{dataset_path}')
    test_data = []
    X = dataset.iloc[:, :-1]
    X = pd.get_dummies(X)
    y = dataset.iloc[:, -1]

    outer_kf = KFold(n_splits=10, shuffle=True)
    outer_fold_index = 1

    for train_index, test_index in outer_kf.split(X):
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
        outer_fold_index += 1
    return test_data


def main():
    test_df = []
    dataset_index = 1
    egp_hyper_parameters = dict(num_trees=range(50, 100, 50), num_forest=range(2, 20, 2),
                                max_generations=range(2, 10, 1))
    egp_str = str(egp_hyper_parameters).replace('\'', '').replace('{', '').replace('}', '')
    print("EGP hyper-parameters are: " + egp_str)
    for dataset_path in os.listdir(f'{working_dir}/regressionDatasets/'):
        print(f"dataset index: {dataset_index}, dataset name: {dataset_path}")
        test_df.extend(eval_dataset(dataset_path, 'Ensemble Genetic Programming', egp_hyper_parameters, egp_str))
        test_df.extend(eval_dataset(dataset_path, 'Extra Tree Regressor', egp_hyper_parameters, egp_str))
        dataset_index += 1
    pd.DataFrame(test_df).to_csv(f'results_{egp_str}.csv', index=False)


if __name__ == '__main__':
    print('****** START MAIN ******')
    main()
    print('****** END MAIN ******')
