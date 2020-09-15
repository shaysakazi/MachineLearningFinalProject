import os
import xgboost as xgb
import numpy as np
import scipy
from xgboost import plot_importance
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, \
    explained_variance_score, accuracy_score, roc_curve, average_precision_score, auc, precision_recall_curve, \
    roc_auc_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime


def meta_learning(working_dir):
    datasets = os.listdir(f'{working_dir}/regressionDatasets/')
    meta_features_df = pd.read_csv(f'{working_dir}/RegressionAll.csv')
    X = meta_features_df.iloc[:, :-1]
    X = X.fillna(0)
    y = meta_features_df.iloc[:, -1]

    test_data = []
    for dataset_index in range(len(datasets)):
        print(f"Dataset {dataset_index+1}/100")
        test_dataset = datasets[dataset_index]
        test_cross_data = {'Dataset Name': test_dataset[:-4], 'Algorithm Name': 'XGBoost meta learning',
                           'Hyper-Parameters Values': None, 'Accuracy': None, 'TPR': None, 'FPR': None,
                           'Precision': None, 'Predict Probability': None, 'Predict Model': None, 'True Label': None,
                           'Training Time': None, 'Inference Time': None}

        # train_datasets = [datasets[j] for j in range(len(datasets)) if j != i]

        X_train = X.loc[X['name'] != test_dataset[:-4]]
        X_train = X_train.iloc[:, 1:]
        X_test = X.loc[X['name'] == test_dataset[:-4]]
        X_test = X_test.iloc[:, 1:]

        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        meta_learning_model = xgb.XGBClassifier()
        hyper_parameters = dict(max_depth=range(3, 11), eta=np.linspace(0.01, 0.2, 10),
                                colsample_bytree=np.linspace(0.5, 1, 10))
        random_search_cv = RandomizedSearchCV(meta_learning_model, hyper_parameters, n_iter=50, cv=3) \
            .fit(X_train, y_train)

        test_cross_data['Hyper-Parameters Values'] = str(random_search_cv.best_params_) \
            .replace('\'', '').replace('{', '').replace('}', '')
        one_out_model = random_search_cv.best_estimator_

        time_before_train = datetime.now()
        one_out_model.fit(X_train, y_train)
        train_time = datetime.now() - time_before_train
        test_cross_data['Training Time'] = f"{train_time.microseconds} microseconds"

        time_before_predict = datetime.now()
        y_pred = one_out_model.predict(X_test)
        y_scores = one_out_model.predict_proba(X_test)
        predict_time = datetime.now() - time_before_predict
        test_cross_data['Inference Time'] = f"{predict_time.microseconds} microseconds"


        test_cross_data['Predict Probability'] = y_scores[0][1] if y_pred[0] == 1 else y_scores[0][0]
        test_cross_data['Predict Probability'] = f"{test_cross_data['Predict Probability']:.4f}"
        test_cross_data['Predict Model'] = 'Ensemble Genetic Programming' if y_pred[0] == 1 else 'Extra Tree Regressor'
        test_cross_data['True Label'] = 'Ensemble Genetic Programming' if y_test.values[0] == 1 \
            else 'Extra Tree Regressor'

        test_cross_data['Accuracy'] = accuracy_score(y_test, y_pred)

        test_cross_data['TPR'] = 1 if y_pred[0] == 1 else 0
        test_cross_data['FPR'] = 0 if y_pred[0] == 1 else 1
        test_cross_data['Precision'] = 1 if y_pred[0] == 1 else 0

        test_data.append(test_cross_data)

        # importance_types = ['weight', 'cover', 'gain']
        # plt.rcParams["figure.figsize"] = (40, 40)  # TODO - change size of fig
        # for imp_type in importance_types:
        #     ax = plot_importance(one_out_model, max_num_features=167, importance_type=imp_type,
        #                          title=f'Meta Learning XGBoost {imp_type} importance')
        #     plt.show()
        #
        # explainer = shap.TreeExplainer(one_out_model)  # TODO - check shap
        # shap_values = explainer.shap_values(X_train)
        # shap.force_plot(explainer.expected_value, shap_values[0], X_train)
        # shap.summary_plot(shap_values, X_train, max_display=5)


    pd.DataFrame(test_data).to_csv('meta_learning_final_results.csv', index=False)
