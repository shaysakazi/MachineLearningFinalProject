import os
import xgboost as xgb
import numpy as np
import scipy
from xgboost import plot_importance
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime


def meta_learning(working_dir):
    meta_features_df = pd.read_csv(f'{working_dir}/RegressionAll.csv')
    datasets = list(meta_features_df['name'])

    X = meta_features_df.iloc[:, :-1]
    X = X.fillna(0)
    y = meta_features_df.iloc[:, -1]

    test_data = []

    for dataset_index in range(len(datasets)):
        dataset_name = datasets[dataset_index]

        print(f"Dataset {dataset_name}, {dataset_index+1}/100")

        test_cross_data = {'Dataset Name': dataset_name, 'Algorithm Name': 'XGBoost meta learning',
                           'Hyper-Parameters Values': None, 'Accuracy': None, 'TPR': None, 'FPR': None,
                           'Precision': None, 'AUC': ' ', 'PR Curve': ' ', 'Predict Probability': None,
                           'Predict Model': None, 'True Label': None, 'Training Time': None, 'Inference Time': None}

        X_train = X.loc[X['name'] != dataset_name]
        X_train = X_train.iloc[:, 1:]
        X_test = X.loc[X['name'] == dataset_name]
        X_test = X_test.iloc[:, 1:]

        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        meta_learning_model = xgb.XGBClassifier()

        time_before_train = datetime.now()
        meta_learning_model.fit(X_train, y_train)
        train_time = datetime.now() - time_before_train
        test_cross_data['Training Time'] = f"{train_time.microseconds} microseconds"

        time_before_predict = datetime.now()
        y_pred = meta_learning_model.predict(X_test)
        y_scores = meta_learning_model.predict_proba(X_test)
        predict_time = datetime.now() - time_before_predict
        test_cross_data['Inference Time'] = f"{predict_time.microseconds} microseconds"

        test_cross_data['Predict Probability'] = y_scores[0][1]
        test_cross_data['Predict Probability'] = f"{test_cross_data['Predict Probability']:.4f}"
        test_cross_data['Predict Model'] = 'Ensemble Genetic Programming' if y_pred[0] == 1 else 'Extra Tree Regressor'
        test_cross_data['True Label'] = 'Ensemble Genetic Programming' if y_test.values[0] == 1 \
            else 'Extra Tree Regressor'

        test_cross_data['Accuracy'] = accuracy_score(y_test, y_pred)

        test_cross_data['TPR'] = 1 if y_pred[0] == y_test.values[0] else 0
        test_cross_data['FPR'] = 0 if y_pred[0] == y_test.values[0] else 1
        test_cross_data['Precision'] = test_cross_data['TPR']

        test_data.append(test_cross_data)
        
    pd.DataFrame(test_data).to_csv('meta_learning_final_results.csv', index=False)

    X = X.iloc[:, 1:]
    meta_learning_model = xgb.XGBClassifier()
    meta_learning_model.fit(X, y)

    importance_types = ['weight', 'cover', 'gain']
    plt.rcParams["figure.figsize"] = (40, 40)
    for imp_type in importance_types:
        ax = plot_importance(meta_learning_model, max_num_features=167, importance_type=imp_type,
                             title=f'Meta Learning XGBoost {imp_type} importance')
        plt.show()

    shap.initjs()
    explainer = shap.TreeExplainer(meta_learning_model)
    shap_values = explainer.shap_values(X)
    shap.save_html('SHAP force plot.html', shap.force_plot(explainer.expected_value, shap_values, X, figsize=(20, 20)))
    shap.summary_plot(shap_values, X, plot_size=(20, 20), title="SHAP summary plot")
