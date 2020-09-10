from sklearn.ensemble import RandomForestRegressor
from src.ensemble_genetic_programming import EnsembleGeneticProgramming
import pandas as pd

home_dir = "/Users/shaysakazi/PycharmProjects/MachineLearningFinalProject"
server_dir = None
working_dir = home_dir


def main():
    df = pd.read_csv(f'{working_dir}/regressionDatasets/Admission_Predict_kaggle.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    reg = RandomForestRegressor()
    reg.fit(X, y)
    reg.predict(X)
    egp = EnsembleGeneticProgramming()
    egp.fit(X, y)
    egp.predict(X)


def eval_dataset():
    pass


if __name__ == '__main__':
    main()
