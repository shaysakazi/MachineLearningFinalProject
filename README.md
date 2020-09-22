# Machine Learning Final Project

This file contains the instruction of the code pipeline.
ensemble_genetic_programming.py is our implementation of the eGP algorithm, as shown in the article:
https://arxiv.org/abs/2001.07553.
We implement as a scikit-learn regressor model API to fit the model simply use the fit function.
To predict an instance use the predict function.
All other functions are the inner functions of the algorithm.

meta_model.py is our implementation of XGBoost on the meta-features data set. We added a label column, which indicates which of the algorithm has won. One if our eGP implementation get better MSE value than Extra tree regressor

main.py is the file that runs the required tests.
All functions are documented in a pydoc format.

Authors:
Shay Sakazi
Ido Sakazi
