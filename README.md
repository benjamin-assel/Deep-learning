## Kaggle machine learning chalenges

In this repository, I collect files trying to address some kaggle chalenges. The "datasets" folder contains the datasets used for the chalenges.

So far I have the following files:

- **Titanic survivors**: Trying to predict the survivors (0: dead, 1: survive) of the Titanic disaster, based on info about the passengers (like age, number of relatives, class on board, origin, ...).  Simple classification models give reasonable predictions.

- **Don't overfit 2**: An artificial dataset with only 250 training data, each with 300 features, with target binaries. The predictions are then made on a test set of 19750 entries. I try many simple sklearn classification models. Most of them overfit the training data. I also try some feature reduction models, bayesian models and NN models.
