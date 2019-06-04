## Kaggle machine learning challenges

In this repository, I collect files trying to tackle some kaggle challenges.

So far I have the following files:

- **Titanic survivors**: Trying to predict the survivors (0: dead, 1: survive) of the Titanic disaster, based on info about the passengers (like age, number of relatives, class on board, origin, ...).  Simple sklearn classification models give reasonable predictions.

- **Don't overfit 2**: An artificial dataset of only 250 training data, each with 300 features, with binary targets. The predictions are then made on a test set of 19750 test data. I try many simple sklearn classification models. Most of them overfit the training data. I also try some feature reduction models, bayesian models and NN models. The best results are obtained by averaging over simple regressors: Lasso models with feature selection and a logistic regression model with L1 norm regularization. 
Position in the final standings: 117/2330 -> top 5%.

- **LANL Earthquake Predictions**: (Prize Money Competition) In this chalenge we must predict the remaining time until the next laboratory earthquake from true seismic data. The training set is a stream of couples: 'acoustic data' (= seismic activity), 'time to failure' (=remaining time to earthquake).  It is a pretty big training set (more than 6x10e8 datapoints). The test set is composed of 2624 streams of 1.5x10e5 acoustic datapoints, for which we must predict the time to failure. 
I have used two approaches, based on interesting kaggle kernels. One is about generating many new features from the acoustic data and then running a simple LGBM regression model. The second is to built a 1d CNN taking directly the accoustic data and tweak its architecture. I also try different Gradient Boosting models and then average over predictions. 
Position in the final standings: 2233/4541.

