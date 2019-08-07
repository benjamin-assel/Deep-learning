## Kaggle machine learning challenges

In this repository, I collect files trying to tackle some kaggle challenges.

So far I have the following files:

- **Titanic survivors**: Trying to predict the survivors (0: dead, 1: survive) of the Titanic disaster, based on info about the passengers (like age, number of relatives, class on board, origin, ...).  Simple sklearn classification models give reasonable predictions.
    link: https://www.kaggle.com/c/titanic

- **Don't overfit 2**: An artificial dataset of only 250 training data, each with 300 features, with binary targets. The predictions are then made on a test set of 19750 test data. I try many simple sklearn classification models. Most of them overfit the training data. I also try some feature reduction models, bayesian models and NN models. The best results are obtained by averaging over simple regressors: Lasso models with feature selection and a logistic regression model with L1 norm regularization.  
Position in the final standings: 117/2330 -> top 5%.  
    link: https://www.kaggle.com/c/dont-overfit-ii

- **LANL Earthquake Predictions**: (Prize Money Competition) In this challenge we must predict the remaining time until the next laboratory earthquake from true seismic data. The training set is a stream of couples: 'acoustic data' (= seismic activity), 'time to failure' (=remaining time to earthquake).  It is a pretty big training set (more than 6x10e8 datapoints). The test set is composed of 2624 streams of 1.5x10e5 acoustic datapoints, for which we must predict the time to failure. 
I have used two approaches, based on interesting kaggle kernels. One is about generating many new features from the acoustic data and then running a simple LGBM regression model. The second is to built a 1d CNN taking directly the accoustic data and tweak its architecture. I also try different Gradient Boosting models and then average over predictions.  
Position in the final standings: 2233/4541.  
    link: https://www.kaggle.com/c/LANL-Earthquake-Prediction

- **Cactus Identification**: The task is to identify images which contain a cactus, so it is a binary classification task. There are 75000 training images (32x32 pixels, RGB) and the predictions are made on 4000 test images. 
This is a rather simple task with nowadays tools, so the goal is mostly to practice with CNN models, rather than trying to find a smart approach. I reach accuracy 0.998.  
Position in the standings: 687/1228.  
    link: https://www.kaggle.com/c/aerial-cactus-identification

- **Dog Breeds Classification**: The data consists of about 10,000 images of dogs of 120 different breeds. The task is to train a model to identify the correct breed from an image. Since this is not a lot of images, the main idea is to use Transfer Learning from pretrained image classification models. Here I use Google's Inception V3 CNN, with extra FC layers at the end. The model achieves 80% accuracy, which is not too bad.   
    link: https://www.kaggle.com/c/dog-breed-identification/overview

- **Gendered Pronoun Resolution**: The data consists of texts, each containing a pronoun P and two nouns A and B. The task is to determine if P refers to A, B or none of them. The kaggle data comes unlabeled, but one can use the GAP files which contain the labels (A, B or Neither for each text). Following other competitors'kernels, I use the BERT contextual embeddings of the A, B and P words (Transfer Learning technique) and train a Multi-Layer Perceptron (using Keras) to make predictions. 
