import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
                      C=2**np.arange(-8, 1).astype(np.float), seed=42):
    """
    Trains a logistic regression classifier.

    Args:
        trX, trY, vaX, vaY, teX, teY (ndarray): train, validation, test dataset
        penalty (str): regularization type
        C (ndarray): regularization coefficients to test
        seed (int): seed to randomize data for training

    Returns:
        model (sklearn.LogisticRegression): the logistic regression classifer
        score (float): testing accuracy
        c (float): regularization coeffient
        nnotzero (int): number of weights different from zero
        model.coef_ (ndarray): the classifier weights
    """
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return model, score, c, nnotzero, model.coef_


def load_sst(path):
    """
    Loads SST binary dataset into inputs and labels

    Args:
        path (str): the location of the dataset

    Returns:
        X (list): the reviews
        Y (list): the scores of each review
    """
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir='data/'):
    """
    Loads the SST dataset from training, validation and testing

    Args:
        data_dir (str): the location of the dataset

    Returns:
        trX, vaX, teX, trY, vaY, teY (lists): training, validation and testing
                                              datasets
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY
