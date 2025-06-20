import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def amae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype("double")
    amaes = amaes[~np.isnan(amaes)]
    return np.mean(amaes)


def mmae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype("double")
    amaes = amaes[~np.isnan(amaes)]
    return amaes.max()


def rank_probability_score(y, yproba):
    y = np.clip(y, 0, yproba.shape[1] - 1)

    yoh = np.zeros(yproba.shape)
    yoh[np.arange(len(y)), y] = 1

    yoh = yoh.cumsum(axis=1)
    yproba = yproba.cumsum(axis=1)

    rps = 0
    for i in range(len(y)):
        if y[i] in np.arange(yproba.shape[1]):
            rps += np.power(yproba[i] - yoh[i], 2).sum()
        else:
            rps += 1
    return rps / len(y)


def mes(y_true, y_pred):
    # Compute the Mean of the Sensitivities of the first and last class
    sensitivities = np.array(recall_score(y_true, y_pred, average=None))
    # Compute the mean of the sensitivities
    mes = (sensitivities[0] + sensitivities[-1]) / 2.0
    return mes
