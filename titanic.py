import pandas
import numpy as np


def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    epochs = 100
    n = 100
    w = 0.01 * np.random.randn(trainingImages.shape[0], 2)
    trainingImages = trainingImages.T
    idxs = np.random.permutation(trainingImages.shape[0])
    randomImages = trainingImages[idxs]
    randomLabels = trainingLabels[idxs]
    trainingImages = trainingImages.T
    for i in range(epochs):
        for j in range(int(trainingImages.shape[1] / n) - 1):
            miniBatchImages = randomImages[n*j:n*j+99]
            miniBatchLabels = randomLabels[n*j:n*j+99]
            miniBatchImages = miniBatchImages.T
            gradient = gradCE(w, miniBatchImages, miniBatchLabels)
            w = w - epsilon * gradient
    return w


def CE(wtilde, Xtilde, y):
    Z = Xtilde.T.dot(wtilde)
    Z = np.exp(Z)
    yhat = (Z.T/Z.sum(axis=1)).T
    ce = (-1/Xtilde.shape[1]) * np.sum(y * np.log(yhat))
    return ce


def gradCE(wtilde, Xtilde, y):
    Z = Xtilde.T.dot(wtilde)
    Z = np.exp(Z)
    yhat = (Z.T/Z.sum(axis=1)).T
    grad = (1 / Xtilde.shape[1]) * Xtilde.dot(yhat - y)
    return grad


def prediction(wtilde, Xtilde):
    Z = Xtilde.T.dot(wtilde)
    Z = np.exp(Z)
    yhat = (Z.T/Z.sum(axis=1)).T
    yhat = (yhat == yhat.max(axis=1)[:, None]).astype(int)
    pred = np.array([])
    for p in range(yhat.shape[0]):
        if yhat[p][0] == 1:
            pred = np.append(pred, 1)
        else:
            pred = np.append(pred, 0)
    return pred


if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male": 0, "female": 1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    sib = d.SibSp.to_numpy()
    survived = d.Survived.to_numpy()
    b = np.zeros((survived.size, 2))
    b[np.arange(survived.size), survived] = 1
    survived = b
    train = np.vstack((sex, Pclass, sib))

    # Train model using part of homework 3.
    Wtilde = softmaxRegression(
        train, survived, 0, 0, epsilon=0.1, batchSize=100, alpha=.1)

    # Load testing data
    c = pandas.read_csv("test.csv")
    ids = c.PassengerId.to_numpy()
    sexTest = c.Sex.map({"male": 0, "female": 1}).to_numpy()
    PclassTest = c.Pclass.to_numpy()
    sibTest = c.SibSp.to_numpy()
    test = np.vstack((sexTest, PclassTest, sibTest))

    # Compute predictions on test set
    prediction = prediction(Wtilde, test)
    indices_one = prediction == 1
    indices_zero = prediction == 0
    prediction[indices_one] = 0  # replacing 1s with 0s
    prediction[indices_zero] = 1  # replacing 0s with 1s
    prediction = np.vstack((ids, prediction))
    prediction = prediction.T
    prediction = prediction.astype(int)

    # Write CSV file of the format:
    # PassengerId, Survived
    cities = pandas.DataFrame(prediction, columns=['PassengerId', 'Survived'])
    cities.to_csv('prediction.csv', index=False)
