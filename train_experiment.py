import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from neural_net import NeuralNetwork
from operations import *

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("data/abalone.data", "Rings")
#X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]
net = NeuralNetwork(n_features, [32,32,16,1], [ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.01)
epochs = 500

folds = 5
kfx = KFold(n_splits=5)
kfx.get_n_splits(X)
kfy = KFold(n_splits=5)
kfy.get_n_splits(y)
X_train = [None] * folds
Y_train = [None] * folds
X_test = [None] * folds
Y_test = [None] * folds
epochs_loss = [0] * epochs
mean_absolute_errors = [None] * folds
for i, (train_index, test_index) in enumerate(kfx.split(X)):
    X_train[i] = X[train_index]
    X_test[i] = X[test_index]
for i, (train_index, test_index) in enumerate(kfy.split(y)):
    Y_train[i] = y[train_index]
    Y_test[i] = y[test_index]
for i in range(folds):
    trained_W, epoch_losses = net.train(X_train[i], Y_train[i], epochs)
    epochs_loss = np.add(epochs_loss, epoch_losses)
    mean_absolute_errors[i] = net.evaluate(X_test[i], Y_test[i], mean_absolute_error)

print("Average mean absolute error:", np.mean(mean_absolute_errors))
print("Standard deviation of mean absolute error:", np.std(mean_absolute_errors))

plt.plot(np.arange(0, epochs), np.divide(epochs_loss, folds), scalex="Epoch Number", scaley="Average training loss")
plt.show()