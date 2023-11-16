import numpy as np


def train_test_data(seed=42):
    rnd = np.random.RandomState(seed=42)

    X = rnd.choice(range(0, 4), size=(20, 10), p=(0.8, 0.1, 0.05, 0.05))

    O_train = rnd.choice([0, 1], size=(20, 10), p=(0.7, 0.3))
    O_test = np.ones_like(O_train) - O_train

    return X, O_train, O_test
