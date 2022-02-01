import torch
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import PolynomialFeatures
from torchvision import datasets


# Binary classification with Adult UCI dataset, https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
# adaptation_task options: {add_data, remove_data, change_regulariser, change_model}
class UCIDataGenerator():
    def __init__(self, adaptation_task="add_data", seed=1):

        self.adaptation_task = adaptation_task
        np.random.seed(seed)

        # Load UCI Adult data
        data = load_svmlight_file("data/UCI/a7a", n_features=123)
        X_train_initial = data[0].todense().astype(np.float32)
        self.labels_train = data[1].astype(int)

        # Randomly choose 10% of data for all adaptation tasks except "remove_data"
        if self.adaptation_task == "remove_data":
            num_datapoints = (int)(X_train_initial.shape[0])
            perm_inds = list(range(X_train_initial.shape[0]))
        else:
            num_datapoints = (int)(X_train_initial.shape[0] * 0.01)
            perm_inds = list(range(X_train_initial.shape[0]))
            np.random.shuffle(perm_inds)

        self.X_train_initial = X_train_initial[perm_inds[:num_datapoints]]
        self.labels_train = self.labels_train[perm_inds[:num_datapoints]]

        data_test = load_svmlight_file("data/UCI/a7a_test", n_features=123)
        self.X_test_initial = torch.from_numpy(data_test[0].todense().astype(np.float32))
        self.labels_test = torch.tensor(data_test[1], dtype=torch.long)

        # Convert from classes {-1, 1} to {0, 1}
        self.labels_train[self.labels_train < 0] = 0
        self.labels_test[self.labels_test<0] = 0


    # Base task to train on first
    def base_task_data(self):

        # If "change_model", polynomial degree 2, else degree 1
        if self.adaptation_task == "change_model":
            self.update_polynomial_degree(2)
        else:
            self.update_polynomial_degree(1)

        # If "add_data", choose 90% of the data for base task
        if self.adaptation_task == "add_data":
            num_datapoints_base_task = (int)(self.X_train.shape[0] * 0.9)
            X_train = self.X_train[:num_datapoints_base_task]
            labels_train = self.labels_train[:num_datapoints_base_task]
        else:
            X_train = self.X_train
            labels_train = self.labels_train

        self.number_base_points = len(X_train)

        return (torch.from_numpy(X_train), torch.from_numpy(labels_train).reshape(-1)), \
               (self.X_test, self.labels_test.reshape(-1))


    # Adaptation task (second task) to train on
    def adaptation_task_data(self):

        # If "change_model", convert polynomial degree to 1
        if self.adaptation_task == "change_model":
            self.update_polynomial_degree(1)

        # If "add_data", choose 10% of the data for adaptation task
        if self.adaptation_task == "add_data":
            num_datapoints_base_task = (int)(self.X_train.shape[0] * 0.9)
            X_train = self.X_train[num_datapoints_base_task:]
            labels_train = self.labels_train[num_datapoints_base_task:]

            return (torch.from_numpy(X_train), torch.from_numpy(labels_train).reshape(-1)), \
                   (self.X_test, self.labels_test.reshape(-1))
        else:
            return None, (self.X_test, self.labels_test.reshape(-1))


    # Update polynomial degree
    def update_polynomial_degree(self, polynomial_degree=1):
        self.poly = PolynomialFeatures(polynomial_degree)
        self.X_train = self.poly.fit_transform(self.X_train_initial)
        self.X_test = self.poly.fit_transform(self.X_test_initial)
        self.X_test = torch.from_numpy(self.X_test.astype(np.float32))
        self.dimensions = self.X_train.shape[1]



# USPS odd vs even dataset (binary classification)
# adaptation_task options: {add_data, remove_data, change_regulariser, change_model}
class USPSDataGenerator():
    def __init__(self, adaptation_task="add_data", polynomial_degree=None, seed=0, path=None):

        self.adaptation_task = adaptation_task
        self.polynomial_degree = polynomial_degree
        np.random.seed(seed)

        # Load USPS data
        train_dataset = datasets.USPS(root='./data', train=True)
        test_dataset = datasets.USPS(root='./data', train=False)

        self.X_train_initial = np.array(train_dataset.data)
        self.labels_train = np.array(train_dataset.targets)
        self.X_test_initial = np.array(test_dataset.data)
        self.labels_test = np.array(test_dataset.targets)


    # Base task to train on first
    def base_task_data(self):
        # If "change_model" and GLM, polynomial degree 2, else degree 1
        if self.polynomial_degree is not None:
            if self.adaptation_task == "change_model":
                self.update_polynomial_degree(2)
            else:
                self.update_polynomial_degree(1)

        else:
            self.X_train = self.X_train_initial
            self.X_test = self.X_test_initial
            self.dimensions = self.X_train.shape[1]

        # If "add_data", all digits except the digit '9' is base task
        if self.adaptation_task == "add_data":
            return self.data_split(digit_set=[0,1,2,3,4,5,6,7,8])
        else:
            return self.data_split(digit_set=[0,1,2,3,4,5,6,7,8,9])


    # Adaptation task (second task) to train on
    def adaptation_task_data(self):

        # If "change_model" and GLM, convert polynomial degree to 1
        if self.polynomial_degree is not None and self.adaptation_task == "change_model":
            self.update_polynomial_degree(1)

        # If "add_data", add digit '9'
        if self.adaptation_task == "add_data":
            train_data, _ = self.data_split(digit_set=[9])
            _, test_data = self.data_split(digit_set=[0,1,2,3,4,5,6,7,8,9])
            return train_data, test_data

        # If "remove_data", remove digit '8'
        elif self.adaptation_task == "remove_data":
            train_data, _ = self.data_split(digit_set=[8])
            _, test_data = self.data_split(digit_set=[0,1,2,3,4,5,6,7,9])
            return train_data, test_data

        else:
            _, test_data = self.data_split(digit_set=[0,1,2,3,4,5,6,7,8,9])
            return None, test_data


    # Update polynomial degree
    def update_polynomial_degree(self, polynomial_degree=None):
        if polynomial_degree is not None:
            self.poly = PolynomialFeatures(polynomial_degree)
            self.X_train = self.poly.fit_transform(self.X_train_initial)
            self.X_test = self.poly.fit_transform(self.X_test_initial).astype(np.float32)
            self.dimensions = self.X_train.shape[1]


    # Return trainloader and testloader for a specific split_ind
    def data_split(self, digit_set=[]):
        next_x_train = None
        next_y_train = None
        next_x_test = None
        next_y_test = None

        # Loop over all classes in current iteration
        for digit in digit_set:

            # Find the correct set of training inputs, and stack
            train_id = np.where(self.labels_train == digit)[0]
            if next_x_train is None:
                next_x_train = self.X_train[train_id]
            else:
                next_x_train = np.vstack((next_x_train, self.X_train[train_id]))

            # Only interested in binary classification
            next_y_train_interm = np.abs(digit)*np.ones(len(train_id), dtype=np.int64)
            next_y_train_interm = next_y_train_interm % 2

            if next_y_train is None:
                next_y_train = next_y_train_interm
            else:
                next_y_train = np.hstack((next_y_train, next_y_train_interm))

            # Repeat above process for test inputs
            test_id = np.where(self.labels_test == digit)[0]
            if next_x_test is None:
                next_x_test = self.X_test[test_id]
            else:
                next_x_test = np.vstack((next_x_test, self.X_test[test_id]))

            # Only interested in binary classification
            next_y_test_interm = np.abs(digit)*np.ones(len(test_id), dtype=np.int64)
            next_y_test_interm = next_y_test_interm % 2

            if next_y_test is None:
                next_y_test = next_y_test_interm
            else:
                next_y_test = np.hstack((next_y_test, next_y_test_interm))

        if next_x_train is not None:
            inputs_train = torch.from_numpy(next_x_train)
            labels_train = torch.from_numpy(next_y_train)
            inputs_test = torch.from_numpy(next_x_test)
            labels_test = torch.from_numpy(next_y_test)

            self.number_base_points = len(inputs_train)

            return (inputs_train, labels_train), (inputs_test, labels_test)
        else:
            return (None, None)
