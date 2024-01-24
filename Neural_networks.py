# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import itertools
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import *
from helpers import separation
import matplotlib.pyplot as plt 


def train_and_evaluate_nn_CV(X_train, y_train, X_test, y_test, input_size, hidden_sizes, learning_rates, batch_sizes, num_epochs=10):
    """Trains and evaluates a neural network with hyperparameter tuning using cross-validation.

    Args:
    - X_train (numpy array): Training feature matrix.
    - y_train (numpy array): Training target labels.
    - X_test (numpy array): Testing feature matrix.
    - y_test (numpy array): Testing target labels.
    - input_size (int): Size of the input features.
    - hidden_sizes (list): List of potential hidden layer sizes.
    - learning_rates (list): List of potential learning rates.
    - batch_sizes (list): List of potential batch sizes.
    - num_epochs (int, optional): Number of training epochs. Default is 10.

    Returns:
    - y_true (numpy array): True labels from the testing set.
    - y_pred (numpy array): Predicted labels from the testing set.
    - losses (list): List of training losses for each epoch.
    - best_params (dict): Best hyperparameters found during cross-validation.
    """

    losses = []
    class SimpleClassifier(nn.Module):
        def __init__(self, isnput_size, hidden_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    best_accuracy = 0
    best_params = {}

    for hidden_size, learning_rate, batch_size in itertools.product(hidden_sizes, learning_rates, batch_sizes):
        # Initialize the model, loss function, and optimizer
        num_classes = len(set(y_train))
        model = SimpleClassifier(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.Tensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.Tensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Start training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Print training loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())

        # Evaluate on the test set
        model.eval()
        all_predicted_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predicted_labels.extend(predicted.cpu().numpy())

        y_pred = np.array(all_predicted_labels)
        accuracy = np.mean(y_test == y_pred)

        # Update best parameters if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'batch_size': batch_size}

    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_accuracy)

    # Train the final model with the best parameters
    final_model = SimpleClassifier(input_size, best_params['hidden_size'], num_classes)
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

    for epoch in range(num_epochs):
        final_model.train()
        for inputs, labels in train_loader:
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()

    # Evaluate the final model on the test set
    final_model.eval()
    all_predicted_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = final_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted_labels.extend(predicted.cpu().numpy())

    y_pred = np.array(all_predicted_labels)
    final_accuracy = np.mean(y_test == y_pred)
    print("Final Model Accuracy:", final_accuracy)

    return y_test, y_pred, losses, best_params

# The basic Neural networks used for all
def train_and_evaluate_nn(X_train, y_train, X_test, y_test, input_size, hidden_size, num_epochs=10, batch_size=64, learning_rate=0.001):
    """Trains and evaluates a simple neural network on given training and testing data.

    Args:
    - X_train (numpy array): Training feature matrix.
    - y_train (numpy array): Training target labels.
    - X_test (numpy array): Testing feature matrix.
    - y_test (numpy array): Testing target labels.
    - input_size (int): Size of the input features.
    - hidden_size (int): Size of the hidden layer.
    - num_epochs (int, optional): Number of training epochs. Default is 10.
    - batch_size (int, optional): Batch size for training. Default is 64.
    - learning_rate (float, optional): Learning rate for optimization. Default is 0.001.

    Returns:
    - y_true (numpy array): True labels from the testing set.
    - y_pred (numpy array): Predicted labels from the testing set.
    - losses (list): List of training losses for each epoch.
    """
    # Start by converting all Numpy arrays into Pytorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    losses = []

    # Dataset for training and testing set :
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Dataloader for training and testing set :
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Define a simple Neural network class, with one hidden layer and where we've chosen ReLu as our activation function:
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Initialize the model, loss function, and optimizer
    # num_classes = len(y_train.unique())
    num_classes = len(set(y_train))  # because the labels are integers
    model = SimpleClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print training loss after each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())

    # Evaluation on the test set
    model.eval()
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    y_pred = np.array(all_predicted_labels) # np.array(all_predicted_labels)
    y_true = np.array(all_true_labels) # np.array(all_true_labels)

    return y_true, y_pred, losses


def train_and_evaluate_nn_M(X, y, num_epochs, cross_validate = False, Method = "M1", hidden_size_best = 256, learning_rate_best = 0.001, batch_size_best = 32):
    # Arrays needed for Cross Validation
    hidden_sizes = [64, 128, 256]
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    

    X_train, y_train, X_test, y_test = separation(X,y, Method)

    if(cross_validate):
        y_true, y_pred, losses, best_params = train_and_evaluate_nn_CV(X_train, y_train, X_test, y_test, input_size = len(X_train[0]), hidden_sizes = hidden_sizes, learning_rates = learning_rates, batch_sizes = batch_sizes, num_epochs = 10)
        return y_true, y_pred, losses, best_params

    else:
        y_true, y_pred, losses = train_and_evaluate_nn(X_train, y_train, X_test, y_test, input_size=len(X_train[0]), hidden_size = hidden_size_best, learning_rate = learning_rate_best, batch_size = batch_size_best, num_epochs = 10)
        return y_true, y_pred, losses 


def plot_losses(losses, losses_time, losses_camera, losses_error):
    """Plots training losses for different components of a model over epochs.

    Args:
    - losses (list): Training losses for positional data.
    - losses_time (list): Training losses for time series analysis with participants.
    - losses_camera (list): Training losses for time series analysis with Camera separation.
    - losses_error (list): Training losses for time series analysis with Error separation.

    Returns:
    - None: The function generates a plot and does not return any value.
    """
    epochs = range(1, len(losses) + 1)
    
    plt.plot(epochs, losses, 'bo-', label='Training Loss for positional data')
    plt.plot(epochs, losses_time, 'ro-', label='Training Loss for time series analysis with participants')
    plt.plot(epochs, losses_camera, 'go-', label='Training Loss for time series analysis with Camera separation')
    plt.plot(epochs, losses_error, 'yo-', label='Training Loss for time series analysis with Error separation')
    
    plt.title('Training Loss for Softmax Cross approach for all three methods')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_test_percentage(X, y, Method = 'M1'):
    """Performs training and testing with varying train-test splits and plots accuracy against train set size.

    Args:
    - X (numpy array): Feature matrix.
    - y (numpy array): Target labels.
    - Method (str, optional): Method for separation. Default is 'M1'.

    Returns:
    - accuracies (list): List of accuracies for each train-test split.
    """

    test_size_table = np.array([0.02,0.003,0.005,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    y_true_table = []
    y_pred_table = []
    losses_table = []

    for i, test_size_i in enumerate(test_size_table):
        print("Split happening!")
        X_train, y_train, X_test, y_test = separation(X,y, Method)
  
        print("Split happened!")
        
        print(f"Element {i+1}/{len(test_size_table)}")
        
        y_true, y_pred, losses = train_and_evaluate_nn(X_train, y_train, X_test, y_test, input_size=len(X_train[0]), hidden_size=256)
        
        y_true_table.append(y_true)
        y_pred_table.append(y_pred)
        losses_table.append(losses)
        print(f"{i+1}/{len(test_size_table)}")

        accuracies = []
    train_size_table = 1-test_size_table

    for y_true, y_pred in zip(y_true_table, y_pred_table):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        accuracies.append(accuracy)

    # Print or use the accuracies as needed
    for i, accuracy in enumerate(accuracies):
        print(f'Accuracy for pair {i + 1}: {accuracy * 100:.2f}%')


    # Scatter plot
    plt.scatter(train_size_table, accuracies, marker='o', color='blue')
    plt.title('Accuracy vs. Train Size for NNs')
    plt.xlabel('Train set size')
    plt.ylabel('Accuracy')
    plt.show()
    return accuracies


def train_test_percentage_all(X, y):
    """
    Performs training and testing with varying train-test splits for multiple methods and plots boxplot of accuracies.

    Args:
    - X (numpy array): Feature matrix.
    - y (numpy array): Target labels.

    Returns:
    - accuracies_total (list of lists): List of accuracies for each method and train-test split.
    """
    
    M = ['M1','M2','M3','M4']
    test_size_table = np.array([0.02,0.003,0.005,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    accuracies_total = []
    
    for m in M:
        accuracies_total.append(train_test_percentage(X, y, m))

    # Box plot
    plt.boxplot(accuracies_total, labels=M)
    plt.title('Box plot of Accuracies for Different Methods')
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.show()
    return accuracies_total
