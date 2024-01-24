# -*- coding: utf-8 -*-

import numpy as np
from helpers import separation
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


#############################################################################################################

def train_test_clustering_svm(X: np.array, X_names: list[str], y: np.array, method: str = 'M1', cross_validation: bool = True, show: bool = False):
    """Classification using SVM of our matrix X using a certain separation method

    Args:
        X (np.array): 2D array that wwe want to use to predict our label
        y (np.array): encoded labels
        method (str, optional): separation method. Defaults to 'M1'.
        cross_validation (bool, optional): determine if we do cross-validation to find the best hyper-parameters. Defaults to True.
        show (bool, optional): determine if we do the visualization and dimension reduction. Defaults to False.

    Returns:
        test_predictions (np.array): predicted label for our test set
        accuracy (float): accuracy of our prediction
    """
    
    X_train, y_train_encoded, X_test, y_test_encoded = separation(X, y, method)
    print(f"Separation has been done using method {method}!")
    
    # Train a classifier on the labeled training set: 
    # We first find the best hyperparameters:
    
    # Subsample a fraction of the data for hyperparameter tuning
    if cross_validation:
        N = X_train.shape[0]
        # To modify if wanted
        num_samples = int(N/5)
        subsample_indices = np.random.choice(len(X_train), size=num_samples, replace=False)
        # Use the subsample for hyperparameter tuning
        #subsample_data = X_train.iloc[subsample_indices]
        subsample_data = X_train[subsample_indices]
        subsample_labels = y_train_encoded[subsample_indices]

        # Perform hyperparameter tuning on the subsample:
        # Define the parameter grid for grid search
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        svm_classifier = SVC()
        print("Grid search begins")
        grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
        print("Grid search done")
        
        print("Fit Grid search started")
        grid_search.fit(subsample_data, subsample_labels)
        print("Fit Grid search done")

        # Get the best hyperparameters
        print("Get best parameters")
        best_params = grid_search.best_params_
        print("Found best parameters!")
        print("Best Estimator Parameters (formatted):", best_params)

        # Train the final SVM model with the best hyperparameters on the entire subsample
        final_svm_classifier = SVC(**best_params)
        final_svm_classifier.fit(subsample_data, subsample_labels)
    else:
        final_svm_classifier = SVC(C=0.1, kernel='linear')
        final_svm_classifier.fit(X_train, y_train_encoded)
        
    # Predict on the test set
    test_predictions = final_svm_classifier.predict(X_test)
    
    # Evaluate the accuracy
    accuracy = accuracy_score(y_test_encoded, test_predictions)
    print(f"Accuracy: {accuracy*100:.4f}%")
    
    # Visualization:
    if show:
        X_train_reduced = PCA(n_components=2).fit(X_train).transform(X_train)
        print(f"The features of the first sample for the matrix {X_names[0]} are: %s" % X_train_reduced[0])
        X_test_reduced = PCA(n_components=2).fit(X_test).transform(X_test)
        print(f"The features of the first sample for the matrix {X_names[1]} are: %s" % X_test_reduced[0])
    
        _, axs = plt.subplots(1, 2, figsize=(20,5), sharex=True, sharey=True)
    
        axs[0].scatter(X_test_reduced[:,0], X_test_reduced[:,1], c=test_predictions, alpha=0.6)
        axs[0].set_title(f'Clustering of the matrix {X_names[1]} \n with the predicted labels')
    
        axs[1].scatter(X_test_reduced[:,0], X_test_reduced[:,1], c=y_test_encoded, alpha=0.6)
        axs[1].set_title(f'Clustering of the matrix {X_names[1]} \n with the true labels')
    
        plt.show()
    
    return test_predictions, accuracy

#############################################################################################################

def train_test_clustering_rf(X: np.array, X_names: list[str], y: np.array, method: str = 'M1', cross_validation: bool = True, show: bool = False):
    """Classification using Random Forest of our matrix X using a certain separation method

    Args:
        X (np.array): 2D array that wwe want to use to predict our label
        X_names (list[str]): list of the matrices names (for the train and test matrices, in this order)
        y (np.array): encoded labels
        method (str, optional): separation method. Defaults to 'M1'.
        cross_validation (bool, optional): determine if we do cross-validation to find the best hyper-parameters. Defaults to True.
        show (bool, optional): determine if we do the visualization and dimension reduction. Defaults to False.

    Returns:
        test_predictions (np.array): predicted label for our test set
        accuracy (float): accuracy of our prediction
    """
    X_train, y_train_encoded, X_test, y_test_encoded = separation(X, y, method)
    print(f"Separation has been done using method {method}!")
    
    # Train a classifier on the labeled training set: 
    if cross_validation:
        # We first find the best hyperparameters:
        # Subsample a fraction of the data for hyperparameter tuning
        N = X_train.shape[0]
        # To modify if wanted
        num_samples = int(N/5)
        subsample_indices = np.random.choice(len(X_train), size=num_samples, replace=False)
        # Use the subsample for hyperparameter tuning
        subsample_data = X_train[subsample_indices]
        subsample_labels = y_train_encoded[subsample_indices]

        # Perform hyperparameter tuning on the subsample:
        # Define the parameter grid for grid search
        param_grid = {'n_estimators': [1, 10, 100], 'random_state': [1, 42]}

        rf_classifier = RandomForestClassifier()
        print("Grid search begins")
        grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
        print("Grid search done")
        print("Fit Grid search started")
        grid_search.fit(subsample_data, subsample_labels)
        print("Fit Grid search done")

        # Get the best hyperparameters
        print("Get best parameters")
        best_params = grid_search.best_params_
        print("Found best parameters!")
        print("Best Estimator Parameters (formatted):", best_params)

        # Train the final Random Forest model with the best hyperparameters on the entire subsample
        final_rf_classifier = RandomForestClassifier(**best_params)
        final_rf_classifier.fit(subsample_data, subsample_labels)
    else:
        final_rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        final_rf_classifier.fit(X_train, y_train_encoded)
        
    # Predict on the test set
    test_predictions = final_rf_classifier.predict(X_test)
    
    # Evaluate the accuracy
    accuracy = accuracy_score(y_test_encoded, test_predictions)
    print(f"Accuracy: {accuracy*100:.4f}%")
    
    # Visualization:
    if show:
        X_train_reduced = PCA(n_components=2).fit(X_train).transform(X_train)
        print(f"The features of the first sample for the matrix {X_names[0]} are: %s" % X_train_reduced[0])
        X_test_reduced = PCA(n_components=2).fit(X_test).transform(X_test)
        print(f"The features of the first sample for the matrix {X_names[1]} are: %s" % X_test_reduced[0])
    
        _, axs = plt.subplots(1, 2, figsize=(20,5), sharex=True, sharey=True)
    
        axs[0].scatter(X_test_reduced[:,0], X_test_reduced[:,1], c=test_predictions, alpha=0.6)
        axs[0].set_title(f'Clustering of the matrix {X_names[1]} \n with the predicted labels')
    
        axs[1].scatter(X_test_reduced[:,0], X_test_reduced[:,1], c=y_test_encoded, alpha=0.6)
        axs[1].set_title(f'Clustering of the matrix {X_names[1]} \n with the true labels')
    
        plt.show()  
    
    return test_predictions, accuracy

#############################################################################################################
