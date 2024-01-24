# -*- coding: utf-8 -*-

from helpers import separation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def RF(X, y, Method = 'M1', cross_validation = False, cv=5, random_state=42):
    """Train a Random Forest classifier and evaluate its accuracy.

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - n_estimators: Number of trees in the forest (default is 100)
    - random_state: Seed for random number generation (default is 42)

    Returns:
    - y_true: True labels of the test data
    - y_pred: Predicted labels of the test data
    """

    X_train, y_train, X_test, y_test = separation(X, y, Method)
    print("Separation has been done!")

    if(cross_validation):
        print("Cross Validation chosen")

        # Estimators to test the best :
        n_estimators_list = [50, 100, 200]
        # Create a Random Forest classifier
        print("Classifier Defined")
        rf_classifier = RandomForestClassifier(random_state=random_state)

        # Create GridSearchCV with the specified parameter grid
        param_grid = {'n_estimators': n_estimators_list}
        print("Grid search begins")
        grid_search = GridSearchCV(rf_classifier, param_grid, cv=cv, scoring='accuracy')
        print("Grid search done")

        # Fit the model to the data, performing grid search with cross-validation
        print("Fit Grid search started")
        grid_search.fit(X_train, y_train)
        print("Fit Grid search done")

        # Get the best estimator and its parameters
        print("Get best estimator")
        best_estimator = grid_search.best_estimator_
        print("Found best estimator!")
        print("Get best parameters")

        best_estimator_params = grid_search.best_params_
        print("Found best parameters!")

        print("Best Estimator Parameters:")
        print(best_estimator)
        print("\nBest Estimator Parameters (formatted):", best_estimator_params)

        # Make predictions on the test data using the best estimator
        print("Now predict!")
        y_pred = best_estimator.predict(X_test)

        # Return true labels and predicted labels
        return y_test, y_pred
    
    else:
        n_estimators = 100
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

        # Train the classifier on the training data
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_classifier.predict(X_test)

        # Return true labels and predicted labels
        return y_test, y_pred
