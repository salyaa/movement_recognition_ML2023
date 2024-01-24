
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd 

###########################################################################################################

# Function to compute the percentage of missing values in a Data frame
def percentage_nan_by_row(df):
    """
    Calculate the percentage of NaN values for each row in a DataFrame.

    Args:
    - df (pandas DataFrame): Input DataFrame.

    Returns:
    - pandas Series: Series containing the percentage of NaN values for each row.
    """
    return df.isnull().mean(axis=1)*100

# Data cleaning function
def clean_data(data: pd.DataFrame):
    """Cleaning of the data set and creation of our matrix X and our labels y

    Args:
        data (pd.DataFrame): original data set

    Returns:
        X (pd.DataFrame): cleaned data set without the label column
        y (np.array): labels
        y_encoded (np.array): encoded_labels
    """
    # Rows with NaN values in our data set
    rows_with_nan = data[data.isnull().any(axis=1)]
    
    # Percentage of positional missing values
    percentage_nan_by_rows = percentage_nan_by_row(rows_with_nan.drop(['Participant', 'Set', 'Camera','Exercise', 'time(s)'], axis='columns'))
    print(f"The initial data contains {percentage_nan_by_rows[percentage_nan_by_rows==100].shape[0]} of NaN values")
    print(f"There is {percentage_nan_by_rows[percentage_nan_by_rows==100].shape[0]} rows with 100 percents of NaN positional values.")
    print("Hence all rows can be removed from our data set.")
    # We then remove all the NaNs values of our data
    data_cleaned = data.dropna().copy(deep=True)
    
    # Extract the label y: the column we want to predict
    y = data_cleaned['Exercise']
    print("Label y extracted!")

    # We use a label encoder to encode the column 'Exercise' that we want to predict
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print("Label y encoded!")

    # Create the data X
    X = data_cleaned.drop('Exercise', axis='columns').copy(deep=True)
    print('Matrix X created!')

    return X, y, y_encoded

###########################################################################################################

def submatrices_cameras(X: pd.DataFrame, y_encoded: np.array):
    """Creation of the sub-matrices for each cameras

    Args:
        X (pd.DataFrame): data
        y_encoded (np.array): encoded label that we want to predict

    Returns:
        tuple: Four pairs of sub-matrices and corresponding labels for Frontal_Top, Frontal_Low, Side_Top, and Side_Low.
    """
    # Frontal Top:
    X_ft = X[X['Camera'] == 'Frontal_Top'].drop('Camera', axis='columns').copy(deep=True)
    idx_ft = X['Camera'].isin(['Frontal_Top'])
    y_encoded_ft = y_encoded[idx_ft]
    # Frontal Low:
    X_fl = X[X['Camera'] == 'Frontal_Low'].drop('Camera', axis='columns').copy(deep=True)
    idx_fl = X['Camera'].isin(['Frontal_Low'])
    y_encoded_fl = y_encoded[idx_fl]
    # Side Top:
    X_st = X[X['Camera'] == 'Side_Top'].drop('Camera', axis='columns').copy(deep=True)
    idx_st = X['Camera'].isin(['Side_Top'])
    y_encoded_st = y_encoded[idx_st]
    # Side Low:
    X_sl = X[X['Camera'] == 'Side_Low'].drop('Camera', axis='columns').copy(deep=True)
    idx_sl = X['Camera'].isin(['Side_Low'])
    y_encoded_sl = y_encoded[idx_sl]
    
    return X_ft, y_encoded_ft, X_fl, y_encoded_fl, X_st, y_encoded_st, X_sl, y_encoded_sl

###########################################################################################################

def test_accuracy_f1_scores(y_true, y_pred, Method, model):
    accuracy_time = np.sum(y_true == y_pred) / len(y_true)
    f1_score_NN_macro_time = f1_score(y_true, y_pred, average='macro')
    f1_score_NN_micro_time = f1_score(y_true, y_pred, average='micro')

    print(f'Test Accuracy for method {Method} and model {model}: {accuracy_time * 100:.2f}%')
    print(f'F1 Score macro for method {Method} and model {model}: {f1_score_NN_macro_time:.4f}')
    print(f'F1 Score micro for method {Method} and model {model}: {f1_score_NN_micro_time:.4f}')

###########################################################################################################

def confusion_matrix_M(y, y_pred_pos, y_true_pos, y_pred_time, y_true_time, y_pred_camera, y_true_camera, y_pred_error, y_true_error):
    """Creation of confusion matrices for all the splitting methods

    Args:
        all the arguments are np.array with y the encoded labels for the whole matrix, the others are the true
        and predicted labels for each method respectively.

    Returns:
        None
    """
    y_preds = [y_pred_pos, y_pred_time, y_pred_camera, y_pred_error]
    y_trues = [y_true_pos, y_true_time, y_true_camera, y_true_error]
    titles = ['Confusion matrix for Positional values', 'Confusion matrix for time series of Participants','Confusion matrix for time series of camera values', 'Confusion matrix for time series of error values']
    colors = ['Blues', 'Greens', 'Oranges', 'Purples']
    
    # Loop over the sets of y_true and y_pred
    for i, (y_true, y_pred, title, color) in enumerate(zip(y_trues, y_preds, titles, colors)):
        # Calculate confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        
        # Plot the confusion matrix in the corresponding subplot
        sns.heatmap(cm, annot=True, cmap=color, xticklabels=y.unique(), yticklabels=y.unique(), fmt='.2%')
        plt.title(title)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.show()

###########################################################################################################

def train_test_separation_participant(X, y, data, test_size_=12):
    """We want to split the data set into a train and a test set, taking a certain number of participant in the test.
    
    Args:
        X (data frame): our data set
        y (array): labels
        test_size (int)

    Returns:
        X and y split into tran and test set
    """
    # Determine the participants for training and testing
    participants = data['Participant'].unique()
    participants_train, participants_test = train_test_split(participants, test_size=test_size_, random_state=42)
    # Separate features and labels for train and test sets
    X_train = X[data['Participant'].isin(participants_train)]
    y_train = y[data['Participant'].isin(participants_train)]
    X_test = X[data['Participant'].isin(participants_test)]
    y_test = y[data['Participant'].isin(participants_test)]
    
    return X_train, y_train, X_test, y_test

###########################################################################################################

def separation(X: pd.DataFrame, y_encoded: np.array, Method: str = 'M1'):
    """Split the data for the given method

    Args:
        X (pd.DataFrame): cleaned data set
        y_encoded (np.array): encoded labels
        Method (str, optional): _description_. Defaults to 'M1'.

    Returns:
        Train-test splitting of X and y
    """
    assert Method in ['M1', 'M2', 'M3', 'M4']
    
    if Method == "M1":
        print("Method 1")
        # Parameters found when first running cross validation
        hidden_size_best = 256
        learning_rate_best = 0.001
        batch_size_best = 32

        X = X.loc[:, 'left_ankle_x':]
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)
        X_test = X_test.values
        X_train = X_train.values
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    

    elif Method == "M2":
        print("Method 2")      
        X_time = X.loc[:, 'time(s)':]
        X_train_time, y_train_time, X_test_time, y_test_time = train_test_separation_participant(X_time, y_encoded, X)

        # Same for the second set of variables
        X_test = X_test_time.values
        X_train = X_train_time.values
        y_train = np.array(y_train_time)
        y_test = np.array(y_test_time)
        
    elif Method == "M3":
        print("Method 3")
        X_camera = X.loc[:, 'Camera':]
        X_train_camera, X_test_camera, y_train, y_test = train_test_split(X_camera, y_encoded, test_size=0.5, stratify=X_camera['Camera'], random_state=42)

        X_train = X_train_camera.drop(columns=['Camera'])
        X_test = X_test_camera.drop(columns=['Camera'])
        X_train = X_train.values
        X_test = X_test.values
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
    elif Method == "M4":
        print("Method 4")
        X_error = X.loc[:,'Set':]
        X_error = X_error.drop(columns=['Camera'])
        X_train_error, X_test_error, y_train, y_test = train_test_split(X_error, y_encoded, test_size=0.5, stratify=X_error['Set'], random_state=42)
        X_train = X_train_error.drop(columns=['Set'])
        X_test = X_test_error.drop(columns=['Set'])
        X_train = X_train.values
        X_test = X_test.values
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

###########################################################################################################
