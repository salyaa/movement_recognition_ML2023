import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

def calculate_angles(df):

    '''Calculate 6 angles for each row of the provided dataset based on sensors coordinates.

    Args:
        df, dataset with sensors coordinates.

    Returns:
        df, dataset with 6 additional columns, one for each calculated angle.
    '''

    # Define the six points on each side : ankle, knee, hip, shoulder, elbow and wrist
    # LEFT
    left_ankle = df[['left_ankle_x', 'left_ankle_y', 'left_ankle_z']]
    left_knee = df[['left_knee_x', 'left_knee_y', 'left_knee_z']].values
    left_hip = df[['left_hip_x', 'left_hip_y', 'left_hip_z']].values
    left_shoulder = df[['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z']].values
    left_elbow = df[['left_elbow_x','left_elbow_y','left_elbow_z']].values
    left_wrist = df[['left_wrist_x','left_wrist_y','left_wrist_z']]
    # RIGHT
    right_ankle = df[['right_ankle_x', 'right_ankle_y', 'right_ankle_z']]
    right_knee = df[['right_knee_x', 'right_knee_y', 'right_knee_z']].values
    right_hip = df[['right_hip_x', 'right_hip_y', 'right_hip_z']].values
    right_shoulder = df[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z']].values
    right_elbow = df[['right_elbow_x','right_elbow_y','right_elbow_z']].values
    right_wrist = df[['right_wrist_x','right_wrist_y','right_wrist_z']]


    # Calculate vectors between points
    # LEFT 
    vec_knee_hip_l = left_hip - left_knee
    vec_shoulder_hip_l = left_hip - left_shoulder
    vec_ankle_knee_l = left_knee - left_ankle
    vec_shoulder_elbow_l = left_elbow - left_shoulder
    vec_elbow_wrist_l = left_wrist - left_elbow
    # RIGHT 
    vec_knee_hip_r = right_hip - right_knee
    vec_shoulder_hip_r = right_hip - right_shoulder
    vec_ankle_knee_r = right_knee - right_ankle
    vec_shoulder_elbow_r = right_elbow - right_shoulder
    vec_elbow_wrist_r = right_wrist - right_elbow


    # Calculate dot product and magnitudes for angle 1 (body bending) 
    # LEFT 
    dot_product_l = np.sum(vec_knee_hip_l * vec_shoulder_hip_l, axis=1)
    magnitude_knee_hip_l = np.linalg.norm(vec_knee_hip_l, axis=1)
    magnitude_shoulder_hip_l = np.linalg.norm(vec_shoulder_hip_l, axis=1)
    # RIGHT 
    dot_product_r = np.sum(vec_knee_hip_r * vec_shoulder_hip_r, axis=1)
    magnitude_knee_hip_r = np.linalg.norm(vec_knee_hip_r, axis=1)
    magnitude_shoulder_hip_r = np.linalg.norm(vec_shoulder_hip_r, axis=1)


    # Calculate the cosine of the angle
    cosine_angle_l = np.clip(dot_product_l / (magnitude_knee_hip_l * magnitude_shoulder_hip_l), -1.0, 1.0) # LEFT 
    cosine_angle_r = np.clip(dot_product_r / (magnitude_knee_hip_r * magnitude_shoulder_hip_r), -1.0, 1.0) # RIGHT 


    # Calculate the angle in radians and convert to degrees
    # LEFT 
    angle_radians_l = np.arccos(cosine_angle_l)
    angle_degrees_l = np.degrees(angle_radians_l)
    # RIGHT 
    angle_radians_r = np.arccos(cosine_angle_r)
    angle_degrees_r = np.degrees(angle_radians_r)

    # Add a new column to the DataFrame with the calculated angles
    df['angle_left_knee_hip_shoulder'] = angle_degrees_l # LEFT 
    df['angle_right_knee_hip_shoulder'] = angle_degrees_r # RIGHT 


    # Calculate vectors between points for angle 2 (leg bending)
    # LEFT 
    dot_product_ankle_knee_l = np.sum(vec_ankle_knee_l * vec_knee_hip_l, axis=1)
    magnitude_ankle_knee_l = np.linalg.norm(vec_ankle_knee_l, axis=1)
    # RIGHT 
    dot_product_ankle_knee_r = np.sum(vec_ankle_knee_r * vec_knee_hip_r, axis=1)
    magnitude_ankle_knee_r = np.linalg.norm(vec_ankle_knee_r, axis=1)

    # Calculate the cosine of the angle
    cosine_angle_ankle_knee_l = np.clip(dot_product_ankle_knee_l / (magnitude_ankle_knee_l * magnitude_knee_hip_l), -1.0, 1.0) # LEFT 
    cosine_angle_ankle_knee_r = np.clip(dot_product_ankle_knee_r / (magnitude_ankle_knee_r * magnitude_knee_hip_r), -1.0, 1.0) # RIGHT 


    # Calculate the angle in radians and convert to degrees
    # LEFT 
    angle_radians_ankle_knee_l = np.arccos(cosine_angle_ankle_knee_l)
    angle_degrees_ankle_knee_l = np.degrees(angle_radians_ankle_knee_l)
    # RIGHT 
    angle_radians_ankle_knee_r = np.arccos(cosine_angle_ankle_knee_r)
    angle_degrees_ankle_knee_r = np.degrees(angle_radians_ankle_knee_r)


    # Add a new column to the DataFrame with the calculated angles
    df['angle_left_ankle_knee_hip'] = angle_degrees_ankle_knee_l # LEFT 
    df['angle_right_ankle_knee_hip'] = angle_degrees_ankle_knee_r # RIGHT 


    # Calculate dot product and magnitudes for angle 3 (arm bending) 
    # LEFT 
    dot_product_elbow_wrist_l = np.sum(vec_shoulder_elbow_l * vec_elbow_wrist_l, axis=1)
    magnitude_shoulder_elbow_l = np.linalg.norm(vec_shoulder_elbow_l, axis=1)
    magnitude_elbow_wrist_l = np.linalg.norm(vec_elbow_wrist_l, axis=1)
    # RIGHT 
    dot_product_elbow_wrist_r = np.sum(vec_shoulder_elbow_r * vec_elbow_wrist_r, axis=1)
    magnitude_shoulder_elbow_r = np.linalg.norm(vec_shoulder_elbow_r, axis=1)
    magnitude_elbow_wrist_r = np.linalg.norm(vec_elbow_wrist_r, axis=1)


    # Calculate the cosine of the angle
    cosine_angle_elbow_wrist_l = np.clip(dot_product_elbow_wrist_l / (magnitude_shoulder_elbow_l * magnitude_elbow_wrist_l), -1.0, 1.0) # LEFT 
    cosine_angle_elbow_wrist_r = np.clip(dot_product_elbow_wrist_r / (magnitude_shoulder_elbow_r * magnitude_elbow_wrist_r), -1.0, 1.0) # RIGHT 


    # Calculate the angle in radians and convert to degrees
    # LEFT 
    angle_radians_elbow_wrist_l = np.arccos(cosine_angle_elbow_wrist_l)
    angle_degrees_elbow_wrist_l = np.degrees(angle_radians_elbow_wrist_l)
    # RIGHT 
    angle_radians_elbow_wrist_r = np.arccos(cosine_angle_elbow_wrist_r)
    angle_degrees_elbow_wrist_r = np.degrees(angle_radians_elbow_wrist_r)

    # Add a new column to the DataFrame with the calculated angles
    df['angle_left_shoulder_elbow_wrist'] = angle_degrees_elbow_wrist_l # LEFT 
    df['angle_right_shoulder_elbow_wrist'] = angle_degrees_elbow_wrist_r # RIGHT 


    return df



def train_and_evaluate_model(df, angle_columns):

    '''Train and evaluate 3 different classifiers (RF, SVC, GB) on dataset with only angles.

    Args:
        df : DataFrame, dataset with 6 angles as columns
        angle_columns : list of str, name of these 6 columns.

    Returns:
        predictions, contains y_pred for each classifier 
        y_test, the labels of testing set.
    '''

    # Avoid SettingWithCopyWarning
    df['time(s)'] = pd.to_datetime(df['time(s)'])
    df_reduced = df.set_index('time(s)')

    # Encode 'Exercise' using label encoding
    label_encoder = LabelEncoder()
    df_reduced['Exercise'] = label_encoder.fit_transform(df_reduced['Exercise'])

    # Feature engineering - calculate relative angles
    for i in range(1, len(angle_columns)):
        df_reduced[f'relative_angle_{i}'] = df_reduced[angle_columns[i-1]] - df_reduced[angle_columns[i]]

    # 'Exercise' is the target variable
    X = df_reduced.drop(['Exercise'], axis=1)
    y = df_reduced['Exercise']

    # Split the data into training and testing sets according to splitting method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, random_state=42)

    # Define classifiers and scalers
    classifiers = [RandomForestClassifier(), SVC(), GradientBoostingClassifier()]
    scalers = [StandardScaler() for _ in classifiers]

    predictions = {}

    for clf, scaler in zip(classifiers, scalers):
        # Train
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = clf.predict(X_test_scaled)

        # Evaluate accuracy and F1 score
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        f1_score_macro = f1_score(y_test, y_pred, average='macro')
        f1_score_micro = f1_score(y_test, y_pred, average='micro')

        print(f'{clf.__class__.__name__} Test Accuracy: {accuracy * 100:.2f}%')
        print(f'{clf.__class__.__name__} F1 Score macro: {f1_score_macro:.4f}')
        print(f'{clf.__class__.__name__} F1 Score micro: {f1_score_micro:.4f}')
        print('-----')

        # Save predictions
        predictions[clf.__class__.__name__] = y_pred


    return predictions, y_test

def max_min_angle_per_participant(target_data):

    '''Calculate max and min values of each participant over all the exercise and add these max and min values into columns

    Args:
        target_data, dataset with angular measurements for active part of the exercise.

    Returns:
        result_df, dataset with max and min angle for each participant during each exercise.
    '''

    # We only keep their best camera for each exercise
    exercise_camera_mapping = {
        'Abduction': 'Frontal_Top',
        'Bird': 'Frontal_Low',
        'Bridge': 'Frontal_Top',
        'Knee': 'Frontal_Low',
        'Shoulder': 'Side_Top',
        'Squat': 'Side_Top',
        'Stretch': 'Side_Low'
    }

    # Initialize an empty list to store results for each group
    result_data = []

    # Columns for which min, max, and mean need to be calculated
    columns_to_process = ['angle_left_knee_hip_shoulder', 'angle_right_knee_hip_shoulder',	
                        'angle_left_ankle_knee_hip', 'angle_right_ankle_knee_hip',
                        'angle_left_shoulder_elbow_wrist', 'angle_right_shoulder_elbow_wrist']

    # Iterate through each group
    for _, group_df in target_data.groupby(['Exercise', 'Participant']):
        # Check if the exercise is in the mapping
        if group_df['Exercise'].iloc[0] in exercise_camera_mapping:
            # Get the desired camera for the exercise
            desired_camera = exercise_camera_mapping[group_df['Exercise'].iloc[0]]

            # Filter rows where 'Camera' is the desired camera
            exercise_camera_data = group_df[group_df['Camera'] == desired_camera]

            if not exercise_camera_data.empty:
                # Initialize a dictionary to store results for the current group
                result_dict = {
                    'Exercise': exercise_camera_data['Exercise'].iloc[0],
                    'Participant': exercise_camera_data['Participant'].iloc[0],
                    'Camera': desired_camera
                }

                # Calculate min, max, and mean for each specified column
                for column in columns_to_process:
                    max_value = exercise_camera_data[column].max()
                    min_value = exercise_camera_data[column].min()

                    # Add the results to the dictionary
                    result_dict[f'Max_{column}'] = max_value
                    result_dict[f'Min_{column}'] = min_value

                # Append the dictionary to the list
                result_data.append(result_dict)

    # Create a new DataFrame from the list of results
    result_df = pd.DataFrame(result_data)

    return result_df

def max_max_min_min_angles(result_df):

    '''Calculate max and min values over all participants for a same exercise

    Args:
        result_df, dataset with max and min angle for each participant during each exercise.

    Returns:
        result_df, a table with max and min values for each exercise.
    '''

    # Initialize the dictionary to store results for each angle
    angle_results = {'Exercise': [], 'Max left body bending': [], 'Min left body bending': [],
                    'Max left leg bending': [], 'Min left leg bending': [],
                    'Max left arm bending': [], 'Min left arm bending': [],
                    'Max right body bending': [], 'Min right body bending': [],
                    'Max right leg bending': [], 'Min right leg bending': [],
                    'Max right arm bending': [], 'Min right arm bending': []
                    }

    # Columns for which min and max need to be calculated
    columns_to_process = ['angle_left_knee_hip_shoulder', 'angle_left_ankle_knee_hip', 'angle_left_shoulder_elbow_wrist',
                        'angle_left_knee_hip_shoulder', 'angle_left_ankle_knee_hip', 'angle_left_shoulder_elbow_wrist']

    # Iterate through each exercise
    for exercise in result_df['Exercise'].unique():
        # Filter rows for the current exercise
        exercise_data = result_df[result_df['Exercise'] == exercise]

        # Calculate max and min values for each angle across all participants
        max_knee_hip_shoulder_l = exercise_data['Max_angle_left_knee_hip_shoulder'].max()
        min_knee_hip_shoulder_l = exercise_data['Min_angle_left_knee_hip_shoulder'].min()
        max_ankle_knee_hip_l = exercise_data['Max_angle_left_ankle_knee_hip'].max()
        min_ankle_knee_hip_l = exercise_data['Min_angle_left_ankle_knee_hip'].min()
        max_shoulder_elbow_wrist_l = exercise_data['Max_angle_left_shoulder_elbow_wrist'].max()
        min_shoulder_elbow_wrist_l = exercise_data['Min_angle_left_shoulder_elbow_wrist'].min()


        max_knee_hip_shoulder_r = exercise_data['Max_angle_right_knee_hip_shoulder'].max()
        min_knee_hip_shoulder_r = exercise_data['Min_angle_right_knee_hip_shoulder'].min()
        max_ankle_knee_hip_r = exercise_data['Max_angle_right_ankle_knee_hip'].max()
        min_ankle_knee_hip_r = exercise_data['Min_angle_right_ankle_knee_hip'].min()
        max_shoulder_elbow_wrist_r = exercise_data['Max_angle_right_shoulder_elbow_wrist'].max()
        min_shoulder_elbow_wrist_r = exercise_data['Min_angle_right_shoulder_elbow_wrist'].min()

        # Append values to the dictionary
        angle_results['Exercise'].append(exercise)
        angle_results['Max left body bending'].append(max_knee_hip_shoulder_l)
        angle_results['Min left body bending'].append(min_knee_hip_shoulder_l)
        angle_results['Max left leg bending'].append(max_ankle_knee_hip_l)
        angle_results['Min left leg bending'].append(min_ankle_knee_hip_l)
        angle_results['Max left arm bending'].append(max_shoulder_elbow_wrist_l)
        angle_results['Min left arm bending'].append(min_shoulder_elbow_wrist_l)

        angle_results['Max right body bending'].append(max_knee_hip_shoulder_r)
        angle_results['Min right body bending'].append(min_knee_hip_shoulder_r)
        angle_results['Max right leg bending'].append(max_ankle_knee_hip_r)
        angle_results['Min right leg bending'].append(min_ankle_knee_hip_r)
        angle_results['Max right arm bending'].append(max_shoulder_elbow_wrist_r)
        angle_results['Min right arm bending'].append(min_shoulder_elbow_wrist_r)


    # Create a DataFrame from the dictionary
    angle_results_df = pd.DataFrame(angle_results)

    # Transpose the DataFrame
    angle_results_df_transposed = angle_results_df.set_index('Exercise').transpose()

    return angle_results_df_transposed

def predict_ex_range(row, table_ranges):

    '''Predict the exercise based on the number of angles in the angular ranfes of table_ranges

    Args:
        row, row of dataset to extract angular values from
        table_ranges, table containing angular ranges to compare and predict the exercise.

    Returns:
        exercise_pred, predicted exercise based on the max number of angles in ranges for the 7 exercises.
    '''
    
    exercises = ['Abduction', 'Bird', 'Bridge', 'Knee', 'Shoulder', 'Squat', 'Stretch']
    probabilities = []
    for i in range(7):

        # Check if each angle is within the specified range
        angle1_l_in_range = int(table_ranges.iloc[1, i] <= row['angle_left_knee_hip_shoulder'] <= table_ranges.iloc[0, i])
        angle2_l_in_range = int(table_ranges.iloc[3, i] <= row['angle_left_ankle_knee_hip'] <= table_ranges.iloc[2, i])
        angle3_l_in_range = int(table_ranges.iloc[5, i] <= row['angle_left_shoulder_elbow_wrist'] <= table_ranges.iloc[4, i])
        angle1_r_in_range = int(table_ranges.iloc[7, i] <= row['angle_right_knee_hip_shoulder'] <= table_ranges.iloc[6, i])
        angle2_r_in_range = int(table_ranges.iloc[9, i] <= row['angle_right_ankle_knee_hip'] <= table_ranges.iloc[8, i])
        angle3_r_in_range = int(table_ranges.iloc[11, i] <= row['angle_right_shoulder_elbow_wrist'] <= table_ranges.iloc[10, i])

        # Calculate the probability based on the number of angles in range
        probability = angle1_l_in_range + angle1_r_in_range + angle2_l_in_range + angle2_r_in_range + angle3_l_in_range + angle3_r_in_range

        # Append the probability to the list
        probabilities.append(probability)

    exercise_pred = exercises[np.argmax(probabilities)]

    return exercise_pred
