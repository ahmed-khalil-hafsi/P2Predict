from sklearn.model_selection import train_test_split



def prepare_data(data,selected_columns,target_column):
    # Separate the features and the target variable
    X = data[selected_columns]
    y = data[target_column]

    # Split the data into training (80%) and test sets (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Identify categorical and numerical columns - for now this is automated. TODO: Let user select which columns are categorical and which are numerical
    numerical_cols, categorical_cols = Get_Column_Types(X)

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols

def select_features(data, columns):
    return data[columns]

def Get_Column_Types(X):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    return numerical_cols,categorical_cols