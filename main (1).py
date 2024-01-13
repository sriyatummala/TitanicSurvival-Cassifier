import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the training dataset
    train_df = pd.read_csv('train.csv')

    # Feature engineering: creating a new feature 'FamilySize'
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    # Drop unnecessary columns
    train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Encode categorical variables
    train_df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

    # Fill missing 'Age' values with KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    train_df['Age'] = imputer.fit_transform(train_df[['Age']])

    # Fill missing 'Embarked' values with the most common value
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

    # Separate features and target
    X = train_df.drop('Survived', axis=1)
    Y = train_df['Survived']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training data & Test data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

    # Model Training: Logistic Regression with Hyperparameter Tuning and Cross-Validation
    logistic_regression_params = {
        'C': [0.1, 0.5, 1, 5, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    # Create Logistic Regression model instance
    model = LogisticRegression()

    # Create GridSearchCV instance
    clf = GridSearchCV(model, logistic_regression_params, cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)

    # Best model found by GridSearchCV
    best_model = clf.best_estimator_

    # Model Evaluation
    # Making predictions on the training and test data
    Y_train_pred = best_model.predict(X_train)
    Y_test_pred = best_model.predict(X_test)

    # Evaluation metrics
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    class_report = classification_report(Y_test, Y_test_pred)

    # Displaying improved evaluation results
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)


