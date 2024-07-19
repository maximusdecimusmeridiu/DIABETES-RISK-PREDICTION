import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from a CSV file
data = pd.read_csv('diabetes_data_upload.csv')

# Encode binary categorical columns
binary_categorical_cols = [
    'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
    'Irritability', 'delayed healing', 'partial paresis', 
    'muscle stiffness', 'Alopecia', 'Obesity'
]
label_encoder = LabelEncoder()
for col in binary_categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Encode 'Gender' using one-hot encoding
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Split the data into features (X) and target (y)
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = decision_tree.predict(X_test)

# Evaluate the Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
print(classification_report(y_test, y_pred_dt))

# Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf))
