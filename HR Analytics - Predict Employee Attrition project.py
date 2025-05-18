import warnings
warnings.filterwarnings('ignore')



# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load dataset
# Make sure the CSV file is in the same directory as your script or provide full path
df = pd.read_csv(r'C:\Users\sarik\OneDrive\INTERN\WA_Fn-UseC_-HR-Employee-Attrition.csv') 

# 3. Preprocess data

# Drop columns that are mostly constant or redundant
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)

# Encode categorical columns
label_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']  # Already encoded as 1/0 by LabelEncoder

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Model evaluation

# Accuracy score
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


