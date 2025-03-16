import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset with correct file paths
train_df = pd.read_csv("C:/Users/profa/OneDrive/Documents/infotech/train.csv")
test_df = pd.read_csv("C:/Users/profa/OneDrive/Documents/infotech/test.csv")
sample_submission = pd.read_csv("C:/Users/profa/OneDrive/Documents/infotech/sample_submission.csv")

# Display basic info
print("Train Data:", train_df.info())
print("Test Data:", test_df.info())

# Separate numeric and categorical columns (only from train dataset)
numeric_cols_train = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_train = train_df.select_dtypes(include=['object']).columns

# Fill missing values in train data
train_df[numeric_cols_train] = train_df[numeric_cols_train].fillna(train_df[numeric_cols_train].median())
train_df[categorical_cols_train] = train_df[categorical_cols_train].fillna(train_df[categorical_cols_train].mode().iloc[0])

# Now, process test dataset
numeric_cols_test = test_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_test = test_df.select_dtypes(include=['object']).columns

test_df[numeric_cols_test] = test_df[numeric_cols_test].fillna(test_df[numeric_cols_test].median())
test_df[categorical_cols_test] = test_df[categorical_cols_test].fillna(test_df[categorical_cols_test].mode().iloc[0])

# Convert categorical columns to numerical using One-Hot Encoding
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align test columns with train columns (exclude 'SalePrice' from train)
test_df = test_df.reindex(columns=train_df.columns.drop('SalePrice', errors='ignore'), fill_value=0)

# Define Features (X) and Target (y)
X = train_df.drop(columns=['SalePrice'])  # Ensure 'SalePrice' is the correct target column
y = train_df['SalePrice']

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Model Evaluation
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"📊 Model Performance:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Predict on Test Data
test_predictions = model.predict(test_df)

# Prepare Submission File
submission = pd.DataFrame({'Id': sample_submission['Id'], 'SalePrice': test_predictions})  
submission.to_csv("C:/Users/profa/OneDrive/Documents/infotech/submission.csv", index=False)

print("✅ Submission file created: submission.csv")
