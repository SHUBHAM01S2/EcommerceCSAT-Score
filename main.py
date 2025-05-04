import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

# Read the dataset
df = pd.read_csv('eCommerce_Customer_support_data.csv')

# Remove columns with more than 70% missing values
df.drop(['Customer_City', 'Product_category', 'Item_price', 'connected_handling_time', 'order_date_time', 'Customer Remarks'], axis=1, inplace=True)
df.drop(['Order_id', 'Unique id'], axis=1, inplace=True)

# Convert datetime columns to datetime format
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], format='%d/%m/%Y %H:%M')
df['issue_responded'] = pd.to_datetime(df['issue_responded'], format='%d/%m/%Y %H:%M')
df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'], format='%d-%b-%y')

# Extract datetime features (hour, day, month, weekday)
for col in ['Issue_reported at', 'issue_responded', 'Survey_response_Date']:
    df[col + '_hour'] = df[col].dt.hour
    df[col + '_day'] = df[col].dt.day
    df[col + '_month'] = df[col].dt.month
    df[col + '_weekday'] = df[col].dt.weekday

# Drop original datetime columns
df.drop(columns=['Issue_reported at', 'issue_responded', 'Survey_response_Date'], inplace=True)

# Label Encoding for categorical columns (for low-cardinality features)
le = LabelEncoder()
df['Agent Shift'] = le.fit_transform(df['Agent Shift'])
df['Tenure Bucket'] = le.fit_transform(df['Tenure Bucket'])
df['channel_name'] = le.fit_transform(df['channel_name'])
df['category'] = le.fit_transform(df['category'])

# Target encoding for high-cardinality categorical columns (Agent_name, Supervisor, Sub-category)
encoder = TargetEncoder()
df['Agent_name'] = encoder.fit_transform(df['Agent_name'], df['CSAT Score'])
df['Supervisor'] = encoder.fit_transform(df['Supervisor'], df['CSAT Score'])
df['Sub-category'] = encoder.fit_transform(df['Sub-category'], df['CSAT Score'])

# Handle missing values (imputation) if any remaining
imputer = SimpleImputer(strategy='mean')  # For numerical columns


# Split into features (X) and target (y)
X = df.drop('CSAT Score', axis=1)
y = df['CSAT Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 6 classes (since you are predicting CSAT score classes)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Display classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plotting training history (optional)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
