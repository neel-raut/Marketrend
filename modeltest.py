# Import necessary libraries
import numpy as np
from datetime import datetime
from stock_indicators import Quote
from stock_indicators import indicators
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data_url = "https://raw.githubusercontent.com/sunandrew03/tradingmodel/main/%5Espx_d.csv"
# Load the dataset using pandas
df = pd.read_csv(data_url)
# RAW dataframe columns: ['Open', 'High', 'Low', 'Close', 'Volume']

# Create a target variable based on the price change
# This says '1 if tommorow's close will be higher than today's close, 0 otherwise'
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

#Create new predictive variables as features

#To avoid duplicate data -
#One Momentum, Trend, and Volume Indicator

quotes_list = []
for index, row in df.iterrows():
    datetime_object = datetime.strptime(row['Date'], '%Y-%m-%d')
    quotes_list.append(Quote(datetime_object, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

rsi_results = indicators.get_rsi(quotes_list)
rsi_values = []
for r in rsi_results:
    rsi_values.append(r.rsi)
    
macd_results = indicators.get_macd(quotes_list)
macd_values = []
for m in macd_results:
    macd_values.append(m.macd)
    
cmf_results = indicators.get_cmf(quotes_list)
cmf_values = []
for c in cmf_results:
    cmf_values.append(c.cmf)
    
df['RSI'] = rsi_values
df['MACD'] = macd_values
df['CMF'] = cmf_values


# Drop the last row(s) since it will have a NaN in the target column
df = df.dropna()

# Features (X) and target variable (y)
X = df[['RSI', 'MACD', 'CMF']]
y = df['target']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(fit_intercept=True, max_iter = 1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

print("DEBUG:")
print(df['target'].value_counts())