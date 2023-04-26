import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

# loading the pre processed csv file
df = pd.read_csv("Electrical_H_DB")

# Create dictionary to map categories to integer values
country_dict = {country: i for i, country in enumerate(df['countryname'].unique())} 

# Replace categories with integer values
df['countryname'] = df['countryname'].replace(country_dict)

#encoding target variable, total electricity consumption rate as low or high

# Define a threshold value to distinguish between low and high values
threshold = 50.0

# Create a new column to encode the low/high values

df['elecrate_total'] = pd.cut(df['elecrate_total'], 
                                                    bins=[-float('inf'), threshold, float('inf')], 
                                                    labels=['low', 'high'])




from sklearn.preprocessing import StandardScaler

# Apply standard scaling to the column
scaler = StandardScaler()
df['scaled_elecrate_rural'] = scaler.fit_transform(df[['elecrate_rural']])

print(df)
# feature selection: selecting the features and target variable
X = df.drop(['countrycode','year','elecrate_urban','elecrate_rural','elecrate_total'],axis= 1)
y= df['elecrate_total']

# splitting the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# best parameters after hyperparameter tuning with GridSearchCv
# Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Best score: 0.888

#develop improved model
improved_rf = RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_leaf=1,min_samples_split=5)
improved_rf.fit(X_train,y_train)

import pickle

pickle.dump(improved_rf, open("ESPmodel.pkl", "wb"))

predictions = improved_rf.predict(X_test)

print(confusion_matrix(y_test,predictions))