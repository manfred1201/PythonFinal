import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame()

# Applying the clean_sm function to the 'web1h' and 'marital' columns
ss['sm_li'] = clean_sm(s['web1h'])
ss['married'] = clean_sm(s['marital'])
ss['parent'] = clean_sm(s['par'])
ss['income'] = s['income'].apply(lambda x: x if x <= 9 else np.nan)
ss['education'] = s['educ2'].apply(lambda x: x if x <= 8 else np.nan)
ss['female'] = s['gender'].apply(lambda x: 1 if x == 2 else 0)
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)

# Dropping missing values
ss.dropna(inplace=True)

y = ss['sm_li']
X = ss.drop('sm_li', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

logi_model = LogisticRegression(class_weight='balanced')

logi_model.fit(X_train, y_train)