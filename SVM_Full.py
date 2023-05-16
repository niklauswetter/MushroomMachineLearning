import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn import svm

import warnings

warnings.filterwarnings("ignore")

df_csv = pd.read_csv('secondary_data.csv', sep=';')
features = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment',
            'gill-spacing', 'gill-color', 'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
            'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season']
output = ['class']

numerical = ['cap-diameter', 'stem-height', 'stem-width']
factors = [x for x in features if x not in numerical]

df_csv.head()
df_features = df_csv[features]
df_output = df_csv[output]
df_features[factors] = df_features[factors].apply(lambda x: pd.factorize(x)[0])

imp_numerical = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_factor = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imp_numerical.fit(df_features[numerical])
imp_factor.fit(df_features[factors])
df_features[numerical] = imp_numerical.transform(df_features[numerical])
df_features[factors] = imp_factor.transform(df_features[factors])

X_train, X_test, y_train, y_test = train_test_split(df_features, df_output, test_size=0.25, random_state=15)

df_xtrain = pd.DataFrame(X_train, columns=features)
df_ytrain = pd.DataFrame(y_train, columns=output)
df_xtest = pd.DataFrame(X_test, columns=features)
df_ytest = pd.DataFrame(y_test, columns=output)

print("Working")

clf = svm.SVC(kernel='rbf', gamma=1, C=1)
clf.fit(df_xtrain, df_ytrain)
prediction = clf.predict(df_xtest)
acc = accuracy_score(df_ytest, prediction)
print(acc)

# svm.SVC(): 0.9187188891799842
# linear kernel: