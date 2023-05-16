from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

df_csv = pd.read_csv('secondary_data.csv', sep=';')
features = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment',
            'gill-spacing', 'gill-color', 'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
            'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season']
output = ['class']

numerical = ['cap-diameter', 'stem-height', 'stem-width']
factors = [x for x in features if x not in numerical]

df_features = df_csv[features]
df_output = df_csv[output]

df_features[factors] = df_features[factors].apply(lambda x: pd.factorize(x)[0])

X_train, X_test, y_train, y_test = train_test_split(df_features, df_output, test_size=0.25, random_state=15)

df_xtrain = pd.DataFrame(X_train, columns=features)
df_ytrain = pd.DataFrame(y_train, columns=output)
df_xtest = pd.DataFrame(X_test, columns=features)
df_ytest = pd.DataFrame(y_test, columns=output)

clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
clf.fit(df_xtrain, df_ytrain)
prediction = clf.predict(df_xtest)
acc = accuracy_score(df_ytest, prediction)
print(acc)

# adam, 1e-5: 0.999934503536809