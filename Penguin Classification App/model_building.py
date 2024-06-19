import pandas as pd

penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

X = df.drop('species', axis=1)
Y = df['species']

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, Y)

import pickle
pickle.dump(model, open('penguins_model.pkl', 'wb'))