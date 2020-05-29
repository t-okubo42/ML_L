import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



titanic = pd.read_csv('hogehoge.csv')

y = titanic['Piyopiyo']
x = np.array(titanic.drop(['Piyopiyo', 'Mumyamumya'], axis=1))

y_a = np.array(y)
x_a = np.array(x)

x_train, x_test, y_train, y_test = train_test_split(x_a, y_a, test_size = 0.4, random_state = 0)

rfc = RandomForestClassifier(random_state = 0)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)


print(f'{accuracy_score(y_pred, y_test):5f}')
print('x_train', x_train.shape)