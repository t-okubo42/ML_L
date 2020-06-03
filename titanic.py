import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



titanic = pd.read_csv('hoge_text.csv')
titanic_assignment = pd.read_csv('hoge_hoge_piyo.csv')

y = titanic['Survived']
x = np.array(titanic.drop(['Survived', 'PassengerId'], axis=1))

y_array = np.array(y)
x_array = np.array(x)

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size = 0.251, random_state = 0)

rfc = RandomForestClassifier(random_state = 0)
rfc.fit(x_train, y_train)



x1 = np.array(hoge_hoge.drop(['PassengerId'], axis=1))
x1_test = np.array(x1)


y_pred = rfc.predict(x1_test)

print(x1_test.shape)
print(y_pred.shape)
print(f'{accuracy_score(y_pred, y_test):5f}')


PassengerId = np.array(titanic_assignment["PassengerId"]).astype(int)# PassengerIdを取得
y_sol = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])# y_predとPassengerIdをデータフレームに
y_sol.to_csv("hoge_test.csv", index_label = ["PassengerId"])# hogehoge.csvとして書き出し