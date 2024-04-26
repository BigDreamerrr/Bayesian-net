import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from network import Binner
from network2 import BayesianNetwork

# X = np.array([
#     [0, 1, 1],
#     [1, 0, 2],
#     [1, 1, 1],
#     [0, 2, 3],
#     [0, 2, 1],
#     [0, 0, 0],
#     [0, 2, 0]
# ])

# X = np.array([
#     [0, 1, 1],
#     [0, 2, 1],
#     [0, 0, 1],
#     [1, 2, 2],
#     [1, 0, 2],
#     [1, 1, 2],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 2, 0]
# ])

df = pd.read_csv(r'train.csv')
df = df.dropna()
df = df.drop(columns=['Age', 'Fare'])

X = np.zeros((712, 5))
Y = np.zeros(712)

col_index = -1
tables = {}

for (columnName, columnData) in df.items():
    if columnName == 'Survived':
        for index, data in enumerate(columnData):
            Y[index] = data
        continue

    col_index += 1

    table = []

    for row_id, data in enumerate(columnData):
        if type(data) == int or type(data) == float:
            X[row_id][col_index] = data
            continue

        if data not in table:
            table.append(data)

        X[row_id][col_index] = table.index(data)

    tables[columnName] = table

def Bayesian_classify(bayesian_model, fets, pred_pos):
    unfilled_vec = np.array([-1] * fets.shape[1])
    pred = np.array([0] * fets.shape[0])

    for index in range(len(fets)):
        unfilled_vec[:] = fets[index, :]
        unfilled_vec[pred_pos] = -1
        pred[index] = (0 if bayesian_model.predict(
            pred_pos, 0, unfilled_vec) > 0.5 else 1)

    return pred

# [CATEGORY, CATEGORY, CONTINOUS, CATEGORY, CATEGORY, CONTINOUS, CATEGORY]
binner = Binner()
binner.binning(X, [-1, -1, -1, -1, -1])
binner.transform(X)
X[:, 0] -= 1
fets = np.hstack((X, Y.reshape(Y.shape[0], 1)))

np.save('fets', fets)

model = BayesianNetwork()
model.fit(fets.astype(np.int32), r'my_cpt' , threshold=0.01)

y_pred = Bayesian_classify(model, np.c_[X, np.zeros((712, 1))], 5)

np.save('input_x_train',  np.c_[X, np.zeros((712, 1))])
np.save('y_train', y_pred)

print(accuracy_score(y_pred, Y))

df = pd.read_csv(r'test.csv').fillna(0)
df = df.drop(columns=['Age', 'Fare'])

X_test = np.zeros((418, 5))

col_index = -1
for (columnName, columnData) in df.items():
    col_index += 1

    for row_id, data in enumerate(columnData):
        if type(data) == int or type(data) == float:
            X_test[row_id][col_index] = data
            continue

        X_test[row_id][col_index] = tables[columnName].index(data)

binner.transform(X_test)
X_test[:, 0] -= 1

Y_test = Bayesian_classify(model, np.c_[X_test, np.zeros((418, 1))], 5)

np.save('input_x_test',  np.c_[X_test, np.zeros((418, 1))])
np.save('y_test', Y_test)

ans_dict = { 'PassengerId' : [], 'Survived' : [] }

for i in range(0, 418):
    ans_dict['PassengerId'].append(892 + i)
    ans_dict['Survived'].append(Y_test[i])

df2 = pd.DataFrame.from_dict(ans_dict)
df2.to_csv(r'ans.csv', index=False)