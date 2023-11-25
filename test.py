import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
iris_dataframe = pd.DataFrame(iris_dataset['data'][:,2:4], columns=iris_dataset['feature_names'][2:4])
scat_matrix = pd.plotting.scatter_matrix(iris_dataframe, c=iris_dataset['target'], s=100, figsize=(8,8), alpha=0.9)
# plt.show()
X_train, X_valid, y_train, y_valid = train_test_split(iris_dataset['data'][:,2:4], iris_dataset['target'], test_size=0.25, random_state=0)

X_train_contact = np.concatenate((X_train, y_train.reshape(112, 1)), axis=1)
X_valid_contact = np.concatenate((X_valid, y_valid.reshape(38, 1)), axis=1)

# print(pd.DataFrame(X_train_contact))
# реализуем формулу для растояния крч расчета эээ ну ты пон крч между точками dist = sqrt( (x1-y1)**2 + (x2-y2)**2 + ... + (xn-yn)**2 )
import math

def euclidean_distance(data1, data2):
    disctance = 0
    for i in range(len(data1) - 1):
        disctance += (data1[i] - data2[i]) ** 2
    return math.sqrt(disctance)

def get_neighbors(train, valid, k=1):
    distances = [ (train[i][-1], euclidean_distance(train[i],valid) ) for i in range(len(train)) ]
    distances.sort(key=lambda elem: elem[1])

    neighbors = [distances[i][0] for i in range(k)]

    return neighbors

def prediction(neighbors):
    count = {}
    for instance in neighbors:
        if instance in count:
            count[instance] += 1
        else:
            count[instance] = 1
    target = max(count.items(), key = lambda x: x[1]) [0]
    return target

def accuracy(valid, valid_prediction):
    correct = 0
    for i in range(len(valid)):
        if valid[i][-1] == valid_prediction[i]:
            correct += 1
    return correct/len(valid)

# # ручками
# predictions = []
# for x in range(len(X_valid_contact)):
#     neighbors = get_neighbors(X_train_contact, X_valid_contact[x], k = 5)
#     result = prediction(neighbors)
#     predictions.append(result)
# accuracy_ = accuracy(X_valid_contact, predictions)
# print(accuracy_) 

# НА БУДУЩЕЕ КРЧ точноть это важно, но все идет в расчет что важнее точность или производительность, типа
#  если нам нада цветочки по коробкам раскадать и на задачу выделят 100 рублей то 97 % это заебись, но 
#  если мы решаем например комп зрением человек перед машиной или гавно, из вывода будет следовать затормазит 
#  тачка или нет, то 97 % это мало и ваще залупа, хоть 1млрд но шобы было 100% (ну 99.999999999)

# ну че ручкеами насчитались, в пизду
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=5)
knn_model = knn.fit(X_train, y_train)
knn_predict = knn_model.predict(X_valid)
accuracy_ = accuracy_score(y_valid, knn_predict)
print(accuracy_)
