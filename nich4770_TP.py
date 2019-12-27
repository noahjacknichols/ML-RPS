import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from joblib import dump, load

# names = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5', 'Sensor6', 'Sensor7', 'Sensor8', 'Result']
names = [str(i) for i in range(65)]
data0 = pd.read_csv('0.csv', names = names)
data1 = pd.read_csv('1.csv', names = names)
data2 = pd.read_csv('2.csv', names = names)
data3 = pd.read_csv('3.csv', names = names)

# data0.plot()
# plt.show()

# data1.plot()
# plt.show()

# data2.plot()
# plt.show()

# data3.plot()
# plt.show()


train_data0, valid_data0 = train_test_split(data0, test_size=0.2)

# train_data0 = data0[:len(data0)-50]
# valid_data0 = data0[len(data0)-50:]

train_data1, valid_data1 = train_test_split(data1, test_size=0.2)
# train_data1 = data1[:len(data1)-50]
# valid_data1 = data1[len(data1)-50:]

train_data2, valid_data2 = train_test_split(data2, test_size=0.2)
# train_data2 = data2[:len(data2)-50]

# valid_data2 = data2[len(data2)-50:]

train_data3, valid_data3 = train_test_split(data3, test_size=0.2)
# train_data3 = data3[:len(data3)-50]
# valid_data3 = data3[len(data3)-50:]

vertical_stack = pd.concat([train_data0, train_data1, train_data2,train_data3], axis = 0)
# vertical_stack = vertical_stack.drop(['Sensor3', 'Sensor4','Sensor5','Sensor6'], axis=1)
# vertical_stack.plot()
# plt.show()
vertical_stack = vertical_stack.sample(frac=1)
# vertical_stack = vertical_stack[:math.ceil(len(vertical_stack)/2)]

pred_stack = pd.concat([valid_data0, valid_data1, valid_data2, valid_data3], axis = 0)
pred_stack = pred_stack.sample(frac=1)
# pred_stack = pred_stack.drop(['Sensor3', 'Sensor4','Sensor5','Sensor6'], axis=1)

# vertical_stack = vertical_stack.dropna()
# vertical_stack = vertical_stack.dropna(axis="columns")

# pred_stack.dropna()
# pred_stack.dropna(axis="columns")
print(vertical_stack)
# vertical_stack.plot()
# plt.show()
# pred_stack.plot()
# plt.show()


#plots
print(vertical_stack.shape)
print(vertical_stack.head(20))
print(vertical_stack.describe())

# data0.plot(kind='box', subplots=True, layout=(2,65), sharex=False, sharey=False)
# plt.show()

# data1.plot(kind='box', subplots=True, layout=(2,65), sharex=False, sharey=False)
# plt.show()

# data2.plot(kind='box', subplots=True, layout=(2,65), sharex=False, sharey=False)
# plt.show()

# data3.plot(kind='box', subplots=True, layout=(2,65), sharex=False, sharey=False)
# plt.show()

# vertical_stack.hist()
# plt.show()

# scatter_matrix(vertical_stack)
# plt.show()


classifiers = [
    # KNeighborsClassifier(7),
    # SVC(kernel="linear", C=0.025),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000)
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()]
]


train = pd.DataFrame(vertical_stack, columns = names[:64])
result = pd.DataFrame(vertical_stack, columns = [names[64]])
print(train)
print(result)

pred_x = pd.DataFrame(pred_stack, columns = names[:64])
pred_y = pd.DataFrame(pred_stack, columns = [names[64]])

for classifier in classifiers:
# classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    count = 0
    classifier.fit(train, np.ravel(result))
    for index, row in pred_stack.iterrows():
        x = classifier.predict([row[:64]])
        # print((int(x[0]),row['Result']))
        if(x[0] == int(row['64'])):
            count+=1
    print(classifier.__class__.__name__,":","{:.2}".format((count/len(pred_stack))), "percent accuracy")

    # score = classifier.score(pred_x, pred_y)
    # print(classifier.__class__.__name__, ":",score)

# MLPClassifier.fit(train, np.ravel(result))
dump(MLPClassifier, 'MLP_model.joblib')

