import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve, validation_curve

titanic = pd.read_csv("train.csv")
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"]=="male", "Sex"]=0
titanic.loc[titanic["Sex"]=="female", "Sex"]=1
titanic["Embarked"]=titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0
titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2

titanic_test = pd.read_csv("test.csv")
titanic_test["Age"]=titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"]=="male", "Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female", "Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna('S')
titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2

name = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
data = titanic.loc[:,name]
target = titanic.loc[:,"Survived"]
test_data = titanic_test.loc[:,name]

print titanic.shape
#print test_data.describe()
#print test_data

dt = tree.DecisionTreeClassifier()
DTscores = cross_val_score(dt,data,target,cv=10,scoring='accuracy')
DTtrain_sizes, DTtrain_loss, DTtest_loss = learning_curve(dt,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Decision Tree:'
print DTscores.mean()
print 'Loss:'
print -np.mean(DTtest_loss,axis=1)
print

deep = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,20), random_state=1)
DLscores = cross_val_score(deep,data,target,cv=10,scoring='accuracy')
DLtrain_sizes, DLtrain_loss, DLtest_loss = learning_curve(deep,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Deep Learning:'
print DLscores.mean()
print 'Loss:'
print -np.mean(DLtest_loss,axis=1)
print

alg = svm.SVC(gamma=0.001)
SVMscores = cross_val_score(alg,data,target,cv=10,scoring='accuracy')
param_range = np.logspace(-6,-2.3,5)
SVMtrain_loss, SVMtest_loss = validation_curve(svm.SVC(),data,target,param_name='gamma',param_range=param_range,cv=10,scoring='neg_mean_squared_error')
print 'SVM:'
print SVMscores.mean()
print 'Loss:'
print -np.mean(SVMtest_loss,axis=1)
print

naivebayes = GaussianNB()
NBscores = cross_val_score(naivebayes,data,target,cv=10,scoring='accuracy')
NBtrain_sizes, NBtrain_loss, NBtest_loss = learning_curve(naivebayes,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Naive Bayes:'
print NBscores.mean()
print 'Loss:'
print -np.mean(NBtest_loss,axis=1)
print

logistic = LogisticRegression()
LRscores = cross_val_score(logistic,data,target,cv=10,scoring='accuracy')
LRtrain_sizes, LRtrain_loss, LRtest_loss = learning_curve(logistic,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Logistic Regression:'
print LRscores.mean()
print 'Loss:'
print -np.mean(LRtest_loss,axis=1)
print

knn = KNeighborsClassifier(30)
knnscores = cross_val_score(knn,data,target,cv=10,scoring='accuracy')
knntrain_sizes, knntrain_loss, knntest_loss = learning_curve(knn,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'k-Nearest Neighbors:'
print knnscores.mean()
print 'Loss:'
print -np.mean(knntest_loss,axis=1)
print

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagscores = cross_val_score(bagging,data,target,cv=10,scoring='accuracy')
bagtrain_sizes, bagtrain_loss, bagtest_loss = learning_curve(bagging,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Bagging:'
print bagscores.mean()
print 'Loss:'
print -np.mean(bagtest_loss,axis=1)
print

forest = RandomForestClassifier(n_estimators=10, min_samples_leaf=1, min_samples_split=2, random_state=1)
forestscores = cross_val_score(forest,data,target,cv=10,scoring='accuracy')
ftrain_sizes, ftrain_loss, ftest_loss = learning_curve(forest,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Random Forest:'
print forestscores.mean()
print 'Loss:'
print -np.mean(ftest_loss,axis=1)
print

adaboost = AdaBoostClassifier(n_estimators=10)
adascores = cross_val_score(adaboost,data,target,cv=10,scoring='accuracy')
adatrain_sizes, adatrain_loss, adatest_loss = learning_curve(adaboost,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'AdaBoost:'
print adascores.mean()
print 'Loss:'
print -np.mean(adatest_loss,axis=1)
print

GB = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, max_depth=None, random_state=0)
GBscores = cross_val_score(GB,data,target,cv=10,scoring='accuracy')
GBtrain_sizes, GBtrain_loss, GBtest_loss = learning_curve(adaboost,data,target,cv=10,scoring='neg_mean_squared_error',
                                                          train_sizes=[0.1,0.25,0.5,0.75,1])
print 'Gradient Boosting'
print GBscores.mean()
print 'Loss:'
print -np.mean(GBtest_loss,axis=1)
print

forest.fit(data,target)
predictions = forest.predict(test_data)
result = pd.DataFrame({'PassengerId':titanic_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int64)})
result.to_csv("result.csv",index=False)
print 'Result is output successfully!'
#print predictions
print 'The result is as follows:'
fin = pd.read_csv('result.csv')
print fin