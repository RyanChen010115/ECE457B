import csv
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def ensemble100(training_feature, training_target, val_feature, val_target, test_feature, test_target):
    trees = []
    ensemble_val = 0
    ensemble_test = 0
    val_accurate = [0]*100
    test_accurate = [0]*100
    hist = []
    for i in range(100):
        temp = tree.DecisionTreeClassifier(max_depth=3)
        temp.fit(training_feature,training_target)
        trees.append(temp)
    
    for i, curr in enumerate(val_feature):
        vote = 0
        for j,t in enumerate(trees):
            temp = t.predict([curr])[0]
            vote += temp
            if temp == val_target[i]:
                val_accurate[j] += 1
        if vote >= 50 and 1 == val_target[i]:
            ensemble_val += 1
        if vote < 50 and 0 == val_target[i]:
            ensemble_val += 1

    for i, curr in enumerate(test_feature):
        vote = 0
        for j, t in enumerate(trees):
            temp = t.predict([curr])[0]
            vote += temp
            if temp == test_target[i]:
                test_accurate[j] += 1
        if vote >= 50 and 1 == test_target[i]:
            ensemble_test += 1
        if vote < 50 and 0 == test_target[i]:
            ensemble_test += 1

    val_ensemble_acc = ensemble_val/len(val_target)
    hist.append(val_ensemble_acc)
    test_ensemble_acc = ensemble_test/len(test_target)
    hist.append(test_ensemble_acc)
    for i in val_accurate:
        hist.append(i/len(val_target))
    for i in test_accurate:
        hist.append(i/len(test_target))

    plt.hist(hist)
    plt.show()
    print(test_ensemble_acc)
    print(val_ensemble_acc)

def adaboostNone(training_feature, training_target, val_feature, val_target):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(training_feature, training_target)
    correct = 0
    
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == val_target[i]:
            correct += 1
    print(correct/len(val_target))

def adaboostEstimator(training_feature, training_target, val_feature, val_target, e):
    clf = AdaBoostClassifier(n_estimators=100, base_estimator=e)
    clf.fit(training_feature, training_target)
    correct = 0
    
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == val_target[i]:
            correct += 1
    print(correct/len(val_target))
    
def gradientBoost(training_feature, training_target, val_feature, val_target):
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(training_feature, training_target)
    correct = 0
    
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == val_target[i]:
            correct += 1
    print(correct/len(val_target))

def gradientBoost(training_feature, training_target, val_feature, val_target):
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(training_feature, training_target)
    correct = 0
    
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == val_target[i]:
            correct += 1
    print(correct/len(val_target))
    

def xgboost(training_feature, training_target, val_feature, val_target):
    bst = XGBClassifier(n_estimators=100)
    bst.fit(training_feature, training_target)
    # clf.fit(training_feature, training_target)
    correct = 0
    for i, curr in enumerate(val_feature):
        if bst.predict([curr])[0] == val_target[i]:
            correct += 1
    print(correct/len(val_target))

sk_data = []
sk_target = []
with open("A1Q4DecisionTrees.csv") as f:
    t = csv.reader(f)
    header = next(t)
    for row in t:
        sk_target.append(int(row[-1]))
        temp_features = []
        for index,entry in enumerate(row):
            if index < 9:
                temp_features.append(float(entry))
        sk_data.append(temp_features)
l = len(sk_target)
#Part1
# ensemble100(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)],sk_data[math.floor(l*.9):],sk_target[math.floor(l*.9):])

#Part2A
# print("default")
# adaboostNone(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])

# print("Decision Tree")
# dt = tree.DecisionTreeClassifier()
# adaboostEstimator(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)], dt)

# rf = RandomForestClassifier()
# print("Random Forest")
# adaboostEstimator(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)], rf)

#part 2c
print("adaboost")
adaboostNone(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
print("gradient boost")
gradientBoost(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
print("XGBoost")
xgboost(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
