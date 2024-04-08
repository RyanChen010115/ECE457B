import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt
import numpy as np
# from imblearn.over_sampling import RandomOverSampler


def knn(training_feature, training_target, val_feature, val_target):
    clf = KNeighborsClassifier()
    # clf.fit(training_feature,training_target)
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    # X_over, y_over = oversample.fit_resample(training_feature, training_target)
    # clf.fit(X_over, y_over)
    clf.fit(training_feature,training_target)
    #part 1a
    correct = 0
    # for i, curr in enumerate(val_feature):
    #     if clf.predict([curr])[0] == val_target[i]:
    #         correct += 1
    # print(correct/len(val_target))

    #part 1d
    # correct1 = 0
    # total1 = 0
    # correct0 = 0
    # total0 = 0
    # for i, curr in enumerate(val_feature):
    #     if val_target[i] == 0.0:
    #         total0 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct0 += 1
    #     else:
    #         total1 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct1 += 1
    # print("class1")
    # if total1 > 0:
    #     print(correct1/total1)
    # print("class0")
    # if total0 > 0:
    #     print(correct0/total0)

    #part1e
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == 1:
            if val_target[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if val_target[i] == 0:
                tn += 1
            else:
                fn += 1
    print("tp, fp, tn, fn")
    print([tp,fp,tn,fn])

    

def decisionTree(training_feature, training_target, val_feature, val_target):
    clf = DecisionTreeClassifier()
    # clf.fit(training_feature,training_target)
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    # X_over, y_over = oversample.fit_resample(training_feature, training_target)
    # clf.fit(X_over, y_over)
    clf.fit(training_feature,training_target)
    #part 1a
    correct = 0
    # for i, curr in enumerate(val_feature):
    #     if clf.predict([curr])[0] == val_target[i]:
    #         correct += 1
    # print(correct/len(val_target))

    #part 1d
    # correct1 = 0
    # total1 = 0
    # correct0 = 0
    # total0 = 0
    # for i, curr in enumerate(val_feature):
    #     if val_target[i] == 0.0:
    #         total0 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct0 += 1
    #     else:
    #         total1 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct1 += 1
    # print("class1")
    # if total1 > 0:
    #     print(correct1/total1)
    # print("class0")
    # if total0 > 0:
    #     print(correct0/total0)

    #part1e
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == 1:
            if val_target[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if val_target[i] == 0:
                tn += 1
            else:
                fn += 1
    print("tp, fp, tn, fn")
    print([tp,fp,tn,fn])


def linear_regression(training_feature, training_target, val_feature, val_target):
    clf = LinearRegression()
    # clf.fit(training_feature,training_target)
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    # X_over, y_over = oversample.fit_resample(training_feature, training_target)
    # clf.fit(X_over, y_over)
    clf.fit(training_feature,training_target)
    #part 1a
    correct = 0
    # for i, curr in enumerate(val_feature):
    #     if clf.predict([curr])[0] == val_target[i]:
    #         correct += 1
    # print(correct/len(val_target))

    #part 1d
    # correct1 = 0
    # total1 = 0
    # correct0 = 0
    # total0 = 0
    # for i, curr in enumerate(val_feature):
    #     if val_target[i] == 0.0:
    #         total0 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct0 += 1
    #     else:
    #         total1 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct1 += 1
    # print("class1")
    # if total1 > 0:
    #     print(correct1/total1)
    # print("class0")
    # if total0 > 0:
    #     print(correct0/total0)

    #part1e
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == 1:
            if val_target[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if val_target[i] == 0:
                tn += 1
            else:
                fn += 1
    print("tp, fp, tn, fn")
    print([tp,fp,tn,fn])


def logistic_regression(training_feature, training_target, val_feature, val_target):
    clf = LogisticRegression(max_iter=1000)
    # clf.fit(training_feature,training_target)
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    # X_over, y_over = oversample.fit_resample(training_feature, training_target)
    # clf.fit(X_over, y_over)
    clf.fit(training_feature,training_target)
    #part 1a
    correct = 0
    # for i, curr in enumerate(val_feature):
    #     if clf.predict([curr])[0] == val_target[i]:
    #         correct += 1
    # print(correct/len(val_target))

    #part 1d
    # correct1 = 0
    # total1 = 0
    # correct0 = 0
    # total0 = 0
    # for i, curr in enumerate(val_feature):
    #     if val_target[i] == 0.0:
    #         total0 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct0 += 1
    #     else:
    #         total1 += 1
    #         if clf.predict([curr])[0] == val_target[i]:
    #             correct1 += 1
    # print("class1")
    # if total1 > 0:
    #     print(correct1/total1)
    # print("class0")
    # if total0 > 0:
    #     print(correct0/total0)

    #part1e
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i, curr in enumerate(val_feature):
        if clf.predict([curr])[0] == 1:
            if val_target[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if val_target[i] == 0:
                tn += 1
            else:
                fn += 1
    print("tp, fp, tn, fn")
    print([tp,fp,tn,fn])

def scatterplot(x,y,co):
    colour = ['r' if yy==0.0 else 'b' for yy in co]

    plt.scatter(y, x, c=colour)
    plt.show()

def knn2(training_feature, training_target, val_feature, val_target):
    for i in [1,3,5]:
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(training_feature,training_target)
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i, curr in enumerate(val_feature):
            if clf.predict([curr])[0] == 1:
                if val_target[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if val_target[i] == 0:
                    tn += 1
                else:
                    fn += 1
        print("Neighbors: &d",i)
        print("tp, fp, tn, fn")
        print([tp,fp,tn,fn])

def decisionTree2(training_feature, training_target, val_feature, val_target):
    for k in[1,3,5]:
        clf = DecisionTreeClassifier(max_depth=k,random_state=123123)
        clf.fit(training_feature,training_target)
    
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i, curr in enumerate(val_feature):
            if clf.predict([curr])[0] == 1:
                if val_target[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if val_target[i] == 0:
                    tn += 1
                else:
                    fn += 1
        print(f'Depth: {k}')
        print("tp, fp, tn, fn")
        print([tp,fp,tn,fn])

allx = []
ally = []
allc = []
sk_data = []
sk_target = []
with open("A1Q6RawData.csv") as f:
    t = csv.reader(f)
    header = next(t)
    for row in t:
        sk_target.append(float(row[-1]))
        temp_features = []
        for index,entry in enumerate(row):
            if index < 9:
                temp_features.append(float(entry))
        sk_data.append(temp_features)
        allx.append(float(row[1]))
        ally.append(float(row[2]))
        allc.append(float(row[3]))
l = len(sk_target)
# #Part1a/part1d/part1e
# print("knn")
# knn(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
# print("decision tree")
# decisionTree(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
# print("linear regression")
# linear_regression(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
# print("logistic regression")
# logistic_regression(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])

#Part1c
# scatterplot(allx,ally,allc)

#Part2b
# knn2(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])
#Part2c
decisionTree2(sk_data[:math.floor(l*.8)],sk_target[:math.floor(l*.8)],sk_data[math.floor(l*.8):math.floor(l*.9)],sk_target[math.floor(l*.8):math.floor(l*.9)])

