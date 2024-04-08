import csv
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
def gender(data):
    total_men = 0
    total_women = 0
    survived_men = 0
    survived_women = 0
    for row in data:
        if row[8] == 1:
            total_men += 1
            survived_men += row[9]
        else:
            total_women += 1
            survived_women += row[9]

    men_ratio = survived_men/total_men
    woman_ratio = survived_women/total_women

    print(total_men)
    print(survived_men)
    print(total_women)
    print(survived_women)
    if men_ratio > woman_ratio:
        return "men"
    return "women"

def simple25(data):
    total_under = 0
    total_over = 0
    survived_under = 0
    survived_over = 0
    for row in data:
        if row[1] <= 25:
            total_under += 1
            survived_under += row[9]
        else:
            total_over += 1
            survived_over += row[9]
    print(total_under)
    print(survived_under)
    print(total_over)
    print(survived_over)

    if survived_under/total_under > survived_over/total_over:
        return "under 25"
    return "over 25"

def simple65(data):
    total_under = 0
    total_over = 0
    survived_under = 0
    survived_over = 0
    for row in data:
        if row[1] < 65:
            total_under += 1
            survived_under += row[9]
        else:
            total_over += 1
            survived_over += row[9]
    print(total_under)
    print(survived_under)
    print(total_over)
    print(survived_over)

    if survived_under/total_under > survived_over/total_over:
        return "under 65"
    return "over 65"

def depth3_genderFirst_25(data):
    total_men_and_under = 0
    survived_men_and_under = 0
    total_men_and_over = 0
    survived_men_and_over = 0
    total_women_and_under = 0
    survived_women_and_under = 0
    total_women_and_over = 0
    survived_women_and_over = 0
    for row in data:
        if row[8] == 1:# men
            if row[1] <= 25:
                total_men_and_under += 1
                survived_men_and_under += row[9]
            else:
                total_men_and_over += 1
                survived_men_and_over += row[9]
        else:
            if row[1] <= 25:
                total_women_and_under += 1
                survived_women_and_under += row[9]
            else:
                total_women_and_over += 1
                survived_women_and_over += row[9]
    res = defaultdict()
    if total_men_and_under > 0:
        res["men and under"] = survived_men_and_under/total_men_and_under
    if total_men_and_over > 0:
        res["men and over"] = survived_men_and_over/total_men_and_over
    if total_women_and_under > 0:
        res["women and under"] = survived_women_and_under/total_women_and_under
    if total_women_and_over > 0:
        res["women and over"] = survived_women_and_over/total_women_and_over
    keeper = defaultdict()
    keeper["men and under"] = (survived_men_and_under,total_men_and_under)
    keeper["men and over"] = (survived_men_and_over,total_men_and_over)
    keeper["women and under"] = (survived_women_and_under,total_women_and_under)
    keeper["women and over"] = (survived_women_and_over,total_women_and_over)

    k = max(res, key = res.get)
    print(k)
    impurity = 0
    for i in keeper:
        if i == k:
            impurity += keeper[k][1]-keeper[k][0]
        else:
            impurity += keeper[i][0]
    print("impurity: %d",impurity/len(data))

def depth3_genderFirst_65(data):
    total_men_and_under = 0
    survived_men_and_under = 0
    total_men_and_over = 0
    survived_men_and_over = 0
    total_women_and_under = 0
    survived_women_and_under = 0
    total_women_and_over = 0
    survived_women_and_over = 0
    for row in data:
        if row[8] == 1:# men
            if row[1] < 65:
                total_men_and_under += 1
                survived_men_and_under += row[9]
            else:
                total_men_and_over += 1
                survived_men_and_over += row[9]
        else:
            if row[1] < 65:
                total_women_and_under += 1
                survived_women_and_under += row[9]
            else:
                total_women_and_over += 1
                survived_women_and_over += row[9]
    res = defaultdict()
    if total_men_and_under > 0:
        res["men and under"] = survived_men_and_under/total_men_and_under
    if total_men_and_over > 0:
        res["men and over"] = survived_men_and_over/total_men_and_over
    if total_women_and_under > 0:
        res["women and under"] = survived_women_and_under/total_women_and_under
    if total_women_and_over > 0:
        res["women and over"] = survived_women_and_over/total_women_and_over
    keeper = defaultdict()
    keeper["men and under"] = (survived_men_and_under,total_men_and_under)
    keeper["men and over"] = (survived_men_and_over,total_men_and_over)
    keeper["women and under"] = (survived_women_and_under,total_women_and_under)
    keeper["women and over"] = (survived_women_and_over,total_women_and_over)

    k = max(res, key = res.get)
    print(k)
    impurity = 0
    for i in keeper:
        if i == k:
            impurity += keeper[k][1]-keeper[k][0]
        else:
            impurity += keeper[i][0]
    print("impurity: %d",impurity/len(data))

def depth3_ageFirst_25(data):
    total_men_and_under = 0
    survived_men_and_under = 0
    total_men_and_over = 0
    survived_men_and_over = 0
    total_women_and_under = 0
    survived_women_and_under = 0
    total_women_and_over = 0
    survived_women_and_over = 0
    for row in data:
        if row[1] <= 25:
            if row[8] == 1:# men
                total_men_and_under += 1
                survived_men_and_under += row[9]
            else:
                total_women_and_under += 1
                survived_women_and_under += row[9]
        else:
            if row[8] == 1:# men
                total_men_and_over += 1
                survived_men_and_over += row[9]
            else:
                total_women_and_over += 1
                survived_women_and_over += row[9]
    res = defaultdict()
    if total_men_and_under > 0:
        res["men and under"] = survived_men_and_under/total_men_and_under
    if total_men_and_over > 0:
        res["men and over"] = survived_men_and_over/total_men_and_over
    if total_women_and_under > 0:
        res["women and under"] = survived_women_and_under/total_women_and_under
    if total_women_and_over > 0:
        res["women and over"] = survived_women_and_over/total_women_and_over
    keeper = defaultdict()
    keeper["men and under"] = (survived_men_and_under,total_men_and_under)
    keeper["men and over"] = (survived_men_and_over,total_men_and_over)
    keeper["women and under"] = (survived_women_and_under,total_women_and_under)
    keeper["women and over"] = (survived_women_and_over,total_women_and_over)

    k = max(res, key = res.get)
    print(k)
    impurity = 0
    for i in keeper:
        if i == k:
            impurity += keeper[k][1]-keeper[k][0]
        else:
            impurity += keeper[i][0]
    print("impurity: %d",impurity/len(data))

def depth3_ageFirst_65(data):
    total_men_and_under = 0
    survived_men_and_under = 0
    total_men_and_over = 0
    survived_men_and_over = 0
    total_women_and_under = 0
    survived_women_and_under = 0
    total_women_and_over = 0
    survived_women_and_over = 0
    for row in data:
        if row[1] < 65:
            if row[8] == 1:# men
                total_men_and_under += 1
                survived_men_and_under += row[9]
            else:
                total_women_and_under += 1
                survived_women_and_under += row[9]
        else:
            if row[8] == 1:# men
                total_men_and_over += 1
                survived_men_and_over += row[9]
            else:
                total_women_and_over += 1
                survived_women_and_over += row[9]
    res = defaultdict()
    if total_men_and_under > 0:
        res["men and under"] = survived_men_and_under/total_men_and_under
    if total_men_and_over > 0:
        res["men and over"] = survived_men_and_over/total_men_and_over
    if total_women_and_under > 0:
        res["women and under"] = survived_women_and_under/total_women_and_under
    if total_women_and_over > 0:
        res["women and over"] = survived_women_and_over/total_women_and_over
    keeper = defaultdict()
    keeper["men and under"] = (survived_men_and_under,total_men_and_under)
    keeper["men and over"] = (survived_men_and_over,total_men_and_over)
    keeper["women and under"] = (survived_women_and_under,total_women_and_under)
    keeper["women and over"] = (survived_women_and_over,total_women_and_over)

    k = max(res, key = res.get)
    print(k)
    impurity = 0
    for i in keeper:
        if i == k:
            impurity += keeper[k][1]-keeper[k][0]
        else:
            impurity += keeper[i][0]
    print("impurity: %d",impurity/len(data))

def gini(data):
    total_men = 0
    total_women = 0
    survived_men = 0
    survived_women = 0
    ages = {}
    for row in data:
        if row[8] == 1:
            total_men += 1
            survived_men += row[9]
        else:
            total_women += 1
            survived_women += row[9]
        if row[1] not in ages:
            ages[row[1]] = [0,0]
        ages[row[1]][0] += 1
        ages[row[1]][1] += row[9]

    #gini gender
    gini_gender = (total_men/len(data))*(1-((survived_men/total_men)**2 + ((total_men-survived_men)/total_men)**2)) + (total_women/len(data))*(1-((survived_women/total_women)**2 + ((total_women-survived_women)/total_women)**2))
    
    #gini age
    gini_age = 0
    for ind in ages:
        total, survive = ages[ind]
        gini_age += (total/len(data))*(1-((survive/total)**2 + ((total-survive)/total)**2))

    print("gender")
    print(gini_gender)
    print("age")
    print(gini_age)

def shannon(data):
    total_men = 0
    total_women = 0
    survived_men = 0
    survived_women = 0
    ages = {}
    for row in data:
        if row[8] == 1:
            total_men += 1
            survived_men += row[9]
        else:
            total_women += 1
            survived_women += row[9]
        if row[1] not in ages:
            ages[row[1]] = [0,0]
        ages[row[1]][0] += 1
        ages[row[1]][1] += row[9]
    shannon_gender = (-1*total_men/len(data))*((survived_men/total_men)*math.log2(survived_men/total_men) + ((total_men-survived_men)/total_men)*math.log2((total_men-survived_men)/total_men)) + (-1*total_women/len(data))*(((survived_women/total_women)*math.log2(survived_women/total_women) + ((total_women-survived_women)/total_women)*math.log2((total_women-survived_women)/total_women)))
    shannon_age = 0
    for ind in ages:
        total, survive = ages[ind]
        if survive/total <= 0 or (total-survive)/total <= 0:
            continue
        shannon_age += (-1*total/len(data))*((survive/total)*math.log2(survive/total) + ((total-survive)/total)*math.log2((total-survive)/total))
    print("gender")
    print(shannon_gender)
    print("age")
    print(shannon_age)

def decisionTree(features, target, val_features, val_target, test_features, test_target):
    print("Gini")
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=123123, criterion="gini")
    clf = clf.fit(features,target)
    # tree.plot_tree(clf)
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf)
    # plt.show()
    accurate = 0
    
    for i in range(len(val_features)):
        # print(val_features[i])
        # print(clf.predict([val_features[i]]))
        if clf.predict([val_features[i]])[0] == val_target[i]:
            accurate+=1
    print("Validation")
    print(accurate/len(val_features))
    
    test_accurate = 0 
    for i in range(len(test_features)):
        # print(val_features[i])
        # print(clf.predict([val_features[i]]))
        if clf.predict([test_features[i]])[0] == test_target[i]:
            test_accurate+=1
    print("Test")
    print(test_accurate/len(test_features))

def decisionTreeEntropy(features, target, val_features, val_target, test_features, test_target):
    print("Entropy")
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=123123, criterion="entropy")
    clf = clf.fit(features,target)
    # tree.plot_tree(clf)
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf)
    # plt.show()
    accurate = 0
    
    for i in range(len(val_features)):
        # print(val_features[i])
        # print(clf.predict([val_features[i]]))
        if clf.predict([val_features[i]])[0] == val_target[i]:
            accurate+=1
    print("Validation")
    print(accurate/len(val_features))
    
    test_accurate = 0 
    for i in range(len(test_features)):
        # print(val_features[i])
        # print(clf.predict([val_features[i]]))
        if clf.predict([test_features[i]])[0] == test_target[i]:
            test_accurate+=1
    print("Test")
    print(test_accurate/len(test_features))

file = []
sk_data = []
sk_target = []
with open("A1Q4DecisionTrees.csv") as f:
    t = csv.reader(f)
    header = next(t)
    for row in t:
        temp = []
        sk_target.append(int(row[-1]))
        temp_features = []
        for index,entry in enumerate(row):
            temp.append(float(entry))
            if index < 9:
                temp_features.append(float(entry))
        file.append(temp)
        sk_data.append(temp_features)
    


#part 1a
# print(gender(file))
        
#part 1c
# print(simple25(file))
# print(simple65(file))
        
#part 1d
# print("gender first 25")
# depth3_genderFirst_25(file)
# print("gender first 65")
# depth3_genderFirst_65(file)
# print("age first 25")
# depth3_ageFirst_25(file)
# print("age first 65")
# depth3_ageFirst_65(file)
        
#part 1e
# gini(file)
# shannon(file)
        
#part 2a
# decisionTree(sk_data,sk_target)

# part 3a
l = len(sk_data)
print("80/10/10")
decisionTree(sk_data[:math.floor(l*0.8)], sk_target[:math.floor(l*.8)], sk_data[math.floor(l*0.8):math.floor(l*0.9)], sk_target[math.floor(l*.8):math.floor(l*.9)],sk_data[math.floor(l*0.9):], sk_target[math.floor(l*.9):])
decisionTreeEntropy(sk_data[:math.floor(l*0.8)], sk_target[:math.floor(l*.8)], sk_data[math.floor(l*0.8):math.floor(l*0.9)], sk_target[math.floor(l*.8):math.floor(l*.9)],sk_data[math.floor(l*0.9):], sk_target[math.floor(l*.9):])

print("34/33/33")
decisionTree(sk_data[:math.floor(l*0.34)], sk_target[:math.floor(l*.34)], sk_data[math.floor(l*0.34):math.floor(l*0.67)], sk_target[math.floor(l*.34):math.floor(l*.67)],sk_data[math.floor(l*0.67):], sk_target[math.floor(l*.67):])
decisionTreeEntropy(sk_data[:math.floor(l*0.34)], sk_target[:math.floor(l*.34)], sk_data[math.floor(l*0.34):math.floor(l*0.67)], sk_target[math.floor(l*.34):math.floor(l*.67)],sk_data[math.floor(l*0.67):], sk_target[math.floor(l*.67):])

print("25/25/50")
decisionTree(sk_data[:math.floor(l*0.25)], sk_target[:math.floor(l*.25)], sk_data[math.floor(l*0.25):math.floor(l*0.5)], sk_target[math.floor(l*.25):math.floor(l*.5)],sk_data[math.floor(l*0.5):], sk_target[math.floor(l*.5):])
decisionTreeEntropy(sk_data[:math.floor(l*0.25)], sk_target[:math.floor(l*.25)], sk_data[math.floor(l*0.25):math.floor(l*0.5)], sk_target[math.floor(l*.25):math.floor(l*.5)],sk_data[math.floor(l*0.5):], sk_target[math.floor(l*.5):])




