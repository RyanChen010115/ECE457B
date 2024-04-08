import math
import heapq
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict

def knn(num_neighbors, metric, training, validation):
    if metric not in ["Euclidean","Cosine"]:
        return
    total_validation = len(validation)
    correct_validation = 0
    for ogtt,bmi,age,c in validation:
        neighbors = []
        for t_ogtt,t_bmi,t_age,t_c in training:
            dist = 0
            if metric == "Euclidean":
                dist = math.sqrt((ogtt-t_ogtt)**2+(bmi-t_bmi)**2+(age-t_age)**2)
            elif metric == "Cosine":
                dist = (ogtt*t_ogtt+bmi*t_bmi+age*t_age)/(math.sqrt(ogtt**2+bmi**2+age**2)*math.sqrt(t_ogtt**2+t_bmi**2+t_age**2))
            heapq.heappush(neighbors,(dist,t_c))
        curr_positive = 0
        for _ in range(num_neighbors):
            d,classification = heapq.heappop(neighbors)
            curr_positive+=classification
        if (curr_positive >= (num_neighbors/2) and c == 1) or (curr_positive<(num_neighbors/2) and c == 0):
            correct_validation += 1

    return correct_validation/total_validation

file = []
with open("A1Q3NearestNeighbors.csv") as f:
    t = csv.reader(f)
    for row in t:
        file.append([int(row[0]),float(row[1]),int(row[2]),int(row[3])])
l = len(file)
sk_feature = []
sk_target = []
for i in file:
    sk_feature.append(i[:-1])
    sk_target.append(i[-1])
splits = ["80/10/10","34/33/33","25/25/50"]
my_val_scores = defaultdict(list)
sk_val_scores = defaultdict(list)
my_cosine = defaultdict(list)
sk_cosine = defaultdict(list)
neighbors = [1,3,5,11]
for k in neighbors:
    skl = KNeighborsClassifier(n_neighbors=k)
    sk_cos = KNeighborsClassifier(n_neighbors=k, metric="cosine")

    val = math.floor(l*0.8)
    test = math.floor(l*0.9)
    my_val_scores[k].append(knn(k,"Euclidean",file[:val],file[val:test]))
    my_cosine[k].append(knn(k,"Cosine",file[:val],file[val:test]))
    sk_cos.fit(sk_feature[:val],sk_target[:val])
    cos_pred = sk_cos.predict(sk_feature[val:test])
    sk_cosine[k].append(accuracy_score(sk_target[val:test],cos_pred))
    skl.fit(sk_feature[:val],sk_target[:val])
    v_pred = skl.predict(sk_feature[val:test])
    sk_val_scores[k].append(accuracy_score(sk_target[val:test],v_pred))

    val = math.floor(l*0.34)
    test = math.floor(l*0.67)
    my_val_scores[k].append(knn(k,"Euclidean",file[:val],file[val:test]))
    skl.fit(sk_feature[:val],sk_target[:val])
    v_pred = skl.predict(sk_feature[val:test])
    sk_val_scores[k].append(accuracy_score(sk_target[val:test],v_pred))
    my_cosine[k].append(knn(k,"Cosine",file[:val],file[val:test]))
    sk_cos.fit(sk_feature[:val],sk_target[:val])
    cos_pred = sk_cos.predict(sk_feature[val:test])
    sk_cosine[k].append(accuracy_score(sk_target[val:test],cos_pred))

    val = math.floor(l*0.35)
    test = math.floor(l*0.5)
    my_val_scores[k].append(knn(k,"Euclidean",file[:val],file[val:test]))
    skl.fit(sk_feature[:val],sk_target[:val])
    v_pred = skl.predict(sk_feature[val:test])
    sk_val_scores[k].append(accuracy_score(sk_target[val:test],v_pred))
    my_cosine[k].append(knn(k,"Cosine",file[:val],file[val:test]))
    sk_cos.fit(sk_feature[:val],sk_target[:val])
    cos_pred = sk_cos.predict(sk_feature[val:test])
    sk_cosine[k].append(accuracy_score(sk_target[val:test],cos_pred))

cosine1 = plt.subplot(4,2,1)
euclid1 = plt.subplot(4,2,2)
euclid1.plot(splits,sk_val_scores[1])
euclid1.plot(splits, my_val_scores[1])
euclid1.set_title("Euclidean Distance")
cosine1.plot(splits, sk_cosine[1], label = "SKLearn Accuracy")
cosine1.plot(splits, my_cosine[1], label = "My Accuracy")
cosine1.set_title("k = 1 Cosine Similarity")
cosine1.legend()

cosine3 = plt.subplot(4,2,3)
euclid3 = plt.subplot(4,2,4)
euclid3.plot(splits,sk_val_scores[3])
euclid3.plot(splits, my_val_scores[3])
cosine3.plot(splits, sk_cosine[3])
cosine3.plot(splits, my_cosine[3])
cosine3.set_title("k = 3")

cosine3 = plt.subplot(4,2,5)
euclid3 = plt.subplot(4,2,6)
euclid3.plot(splits,sk_val_scores[5])
euclid3.plot(splits, my_val_scores[5])
cosine3.plot(splits, sk_cosine[5])
cosine3.plot(splits, my_cosine[5])
cosine3.set_title("k = 5")

cosine3 = plt.subplot(4,2,7)
euclid3 = plt.subplot(4,2,8)
euclid3.plot(splits,sk_val_scores[11])
euclid3.plot(splits, my_val_scores[11])
cosine3.plot(splits, sk_cosine[11])
cosine3.plot(splits, my_cosine[11])
cosine3.set_title("k = 11")
plt.show()