import csv
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
import time
def fun(filename1,filename2):
    csv_reader1 = csv.reader(open(filename1))
    csv1 = []
    for i in csv_reader1:
        temp = []
        a = 1
        while a < len(i):
            temp.append(float(i[a]))
            a += 1
        csv1.append(temp)

    csv_reader2 = csv.reader(open(filename2))
    y = [int(i[0]) for i in csv_reader2]
    return csv1,y


def printscore(model,X_test,y_test):
    y_pred = model.predict(X_test)
    # print('micro :', f1_score(y_test, y_pred, average='micro'))
    # print('macro :', f1_score(y_test, y_pred, average='macro'))
    # print('accuracy:',accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

    # return accuracy_score(y_test,y_pred)

def testallfeature():
    X, y = fun('final.csv', 'Category.csv')
    ac1 = 0
    ac2 = 0
    ac3 = 0
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier().fit(X_train, y_train)
    svc = LinearSVC().fit(X_train, y_train)
    gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    ac1 = printscore(rfc, X_test, y_test)
    ac2 = printscore(svc, X_test, y_test)
    ac3 = printscore(gbdt, X_test, y_test)

    print("%f\n %f\n %f\n" % ( ac1, ac2,  ac3))

def testSelectKBest():
    X, y = fun('finaldata70+5.csv', 'Category.csv')
    ac1 = 0
    ac2 = 0
    ac3 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(5, 40):
        X_new = SelectKBest(chi2, i).fit_transform(X, y)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
        # rfc = RandomForestClassifier().fit(X_train, y_train)
        # svc = LinearSVC().fit(X_train, y_train)
        gbdt = GradientBoostingClassifier().fit(X_train, y_train)
        # if printscore(rfc, X_test, y_test) > ac1:
        #     i1 = i
        #     ac1 = printscore(rfc, X_test, y_test)
        # if printscore(svc, X_test, y_test) > ac2:
        #     i2 = i
        #     ac2 = printscore(svc, X_test, y_test)
        if printscore(gbdt, X_test, y_test) > ac3:
            i3 = i
            ac3 = printscore(gbdt, X_test, y_test)
    print("%f %f\n %f %f\n %f %f\n" % (i1, ac1, i2, ac2, i3, ac3))

#  testSelectKBest result
#  29.000000 0.323741  33
#  20.000000 0.291367   32
#  13.000000 0.356115   35


def bestfeature():
    X, y = fun('number_age_col70.csv', 'Category.csv')
    X_new = SelectKBest(chi2, 18).fit_transform(X, y)
    print(X_new[0])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
    gbdt = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    importance1 = gbdt.feature_importances_
    indict1 = np.argsort(importance1)[::-1]
    for i in range(0, 18):
        print("%d(%f)" % (indict1[i], importance1[indict1[i]]))
    ac3 = printscore(gbdt, X_test, y_test)
    print(ac3)

def selectfrommodel():
    X, y = fun('final.csv', 'Category.csv')
    gbdt = GradientBoostingClassifier().fit(X,y)
    X_new = SelectFromModel(gbdt,prefit=True).transform(X)
    print(len(X_new[0]))
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
    gbdt1=GradientBoostingClassifier().fit(X_train,y_train)
    print(printscore(gbdt1,X_test,y_test))


def selectfromsvc():
    X, y = fun('final.csv', 'Category.csv')
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    print(len(X_new[0]))
    print(X_new[0])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
    gbdt1 = GradientBoostingClassifier().fit(X_train, y_train)
    print(printscore(gbdt1, X_test, y_test))


def tiwenyuce():
    X, y = fun('tiwenprediction.csv', 'Category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train[0])
    gbdt1 = GradientBoostingClassifier().fit(X_train, y_train)
    print(printscore(gbdt1, X_test, y_test))


def kmeansfun(n_clusters):
    X,y = fun('tiwenprediction.csv', 'Category.csv')
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_pred = kmeans.predict(X)
    # print(y_pred)
    f=open('kmeanstiwen.csv','w')
    csv_reader =csv.reader(open('Number.csv'))
    number = [i[0] for i in csv_reader]
    # print(number)
    for i in range(0,len(y_pred)):
        f.write(number[i])
        for j in range(0,n_clusters):
            if y_pred[i] == j:
                f.write(',1')
            else:
                f.write(',0')
        f.write('\n')
    f.close()

def cluster():
    X, y = fun('tiwenprediction.csv', 'Category.csv')
    # estimate bandwidth for mean shift
    # bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    # create clustering estimators
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = MiniBatchKMeans(n_clusters=2)
    ward = AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    spectral = SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = DBSCAN(eps=.2)
    affinity_propagation = AffinityPropagation(damping=.9,
                                                       preference=-200)

    average_linkage = AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)

    birch = Birch(n_clusters=2)
    clustering_algorithms = [
        two_means,affinity_propagation, spectral,ward, average_linkage,
        dbscan, birch]
    for algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        print(y_pred)

def predictkmeans():
    X, y = fun('kmeanstiwen.csv', 'Category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    gbdt = GradientBoostingClassifier().fit(X_train,y_train)
    print(printscore(gbdt, X_train, y_train))



if __name__=="__main__":
    # tiwenyuce()
    # kmeansfun(6)
    # predictkmeans()
    # cluster()

    testSelectKBest()

    # X, y = fun('final.csv', 'Category.csv')
    # svc = LinearSVC(penalty='l1',dual=False).fit(X, y)
    # model = SelectFromModel(svc,prefit=True)
    # X_new =model.transform(X)
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
    # rfc = RandomForestClassifier().fit(X_train, y_train)
    # gbdt = GradientBoostingClassifier(n_estimators=120).fit(X_train, y_train)
    # print(printscore(rfc,X_train,y_train))
    # print(printscore(gbdt,X_train,y_train))



    # clf = Pipeline([
    #     ('feature_selection',SelectFromModel(LinearSVC())),
    #     ('classification',RandomForestClassifier())
    # ])
    # clf.fit(X_train,y_train)
    # printscore(clf,X_test,y_test)
    #
    # rf = RandomForestClassifier().fit(X_train,y_train)
    # # printscore(rf,X_test,y_test)
    # importance1 = rf.feature_importances_
    # indict1 = np.argsort(importance1)[::-1]
    # for i in range(0, 30):
    #     print("%d(%f)" % (indict1[i], importance1[indict1[i]]))
    #
    # et = ExtraTreesClassifier(n_estimators=100).fit(X_train,y_train)
    # printscore(et,X_test,y_test)
    # importance = et.feature_importances_
    # indict = np.argsort(importance)[::-1]
    # for i in range (0,30):
    #     print("%d(%f)"%(indict[i],importance[indict[i]]))

    # svc = LinearSVC()
    # rfecv = RFECV(estimator=svc,scoring='accuracy').fit(X,y)
    # print(rfecv.n_features_)
    # plt.figure()
    # plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
    # plt.show()

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2)
    # gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    # printscore(gbdt, X_test, y_test)
    # plt.figure()
    # score_train = list()
    # score_test =list()
    # k =[]
    # for i in range(1,11):
    #     k.append(i)
    #     X_train, X_test, y_train, y_test = fun('number_age_col71.csv', 'Category.csv')
    #     gbdt = GradientBoostingClassifier().fit(X_train, y_train)
    #     score_test.append(accuracy_score(y_test,gbdt.predict(X_test)))
    #     score_train.append(accuracy_score(y_train,gbdt.predict(X_train)))
    # plt.errorbar(k, score_test)
    # plt.errorbar(k,score_train)
    # plt.axis('tight')
    # plt.show()

    # rfc =RandomForestClassifier().fit(X_train,y_train)
    # printscore(rfc,X_test,y_test)

    # X_new =GenericUnivariateSelect(mode='k').fit_transform(csv1, y)
    # X_new = SelectKBest(chi2, k=10).fit_transform(csv1, y)
    # print(X_new)
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2)


