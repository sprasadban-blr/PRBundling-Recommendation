import pyhdb as db
from hdbcli import dbapi
import math
# from pyrfc import Connection
import re
# from pyrfc import Connection, ABAPApplicationError, ABAPRuntimeError, LogonError, CommunicationError
# from configparser import ConfigParser
from pprint import PrettyPrinter
import pandas as pd
import os
import shap
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import time
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from dtreeviz.trees import dtreeviz
import graphviz
from IPython.core.display import display, HTML
from sklearn.ensemble import RandomForestClassifier as rfc
from lime.lime_tabular import LimeTabularExplainer
from sklearn.cluster import MiniBatchKMeans
from numpy import unique
from numpy import where
import json
import pickle



style.use("ggplot")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
ORIG_SCHEMA = os.getenv("SCHEMA")

flief = {}
ekorg = {}
NO_OF_CLUSTER = 6

'''
Assumption:
    a) Account Assignment <> Blank Or Unassigned
    b) Status not in ‘RFQ created’, ‘PO created’, ‘Contract Created’,  and ‘Scheduling Agreement Created’
    c) PO is not yet assigned
    d) PR created by purchasing team (Direct/Indirect) --> ESTKZ in ('R', 'S')
    e) PR's not marked as deleted --> LOEKZ NOT IN ('X')
'''
def getOpenPRs(startdate, enddate):
    #connection = db.connect(HOST, PORT, USER, PASSWORD)
    connection = dbapi.connect(HOST, PORT, USER, PASSWORD)
    query = """SELECT MATKL, MANDT, BANFN, BNFPO, FLIEF, EKORG
       FROM "SAPABAP1"."EBAN"
       WHERE STATU NOT IN ('A', 'B', 'E', 'K', 'L')
           AND EBELN = ''
           AND ESTKZ in ('R', 'S')
           AND KNTTP NOT IN ('','U')
           AND LOEKZ NOT IN ('X')
           AND LFDAT >=""" +  startdate + " AND LFDAT <= "+ enddate
    SQL_Query = pd.read_sql_query(query, connection)

    originalData = pd.DataFrame(SQL_Query, columns=['MATKL', 'MANDT', 'BANFN', 'BNFPO', 'FLIEF', 'EKORG'])
    print(originalData.head())
    print(originalData.info())
    print(originalData.describe())
    return originalData


# class connection():
#     def __init__(self):
#         ASHOST = 'ldcier9.wdf.sap.corp'
#         CLIENT = '001'
#         SYSNR = '00'
#         USER = 'RAJPRI'
#         PASSWD = ''
#         self.conn = Connection(ashost=ASHOST, sysnr=SYSNR,
#                                client=CLIENT, user=USER, passwd=PASSWD)

#     def qry(self, Fields, SQLTable, Where='', MaxRows=50, FromRow=0):
#         """A function to query SAP with RFC_READ_TABLE"""

#         # By default, if you send a blank value for fields, you get all of them
#         # Therefore, we add a select all option, to better mimic SQL.
#         if Fields[0] == '*':
#             Fields = ''
#         else:
#             Fields = [{'FIELDNAME': x} for x in Fields]  # Notice the format

#         # the WHERE part of the query is called "options"
#         options = [{'TEXT': x for x in Where}]  # again, notice the format
#         for x in options:
#             print(x)
#             for k, v in x.items():
#                 print(k, ":", v)

#         # we set a maximum number of rows to return, because it's easy to do and
#         # greatly speeds up testing queries.
#         rowcount = MaxRows

#         # Here is the call to SAP's RFC_READ_TABLE
#         tables = self.conn.call("RFC_READ_TABLE", QUERY_TABLE=SQLTable,
#                                 DELIMITER='|', FIELDS=Fields, OPTIONS=options)

#         # We split out fields and fields_name to hold the data and the column names
#         fields = []
#         fields_name = []

#         data_fields = tables["DATA"]  # pull the data part of the result set
#         # pull the field name part of the result set
#         data_names = tables["FIELDS"]

#         headers = [x['FIELDNAME'] for x in data_names]  # headers extraction
#         long_fields = len(data_fields)  # data extraction
#         long_names = len(data_names)  # full headers extraction if you want it

#         # now parse the data fields into a list
#         ans = []
#         for line in range(0, long_fields):
#             fields.append(data_fields[line]["WA"].strip())

#             # for each line, split the list by the '|' separator

#         # return the 2D list and the headers
#         fields = [x.strip().split('|') for x in fields]
#         return fields, headers


# def getOpenPRs(startdate, enddate):
#     conn = connection()
#     fields = ['MANDT', 'BANFN', 'BNFPO', 'KNTTP', 'LIFNR', 'FLIEF',
#               'MATKL', 'EKORG', 'ESTKZ', 'STATU', 'LOEKZ', 'EBELN']
#     table = 'EBAN'
#     condition = "LFDAT GE " + startdate + \
#         " AND LFDAT LE " + enddate + " AND EBELN EQ '' "
#     # you need to put a where condition in there... could be anything
#     where = [condition]
#     # max number of rows to return

#     # starting row to return
#     fromrow = 0
#     # query SAP
#     maxrows = 100000
#     results, headers = conn.qry(fields, table, where, maxrows, fromrow)

#     # creating the dataframe
#     df = pd.DataFrame(results, columns=fields)

#     # filtering the dataframen i.e. applying more filters
#     estkz = ['R', 'S']
#     df = df[df.ESTKZ.isin(estkz)]

#     statu = ['A', 'B', 'E', 'K', 'L']
#     df = df[~df.STATU.isin(statu)]

#     knttp = ['', 'U']
#     df = df[~df.KNTTP.isin(knttp)]
#     df = df[df.KNTTP != ' ']

#     df = df[df.LOEKZ != 'X']
#     df.reset_index(inplace=True)
#     df.drop(['index'], axis=1, inplace=True)
#     df.drop(['KNTTP', 'LIFNR', 'MATKL', 'ESTKZ', 'STATU',
#              'LOEKZ', 'EBELN'], axis=1, inplace=True)
#     return df


def getFeatureList(data):
    colIndex = 0
    numericalColNames = []
    numericalColIndex = []
    categoricalColNames = []
    categoricalColIndex = []

    for col in data.columns:
        if (is_string_dtype(data[col])):
            categoricalColNames.append(col)
            categoricalColIndex.append(colIndex)
        elif (is_numeric_dtype(data[col])):
            numericalColNames.append(col)
            numericalColIndex.append(colIndex)
        colIndex = colIndex + 1
    print("Numerical Columns")
    print(numericalColNames)
    print("Categorical Columns")
    print(categoricalColNames)
    return (numericalColNames, numericalColIndex, categoricalColNames, categoricalColIndex)


def computeDataUniqueness(data):
    print("computing unique data list for FLIEF")
    flief.update(getDataUniqueValue(data["FLIEF"].unique().tolist()))

    print("computing unique data list for EKORG")
    ekorg.update(getDataUniqueValue(data["EKORG"].unique().tolist()))

    print(len(flief), len(ekorg))


def getDataUniqueValue(data):
    dataMap = {}
    count = 0
    for value in data:
        dataMap[value] = count
        count = count + 1
    return dataMap


def executeKProtoTypeCluster(data, categoricalColIndex, k):
    kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2)
    clusters = kproto.fit_predict(data, categorical=categoricalColIndex)
    # Print cluster centroids of the trained model.
    print("===============================================")
    centroids = kproto.cluster_centroids_
    print("Cluster Centroids:")
    print(centroids)
    labels = kproto.labels_
    print("Labels:")
    print(labels)
    print("===============================================")
    # Print training statistics
#     print(kproto.cost_)
#     print(kproto.n_iter_)
#     print(kproto.labels_)
    return (clusters, centroids, labels)


def executeKModeCluster(data, k):
    print("inside kmode")
    kmodes = KModes(n_clusters=k, init='Cao', n_init=5, verbose=2)
    clusters = kmodes.fit_predict(data)
    # Print the cluster centroids
    print("===============================================")
    centroids = kmodes.cluster_centroids_
    print("Cluster Centroids:")
    print(centroids)
    labels = kmodes.labels_
    print("Labels:")
    print(labels)
    print("===============================================")
    # Print training statistics
#     print(kmodes.cost_)
#     print(kmodes.n_iter_)
#     print(kmodes.labels_)
    return (clusters, centroids, labels)


def getPRList(dataArr):
    uniquePRs = []
    for data in dataArr:
        for pr in data:
            if(pr not in uniquePRs):
                uniquePRs.append(pr)
    return uniquePRs


def getGroupedClusters(originalData):
    clusters = originalData['cluster']
    uniqueClusters = unique(clusters)
    print(uniqueClusters)
    originalDataArr = originalData.to_numpy()
    groupedClusterPRs = {}
    # create scatter plot for samples from each cluster
    for cluster in uniqueClusters:
        # get row indexes for samples with this cluster
        row_ix = where(clusters == cluster)
        groupedClusterPRs[str(cluster)] = getPRList(originalDataArr[row_ix, 1])
    return groupedClusterPRs


def getPRs(startdate, enddate):
    ''' Step1: Get all open PRs '''
    print('Getting open PRs....')
    originalData = getOpenPRs(startdate, enddate)
    print(originalData)
    data = originalData.copy()
    ''' We don't need primary keys for computing clusters '''
    data.drop(['MANDT', 'BANFN', 'BNFPO', 'MATKL'], axis=1, inplace=True)

    ''' Step2: Get categorical and numerical features '''
    print('Checking numerical/categorical features....')
    featureList = getFeatureList(data)
    numericalColNames = featureList[0]
    numericalColIndex = featureList[1]
    categoricalColNames = featureList[2]
    categoricalColIndex = featureList[3]

    ''' Step3: Compute categorical data uniqueness '''
    print('Compute categorical data uniqueness....')
    computeDataUniqueness(data)

    prData = (originalData, data, (numericalColNames,
                                   numericalColIndex, categoricalColNames, categoricalColIndex))
    originalData = prData[0]
    data = prData[1]
    featureList = prData[2]
    numericalColNames = featureList[0]
    numericalColIndex = featureList[1]
    categoricalColNames = featureList[2]
    categoricalColIndex = featureList[3]

    noOfCluster = 3
    print('NO_OF_CLUSTER', noOfCluster)
    print(data)
    print('CLuster PRs....')
    if (len(numericalColNames) > 0):
        clustersColn = executeKProtoTypeCluster(
            data, categoricalColIndex, noOfCluster)
    else:
        clustersColn = executeKModeCluster(data, noOfCluster)
    clusters = clustersColn[0]
    centroids = clustersColn[1]
    labels = clustersColn[2]
    originalData['cluster'] = clusters
    print("original data start head")
    print(originalData.head())
    print("original data end ")

    ''' Step4: Cluster PRs data '''
    groupedClusterPRs = getGroupedClusters(originalData)
    print("Grouped Clusters", len(groupedClusterPRs))
    jsonStr = json.dumps(groupedClusterPRs, indent=4)

    # Step 5: Compute classification for ML interpretibility
    orgData = originalData.copy()
    orgData.drop(['MATKL', 'MANDT', 'BANFN', 'BNFPO', 'cluster'], axis=1, inplace=True)
    print(orgData.head())
    X = pd.get_dummies(orgData, columns=['FLIEF', 'EKORG'])
    # X.drop(['EKORG','FLIEF'],axis=1,inplace=True)
    print(X.head())
    y = originalData['cluster']
    print(y.head())

    clf = DecisionTreeClassifier(random_state=1234)
    model = clf.fit(X, y)

    # Decision tree text representation
    text_representation = tree.export_text(clf)
    print(text_representation)

    viz = dtreeviz(clf, X, y,
                   target_name="Cluster",
                   feature_names=X.columns,
                   class_names=['CLUSTER_0', 'CLUSTER_1', 'CLUSTER_2'])

    # viz.save("dtree_1.svg")

    # DOT data
    dot_data = tree.export_graphviz(clf, out_file=None,
                                     feature_names=X.columns,
                                     class_names=['CLUSTER_0',
                                                   'CLUSTER_1', 'CLUSTER_2'],
                                     filled=True)
     # Draw graph
    # graph = graphviz.Source(dot_data, format="png")
    # png_bytes = graph.pipe(format='png')
    # with open('dtree_2.png', 'wb') as f:
    #     f.write(png_bytes)

        # Shapley values interpretibility
    plt.clf()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, features=X,
                    feature_names=X.columns, show=False)
    plt.savefig('shap1.png')
    plt.close()

    # LIME interpretbility
    clf_lime = rfc()
    clf_lime.fit(X, y)
    ''' Pickle LIME model and data sets for getting instance based ML interpretibility '''
    # save the model to disk
    print('Saving Lime model and data sets..')
    # Pickle lime model
    pickle.dump(clf_lime, open('finalized_lime_model.sav', 'wb'))
    # Pickle orginal data
    pickle.dump(originalData, open('original_data.sav', 'wb'))
    # Pickle one hot vector
    pickle.dump(X, open('explainer_data.sav', 'wb'))
    print('Saving completed !..')

    return groupedClusterPRs, originalData


def generateMLInterpretability(jsonData):
    ''' Load picked file and generate instance based ML interpretibility '''
    # load the model from disk
    print('Loading LIME model and data sets..')
    lime_model = pickle.load(open('finalized_lime_model.sav', 'rb'))
    originalDataLoaded = pickle.load(open('original_data.sav', 'rb'))
    X = pickle.load(open('explainer_data.sav', 'rb'))
    print('Loading completed..')

    json_dict = json.loads(jsonData)
    print(json_dict)

    dataInstance = originalDataLoaded[(originalDataLoaded['MANDT'] == json_dict["MANDT"]) & (originalDataLoaded['BANFN'] == json_dict["BANFN"])
                                      & (originalDataLoaded['BNFPO'] == json_dict["BNFPO"]) & (originalDataLoaded['FLIEF'] == json_dict["FLIEF"])
                                      & (originalDataLoaded['EKORG'] == json_dict["EKORG"])]
    print(dataInstance)
    index = dataInstance.index[0]
    classNames = ['CLUSTER_0', 'CLUSTER_1', 'CLUSTER_2']
    explainer = LimeTabularExplainer(
        X,
        feature_names=X.columns,
        class_names=classNames,
        discretize_continuous=False,
        verbose=True)
    lime = explainer.explain_instance(X.iloc[[index]].to_numpy()[
                                      0], lime_model.predict_proba)
    lime.save_to_file('loaded_lime_instance.html')
    print(os.getcwd())


if __name__ == "__main__":
    jsonStr, originalData = getPRs('20180401', '20181231')
    print(jsonStr)
