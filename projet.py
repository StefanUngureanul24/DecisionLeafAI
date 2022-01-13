from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

COLUMN_X1_LABEL  = "X"
COLUMN_X2_LABEL  = "Y"
COLUMN_HUE_LABEL = "B" 

def get_confusion_matrix(data, inliers):   
    return confusion_matrix(data[COLUMN_HUE_LABEL].values, inliers[COLUMN_HUE_LABEL].values)


def exercice_4_1(df):
    data = df.drop("B", axis = 1)

    attribIdx = [0, 1]
    dl = DecisionLeaf(data, attribIdx)    

    inliers = df.copy()
    inliers[COLUMN_HUE_LABEL] = inliers.apply(lambda row: 0 if not dl.isOutlier(row) else 1, axis=1)
    
    res = sns.scatterplot(
        x = COLUMN_X1_LABEL, y = COLUMN_X2_LABEL, 
        hue  = COLUMN_HUE_LABEL, data = inliers
    )

    plt.show()

    print(get_confusion_matrix(df, inliers))

def exercice_4_2(df):
    data = df.drop("B", axis = 1)

    attribIdx = [0, 1]
    dt = buildDecisionTree_4_2(data, True, attribIdx)

    inliers = df.copy()
    inliers[COLUMN_HUE_LABEL] = inliers.apply(lambda row: 0 if not dt.isOutlier(row) else 1, axis=1)
    
    res = sns.scatterplot(
        x = COLUMN_X1_LABEL, y = COLUMN_X2_LABEL, 
        hue = COLUMN_HUE_LABEL, data = inliers
    )

    plt.show()

    print(get_confusion_matrix(df, inliers))

def exercice_4_3(df, i):
    data = df.drop("B", axis = 1)

    attribIdx = [0, 1]
    dt = buildDecisionTree_4_3(data, True, attribIdx, i)

    inliers = df.copy()
    inliers[COLUMN_HUE_LABEL] = inliers.apply(lambda row: 0 if not dt.isOutlier(row) else 1, axis=1)
    
    res = sns.scatterplot(
        x = COLUMN_X1_LABEL, y = COLUMN_X2_LABEL, 
        hue = COLUMN_HUE_LABEL, data = inliers
    )

    plt.show()

    print(get_confusion_matrix(df, inliers))

def get_attribute(data, attribIdx):
    attribute  = attribIdx[0]
    buffer_std = np.std(data.iloc[:, 0].values)
    
    for i in range(1, len(attribIdx)):
        o_i = np.std(data.iloc[:, i].values)      

        if o_i > buffer_std:
            attribute = attribIdx[i]

    return attribute

def getSplitParameters(data, attribIdx):
    attribute  = get_attribute(data, attribIdx)
    attribute_values = data.iloc[:, attribute].values
    
    init_array = np.array([[min(attribute_values)], [max(attribute_values)]])
    values     = np.array([[v] for v in attribute_values])

    kmeans = KMeans(n_clusters = 2, init = init_array, n_init = 1).fit(values)
    [[a], [b]] = kmeans.cluster_centers_

    return attribute, a, b

def buildDecisionTree_4_2(data, central, attribIdx):
    if len(data) >= 4:
        if len(attribIdx) >= 2: 
            currentAttrib, a, b = getSplitParameters(data, attribIdx)
            
            left   = data[data.iloc[:, currentAttrib].map(lambda x : x <= a)]
            middle = data[data.iloc[:, currentAttrib].map(lambda x : a < x <= b)]
            right  = data[data.iloc[:, currentAttrib].map(lambda x : x < b)]
   
            attribIdx.remove(currentAttrib)
   
            L = buildDecisionTree_4_2(left,   False, attribIdx)
            M = buildDecisionTree_4_2(middle, True,  attribIdx)
            R = buildDecisionTree_4_2(right,  False, attribIdx)
            
            return Node(currentAttrib, a, b, L, M, R)
        else:
            return DecisionLeaf(data, attribIdx)
    else:
        if central:
            return DirectDecision(False)
        else:
            return DirectDecision(True)
 
def buildDecisionTree_4_3(data, central, attribIdx, count_to_end):
    if len(data) >= 4:
        if count_to_end > 1: 
            currentAttrib, a, b = getSplitParameters(data, attribIdx)
            
            left   = data[data.iloc[:, currentAttrib].map(lambda x : x <= a)]
            middle = data[data.iloc[:, currentAttrib].map(lambda x : a < x <= b)]
            right  = data[data.iloc[:, currentAttrib].map(lambda x : x < b)]   
        
            count_to_end = count_to_end - 1            

            L = buildDecisionTree_4_3(left,   False, attribIdx, count_to_end)
            M = buildDecisionTree_4_3(middle, True,  attribIdx, count_to_end)
            R = buildDecisionTree_4_3(right,  False, attribIdx, count_to_end)

            
            return Node(currentAttrib, a, b, L, M, R)
        else:
            return DecisionLeaf(data, attribIdx)
    else:
        if central:
            return DirectDecision(False)
        else:
            return DirectDecision(True)

class DecisionLeaf:
    def __init__(self, data, attribIdx):
        currentAttrib, a, b = getSplitParameters(data, attribIdx)
        self.a = a
        self.b = b
        self.i = currentAttrib
   
    def isOutlier(self, row):
        return not (self.a < row.iloc[self.i] <= self.b)
        
    

class Node:
    def __init__(self, attribut, a, b, L, M, R):
        self.attribut = attribut
        self.a = a
        self.b = b
        self.L = L 
        self.M = M 
        self.R = R

    def isOutlier(self, row):
        value = row.iloc[self.attribut]

        if value <= self.a:
            return self.L.isOutlier(row)
        elif value <= self.b:
            return self.M.isOutlier(row)
        else:
            return self.R.isOutlier(row)
        

class DirectDecision:
    def __init__(self, outlier):
        self.outlier = outlier

    def isOutlier(self, row):
        return self.outlier


def main():
    df = pd.read_csv("data.csv", delimiter = "\t", header = None)
    df.columns = [COLUMN_X1_LABEL, COLUMN_X2_LABEL, COLUMN_HUE_LABEL]
    df.head()

    #decommenter ci-dessous pour afficher le graphique et la matrice de confusion de l'exercice souhaité
    #pour l'exercice 4_3 preciser la hauteur h de l'arbre (df,h)

    #exercice_4_1(df)
    #exercice_4_2(df)
    #exercice_4_3(df,2)
	
	

main()
