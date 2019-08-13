import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Kmeans_params_count():
    '''KMeans 参数k确立和绘图'''
    def __init__(self,x_array):
        self.x_array = x_array

    def kmeans_k_confirm(self,k_range=range(1,9)):
        '''KMeans K值确定，折线图。inertias：样本离最近簇类中心点距离和'''
        inertias = []
        clusters = {}
        for k in k_range:
            kms = KMeans(n_clusters=k).fit(self.x_array)
            inertias.append(kms.inertia_)
            clusters[k] = kms.predict(self.x_array)
        plt.plot([i for i in k_range],inertias,'ro-')
        plt.ylabel('Inertia')
        plt.xlabel('K')
        plt.title('KMeans Inertia Decrease')
        return inertias, clusters

    def kmeans_k_scatter(self,clusters,figsize=(20,8)):
        '''PCA降至2维，绘制簇划分散点图。输入clusters为kmeans_k_confirm返回的第二个值'''
        palette=['r','b','g','y','black','orange','violet','brown']
        pca_results = PCA(n_components=2).fit_transform(self.x_array)
        plt.figure(figsize=figsize)
        n=1
        for k, hue in clusters.items():
            plt.subplot(2,4,n)
            g = sns.scatterplot(x=pca_results[:,0], y=pca_results[:,1], hue=hue, palette=palette[:k], legend=False)
            g.set_title('K=%s'%k)
            n+=1


#k均值聚类：初始聚类中心改进算法
# from numpy import *
# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))#平方，求和，开方
 
# init centroids with random samples 随机初始质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape#求行列数
    centroids = np.zeros((k, dim))#创建空矩阵，放初始点
    #第一个点
    index = int(np.random.uniform(0, numSamples))
    centroids[0, :] = dataSet[index, :]
    #第二个点
    A1=np.mat(np.zeros((numSamples, 1)))
    for i in range(numSamples):
        distance = euclDistance(centroids[0, :], dataSet[i, :])
        A1[i] = distance
    centroids[1, :]= dataSet[np.nonzero(A1[:, 0] == max(A1))[0]]
    
    #第三个点及以后，
    #然后再选择距离前两个点的最短距离最大的那个点作为第三个初始类簇的中心点，
    j = 1
    while j<=k-2:
        mi = np.mat(np.zeros((numSamples, 1)))
        for i in range(numSamples):
            distance1 = euclDistance(centroids[j-1, :], dataSet[i-1, :])
            distance2 = euclDistance(centroids[j, :], dataSet[i-1, :])
            mi[i-1] = min([distance1,distance2])
        centroids[1+j, :]= dataSet[np.nonzero(mi[:, 0] == max(mi))[0]]
        j=j+1
    return centroids
 
# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]#行数
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True
 
    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)#调用初始化质心函数
 
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist  = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])#调用前面的函数
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
                    
            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
 
        ## step 4: update centroids
        for j in range(k):
            #找出每一类的点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            #clusterAssment[:, 0].A == j测试所有数据的类相同为true不同为false
            #np.nonzero()[0]把所有为true的位置写出来
            #pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == j)[0]]  .A的作用目前不清楚，不加也一样
            #求每一类的中心店
            centroids[j, :] = mean(pointsInCluster, axis = 0)
            
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment

# show your cluster only available with 3-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim > 3:
        print("Sorry! I can not draw because the dimension of your data is not 3!")
        return 1
 
    mark = ['r', 'g', 'b', 'y', 'm', 'k']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1
    
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    
    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        ax.scatter(dataSet[i, 0], dataSet[i, 1], dataSet[i, 2], c=mark[markIndex], s=10)
   
    mark = ['r', 'b', 'g', 'k', 'm', 'y']
    # draw the centroids
    for i in range(k):
        ax.scatter(centroids[i, 0], centroids[i, 1], dataSet[i, 2], c=mark[3],s=100)
        #plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)