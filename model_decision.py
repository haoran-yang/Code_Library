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


class ModelChoose():
    '''模型选择'''
    def __init__(self):
        pass
    def model_init(self,reDefine_clf = {},used_clf=[]):
        clfs = {'xgb':xgboost.XGBClassifier(n_estimators=100, max_depth=3),
                'lgb':lightgbm.LGBMClassifier(n_estimators=100, max_depth=3),
                'gbdt':ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=3),
                'rf':ensemble.RandomForestClassifier(n_estimators=200, max_depth=6),
                'logit':linear_model.LogisticRegression(),
                'svc':svm.SVC(probability=True),
                'adbt':ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=3)),
                'bagg':ensemble.BaggingClassifier(n_estimators=100),
                'ext':ensemble.ExtraTreesClassifier(n_estimators=100,max_depth=3),
                #   'perceptron':linear_model.Perceptron(),
                'dt':tree.DecisionTreeClassifier(),
                'knn':neighbors.KNeighborsClassifier(),
                'network':neural_network.MLPClassifier()}
        clfs.update(reDefine_clf) if reDefine_clf else clfs
        if used_clf:
            return dict((key, value) for key, value in clfs.items() if key in used_clf)
        else:
            return clfs
            
    def model_auc(self,clfs,X_array,y_array,contain_train=True,Stratified=True,n_splits=5):
        if Stratified:
            KF = model_selection.StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=6)
        else:
            KF = model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=8)
        auc_dfs = pd.DataFrame()
        for clf_name,clf in clfs.items():
            if contain_train:
                train_auc,test_auc=[],[]
                for train_index, test_index in KF.split(X=X_array, y=y_array):
                    trainX,trainY,testX,testY = X_array[train_index],y_array[train_index],X_array[test_index],y_array[test_index]
                    clf.fit(X=trainX,y=trainY)
                    train_prob = clf.predict_proba(trainX)[:,1]
                    test_prob = clf.predict_proba(testX)[:,1]
                    train_auc.append(metrics.roc_auc_score(trainY,train_prob))
                    test_auc.append(metrics.roc_auc_score(testY,test_prob))
                auc_dfs.loc[clf_name,'train_mean'] = np.array(train_auc).mean()
                auc_dfs.loc[clf_name,'test_mean'] = np.array(test_auc).mean()
                auc_dfs.loc[clf_name,'auc_diff'] = auc_dfs.loc[clf_name,'train_mean'] - auc_dfs.loc[clf_name,'test_mean']
                auc_dfs.loc[clf_name,'train_std'] = np.array(train_auc).std()
                auc_dfs.loc[clf_name,'test_std'] = np.array(test_auc).std()
            else:
                scores = model_selection.cross_val_score(estimator=clf,X=X_array,y=y_array,cv=KF.split(X=X_array, y=y_array),scoring='roc_auc',n_jobs=-1)
                auc_dfs.loc[clf_name,'test_mean'] = scores.mean()
                auc_dfs.loc[clf_name,'test_std'] = scores.std()
        return auc_dfs


def grid_search(clf,param_grid,x_array,y_array,result_df=pd.DataFrame(),socring='roc_auc',Kfold=5):
    '''网格搜索调参，并记录中间结果'''
    KF = model_selection.StratifiedKFold(n_splits=Kfold,random_state=2,shuffle=True)
    gs=model_selection.GridSearchCV(estimator=clf,param_grid=param_grid,scoring=socring,n_jobs=-1,cv=KF.split(x_array,y_array))
    gs.fit(x_array,y_array)
    result_t=pd.DataFrame(gs.cv_results_['params'])
    result_t['mean_test_score']=gs.cv_results_['mean_test_score']
    result_df = pd.concat([result_df,result_t])
    result_df = result_df.sort_values(by=list(result_df.columns[:-1]))
    result_df = result_df.drop_duplicates()
    print('best_score: %s'%gs.best_score_)
    print('best_params:%s'%gs.best_params_)
    return result_df.reset_index(drop=True)

def params_score_plot(result_df,xlim=[0,1]):
    '''网格搜索调参的中间结果绘图（仅满足两个参数）'''
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    y_axis = result_df.iloc[:,0].map(str)
    y_label = result_df.columns[0]
    for i in range(1,result_df.shape[1]-1):
        y_axis = y_axis+' and '+result_df.iloc[:,i].map(str)
        y_label = y_label+' AND '+result_df.columns[i]
    g = sns.barplot(x=result_df['mean_test_score'].values,y=y_axis)
    g.set_xlim(xlim)
    g.set_ylabel(y_label)
    g.set_xlabel('mean_test_score')
    plt.subplot(122)
    g=sns.scatterplot(x=result_df.iloc[:,0].values,y=result_df.iloc[:,1].values,size=result_df['mean_test_score'].values,legend=False)
    g.set_xlabel(result_df.columns[0])
    g.set_ylabel(result_df.columns[1])
    g.set_title('Size=mean_test_score')


# stacking模型融合方法 
def get_oof(clf, x_train, y_train, x_test, k, k_flod, scores=False):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((k, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(k_flod):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)
        if scores:
            oof_train[test_index] = clf.predict_proba(x_te)[:,1]
            oof_test_skf[i,:] = clf.predict_proba(x_test)[:,1]            
        else:
            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i,:] = clf.predict(x_test)            

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def stacking_method(clfs, x_train, y_train, x_test, k, scores=True):
    x_train_Step2, x_test_Step2 =[],[]
    kf = model_selection.StratifiedKFold(n_splits=k,shuffle=True,random_state=2)
    for clf in clfs:
        k_flod = kf.split(x_train,y_train)
        oof_train, oof_test = get_oof(clf, x_train, y_train, x_test, k, k_flod, scores=scores)
        x_train_Step2.append(oof_train)
        x_test_Step2.append(oof_test)
    return np.concatenate(x_train_Step2,axis=1),np.concatenate(x_test_Step2,axis=1)