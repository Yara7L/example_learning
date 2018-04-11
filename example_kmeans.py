<<<<<<< HEAD
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs

X,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=9)
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()

from sklearn.cluster import KMeans
from sklearn import metrics

# cluster=2
y_pred=KMeans(n_clusters=2,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

# k=3
y_pred=KMeans(n_clusters=3,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

# k=4
y_pred=KMeans(n_clusters=4,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

from sklearn.cluster import MiniBatchKMeans
y_pred=MiniBatchKMeans(n_clusters=4,batch_size=200,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
=======
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs

X,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=9)
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()

from sklearn.cluster import KMeans
from sklearn import metrics

# cluster=2
y_pred=KMeans(n_clusters=2,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

# k=3
y_pred=KMeans(n_clusters=3,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

# k=4
y_pred=KMeans(n_clusters=4,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

from sklearn.cluster import MiniBatchKMeans
y_pred=MiniBatchKMeans(n_clusters=4,batch_size=200,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
>>>>>>> ac68293c7cab1c17b997ab83e250a9d2f4f1ba1d
print(metrics.calinski_harabaz_score(X,y_pred))