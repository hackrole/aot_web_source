import numpy as np
import pylab as pl

from sklearn import decomposition
from sklearn import datasets
from itertools import cycle

iris = datasets.load_iris()
X = iris.data
Y = iris.target

targets = range(len(iris.target_names))
colors = cycle('rgb')

pca = decomposition.PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)

for target,color in zip(targets,colors):
	pl.scatter(X[Y==target,0],X[Y==target,1],label=iris.target_names[target],c=color)
pl.legend()
pl.show()

