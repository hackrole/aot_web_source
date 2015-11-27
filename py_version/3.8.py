from sklearn.mixture import GMM
from sklearn import datasets
from itertools import cycle, combinations
import pylab as pl
import matplotlib as mpl
import numpy as np

# make_ellipses method taken from: http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_classifier.html#example-mixture-plot-gmm-classifier-py
# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# License: BSD 3 clause

def make_ellipses(gmm, ax, x, y):
    for n, color in enumerate('rgb'):
	row_idx = np.array([x,y])
	col_idx = np.array([x,y])
        v, w = np.linalg.eigh(gmm._get_covars()[n][row_idx[:,None],col_idx])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, [x,y]], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

iris = datasets.load_iris()

gmm = GMM(n_components=3,covariance_type='full', n_iter=20)
gmm.fit(iris.data)

predictions = gmm.predict(iris.data)

colors = cycle('rgb')
labels = ["Cluster 1","Cluster 2","Cluster 3"]
targets = range(len(labels))

feature_index=range(len(iris.feature_names))
feature_names=iris.feature_names
combs=combinations(feature_index,2)

f,axarr=pl.subplots(3,2)
axarr_flat=axarr.flat

for comb, axflat in zip(combs,axarr_flat):
 	for target, color, label in zip(targets,colors,labels):
  		feature_index_x=comb[0]
  		feature_index_y=comb[1]
  		axflat.scatter(iris.data[predictions==target,feature_index_x],
				iris.data[predictions==target,feature_index_y],c=color,label=label)
  		axflat.set_xlabel(feature_names[feature_index_x])
  		axflat.set_ylabel(feature_names[feature_index_y])
		make_ellipses(gmm,axflat,feature_index_x,feature_index_y)
pl.show()		
