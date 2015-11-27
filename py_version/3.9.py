'''
This code snippet as been taken verbatim from sci-kit learn version 0.15.2.
sklearn/mixture/gmm.py

This code is for illustration only.

All rights are reserved by the original owner/author. 

This code is reproduced here for illlustration only.

Douglas McIlwraith 
Algorithms of the Intelligent Web 2nd Edition
Manning Publications
'''

def fit(self, X):
	"""Estimate model parameters with the expectation-maximization
	algorithm.

	A initialization step is performed before entering the em
	algorithm. If you want to avoid this step, set the keyword
	argument init_params to the empty string '' when creating the
	GMM object. Likewise, if you would like just to do an
	initialization, set n_iter=0.

	Parameters
	----------
	X : array_like, shape (n, n_features)
		List of n_features-dimensional data points.  Each row
		corresponds to a single data point.
	"""
	## initialization step
	X = np.asarray(X, dtype=np.float)
	if X.ndim == 1:
		X = X[:, np.newaxis]
	if X.shape[0] < self.n_components:
		raise ValueError(
			'GMM estimation with %s components, but got only %s samples' %
			(self.n_components, X.shape[0]))

	max_log_prob = -np.infty

	for _ in range(self.n_init):
		if 'm' in self.init_params or not hasattr(self, 'means_'):
			self.means_ = cluster.KMeans(
				n_clusters=self.n_components,
				random_state=self.random_state).fit(X).cluster_centers_

		if 'w' in self.init_params or not hasattr(self, 'weights_'):
			self.weights_ = np.tile(1.0 / self.n_components,
									self.n_components)

		if 'c' in self.init_params or not hasattr(self, 'covars_'):
			cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
			if not cv.shape:
				cv.shape = (1, 1)
			self.covars_ = \
				distribute_covar_matrix_to_match_covariance_type(
					cv, self.covariance_type, self.n_components)

		# EM algorithms
		log_likelihood = []
		# reset self.converged_ to False
		self.converged_ = False
		for i in range(self.n_iter):
			# Expectation step
			curr_log_likelihood, responsibilities = self.score_samples(X)
			log_likelihood.append(curr_log_likelihood.sum())

			# Check for convergence.
			if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
					self.thresh:
				self.converged_ = True
				break

			# Maximization step
			self._do_mstep(X, responsibilities, self.params,
						   self.min_covar)

		# if the results are better, keep it
		if self.n_iter:
			if log_likelihood[-1] > max_log_prob:
				max_log_prob = log_likelihood[-1]
				best_params = {'weights': self.weights_,
							   'means': self.means_,
							   'covars': self.covars_}
	# check the existence of an init param that was not subject to
	# likelihood computation issue.
	if np.isneginf(max_log_prob) and self.n_iter:
		raise RuntimeError(
			"EM algorithm was never able to compute a valid likelihood " +
			"given initial parameters. Try different init parameters " +
			"(or increasing n_init) or check for degenerate data.")
	# self.n_iter == 0 occurs when using GMM within HMM
	if self.n_iter:
		self.covars_ = best_params['covars']
		self.means_ = best_params['means']
		self.weights_ = best_params['weights']
	return self

def _do_mstep(self, X, responsibilities, params, min_covar=0):
	""" Perform the Mstep of the EM algorithm and return the class weihgts.
	"""
	weights = responsibilities.sum(axis=0)
	weighted_X_sum = np.dot(responsibilities.T, X)
	inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

	if 'w' in params:
		self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
	if 'm' in params:
		self.means_ = weighted_X_sum * inverse_weights
	if 'c' in params:
		covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
		self.covars_ = covar_mstep_func(
			self, X, responsibilities, weighted_X_sum, inverse_weights,
			min_covar)
	return weights