"""
Written by Kevin Caleb Eades
Fall 2017

The purpose of this file is to have a bunch of machine learning algorithms that
can be used as regressors for a prediction (of supernova phases in my case but
could be generalized) that can be combined in a meaningful way.
"""


# imports from files I wrote
import tools

# imports
import numpy as np
import random
import matplotlib.pyplot as plt



class combiner(object):
	""" class for combining various predictions from the ML algorithms """
	def __init__(self):
		""" constructor """
		self.classifiers = {}
		self.predictions = {}
		self.num_predictors = 0

		self.final = {}

	def __repr__(self):
		""" the representation of this object """
		if self.final=={}:
			return "Combiner object with no combined prediction yet."
		return \
		"""Combiner object with the following combined predictions:
		-- Uniform Weighted (averaging with uniform weighting):
			-- mean:  {:.3f}
			-- sigma: {:.3f}
		-- Error Weighted (averaging with inverse error weighting):
			-- mean:  {:.3f}
			-- sigma: {:.3f}
		""".format(self.final['Uniform Weighted'].pred,self.final\
			['Uniform Weighted'].sigma,self.final['Error Weighted'].pred\
			,self.final['Error Weighted'].sigma)

	def __str__(self):
		""" the string representation of this object """
		if self.final=={}:
			return "Combiner object with no combined prediction yet."
		return \
		"""Combiner object with the following combined predictions:
		-- Uniform Weighted (averaging with uniform weighting):
			-- mean:  {:.3f}
			-- sigma: {:.3f}
		-- Error Weighted (averaging with inverse error weighting):
			-- mean:  {:.3f}
			-- sigma: {:.3f}
		""".format(self.final['Uniform Weighted'].pred,self.final\
			['Uniform Weighted'].sigma,self.final['Error Weighted'].pred\
			,self.final['Error Weighted'].sigma)

	def AddPredictor(self,name,predictor):
		"""
		adds a predictor with a MakePrediction method

		:name: the name of the predictor
		:predictor: the predictor's class object
		"""
		self.classifiers[name] = predictor
		self.num += 1
		self.predictions[name] = None

	def MakePrediction(self,in_data):
		"""
		makes a prediction for the phase of the input flux data

		:in_data: (float array) list of flux values
		"""
		for name in self.classifiers:
			self.predictions[name] = self.classifiers[name]\
				.MakePrediction(in_data)
		self.UniformWeighted()
		self.ErrorWeighted()

	def UniformWeighted(self):
		""" adds an averaged prediction to the final list """
		mean = np.mean([self.predictions[name].pred for name \
			in self.predictions])
		stds = [self.predictions[name].sigma for name in self.predictions]
		sigma = None
		if None not in stds:
			sigma = np.sum(np.power(sigmas,2))
			sigma = np.sqrt(sigma)/self.num
		self.final['Uniform Weighted'] = prediction(mean,sigma)

	def ErrorWeighted(self):
		""" adds an error weighted prediction to the final list """
		sigmas = [self.predictions[name].sigma for name in self.predictions]
		if None in sigmas:
			self.final['Error Weighted'] = None
			return
		weights = 0
		total = 0
		for name in self.predictions:
			total += self.predictions[name].pred/self.predictions[name].sigma**2
			weights += 1/self.predictions[name].sigma**2
		mean = total/weights
		weighted = np.multiply(sigmas,np.divide(np.divide(1,np.power(sigmas,2))\
			,weights))
		sigma = np.sqrt(np.sum(np.power(weighted,2)))
		self.final['Error Weighted'] = prediction(mean,sigma)



class regressionTree(object):
	""" class for building a single regression tree """
	def __init__(self,min_nodes,min_gain):
		"""
		constructor

		:min_nodes: (int) minimum number of data points to have in a leaf of 
					the regression tree
		:min_gain: (float) minimum amount of information gain to have when
					making a split in the tree
		"""
		self.min_nodes = min_nodes
		self.min_gain = min_gain

		self.leaf = False

	def Train(self,data,indices):
		"""
		trains the regression tree
		"""
		pass
		# TODO



class prediction(object):
	""" prediction object useful for combining machine learning techniques """
	def __init__(self,pred,sigma):
		"""
		constructor

		:pred: (float) the prediction from a ML algorithm
		:sigma: (float) the standard deviation or uncertainty on the prediction
		"""
		self.pred = pred
		self.sigma = sigma

	def __repr__(self):
		""" the representation of the object """
		return \
		"""Prediction object with:
		-- mean:  {:.3f}
		-- sigma: {:.3f}
		""".format(self.pred,self.sigma)

	def __str__(self):
		""" the string representation of the object """
		return \
		"""Prediction object with:
		-- mean:  {:.3f}
		-- sigma: {:.3f}
		""".format(self.pred,self.sigma)



class linreg(object):
	""" Linear regression object """
	def __init__(self):
		""" constructor """
		self.a = None
		self.b = None
		self.delta_a = None
		self.delta_b = None

	def Train(self,data):
		"""
		trains the linear regression based on the data

		:data: (float array) list of (x,y,err=None) tuples where err is optional
		"""
		self.x = [point[0] for point in data]
		self.y = [point[1] for point in data]
		self.errs = None
		if len(data[0])>2:
			self.errs = [point[2] for point in data]
		self.num = len(data)
		x_bar = np.sum(self.x)/self.num
		y_bar = np.sum(self.y)/self.num
		xy_bar = np.sum(np.multiply(self.x,self.y))/self.num
		x2_bar = np.sum(np.power(self.x,2))/self.num
		self.a = (xy_bar - y_bar*x_bar)/(x2_bar - x_bar**2)
		self.b = y_bar - self.a*x_bar

		if self.errs is not None:
			err_bar = np.sum(self.errs)/self.num
			xerr_bar = np.sum(np.multiply(self.x,self.errs))/self.num
			self.delta_a = (xerr_bar-x_bar*err_bar)/(x2_bar-x_bar**2)
			self.delta_b = err_bar-x_bar*self.delta_a
	
	def Fit(self,point):
		"""
		fits a new data point to the linear regression and returns the y value

		:point: (float) x value

		:returns: prediction object with the mean and uncertainty
		"""
		mean = self.a*point + self.b
		sigma = None
		if self.delta_a is not None:
			sigma = self.delta_a*point + self.delta_b
		return prediction(mean,sigma)



class genreg(object):
	""" general regression object for general independent variables """
	def __init__(self):
		""" Constructor	"""
		self.a = None
		self.b = None

	def Train(self,data):
		"""
		trains the multiple linear regression based on the data

		:data: list of (xlist,y) tuples, where xlist is a list of independent 
		       variable values
		"""
		self.n = len(data[0][0])
		self.x = {i:[d[0][i] for d in data] for i in range(self.n)}
		self.y = [d[1] for d in data]
		M = np.zeros((self.n,self.n))
		B = np.zeros((self.n,1))
		y_bar = np.mean(self.y)
		for row in range(self.n):
			xr_bar = np.mean(self.x[row])
			yxr_bar = np.mean(np.multiply(self.y,self.x[row]))
			B[row][0] = y_bar*xr_bar - yxr_bar
			for col in range(row,self.n):
				xc_bar = np.mean(self.x[col])
				xrxc_bar = np.mean(np.multiply(self.x[row],self.x[col]))
				M[row][col] = xr_bar*xc_bar - xrxc_bar
			for col in range(row):
				M[row][col] = M[col][row]
		M_inv = np.linalg.inv(M)
		a_col = np.dot(M_inv,B)
		self.a = [a_col[row][0] for row in range(self.n)]
		self.b = y_bar
		for row in range(self.n):
			xr_bar = np.mean(self.x[row])
			self.b = self.b - self.a[row]*xr_bar

	def Fit(self,point):
		"""
		fts a new data point to the multiple linear regression and returns the
		y value

		:point: a list of x values

		:returns: the fit y value
		"""
		return np.dot(self.a,point) + self.b



class knn(object):
	""" knn classification object """
	def __init__(self,k):
		"""
		constructor

		:k: integer, the number of neighbors to look at
		"""
		self.k = k

	def Train(self,data):
		"""
		trains the knn algorithm with the training data

		:data: (tuple list) list of (key,xlist,y) tuples, where key is the
				supernova key, xlist is a list or array of independent variable
				values, and y is the output
		"""
		self.data = data
		self.num_variables = len(self.data[0][1])

	def Metric(self,point):
		"""
		evaluates the cost function of SSE for all the training data

		:point: a list of x values

		:returns: tupled list of (metric,y) values
		"""
		return [(x[0],tools.FitRMS(point,x[1],self.num_varibles),x[2]) \
			for x in self.data]

	def Fit(self,point):
		"""
		makes the prediction

		:point: list of x values

		:returns: the predicted y value based on the mean of the self.k nearest
		          neighbors
		"""
		added = 0
		neighbors = []
		added_keys = []
		m = self.Metric(point)
		m.sort()
		index = 0
		while added<self.k:
			if m[index][0] not in added_keys:
				added += 1
				added_keys.append(m[index][0])
				neighbors.append(m[index][2])
			index += 1
		return np.mean([n[1] for n in neighbors])



class net(object):
	""" neural net regression object (not a classifier) """
	def __init__(self,sizes,min_y,max_y):
		"""
		constructor

		:sizes: (float array) [data features, first layer nodes, second, ..., 1]
		:min_y: (float) minimum y value
		:max_y: (float) maximum y value
		"""
		self.sizes = sizes
		self.layers = len(sizes)
		self.min_y = min_y
		self.max_y = max_y
		self.biases = [np.random.randn(self.sizes[layer]) for layer \
			in range(1,self.layers)]
		self.weights = [np.random.randn(self.sizes[layer],self.sizes[layer-1]) \
			for layer in range(1,self.layers)]

	def Sigmoid(self,z,scale=None):
		"""
		sigmoid function

		:z: (float or float array)
		:scale: (float) (optional) a scaling to dial back the z values
		    	i.e., if there are a large number of features

		:returns: (float) sigmoid(z)
		"""
		if scale is not None:
			z = np.divide(z,scale)
		return np.divide(1,np.add(1,np.exp(-z)))

	def SigmoidPrime(self,z,scale=None):
		"""
		derivative of the sigmoid function

		:z: (float or float array)
		:scale: (float) (optional) a scaling to dial back the z values
		    	(i.e., if there are a large number of features)

		:returns: derivative of the sigmoid function
		"""
		return np.multiply(self.Sigmoid(z,scale),np.subtract(1\
			,self.Sigmoid(z,scale)))

	def Final(self,activation):
		"""
		turns the final activation into a predicted y value

		:activation: (float) final neuron output (between 0 and 1)

		:returns: (float) the predicted y value
		"""
		return self.min_y + (self.max_y-self.min_y)*activation

	def Fit(self,a):
		"""
		returns the output of the neural net if a is the input

		:a: (float array) initial input to the net

		:returns: (float) predicted y value from the net
		"""
		for layer in range(self.layers-1):
			a = self.Sigmoid(np.add(np.dot(self.weights[layer],np.transpose(a))\
				,self.biases[layer]))
		return self.Final(a)[0]

	def Evaluate(self,data):
		"""
		evaluates the data on the net and returns the mean and std of
		the results

		:data: (float array) list of (a,output) tuples with a being a vector
				of features

		:returns: (float tuple) (mean,std) of the difference prediction-real
					 for the y value
		"""
		results = [self.Fit(a)-output for a,output in data]
		return (np.mean(results),np.std(results))

	def Train(self,data,epochs,batch_size,eta):
		"""
		training the neural net with partial progress printed. Training uses
		stochastic gradient descent.

		:data: list of [(features,output)] where features is a list
		:epochs: the number of iterations to train through
		:batch_size: the size of the batch to use
		:eta: the training rate
		"""
		num = len(data)
		for epoch in range(epochs):
			random.shuffle(data)
			batches = [data[k:k+batch_size] for k in range(0,num,batch_size)]
			for batch in batches:
				self.UpdateBatch(batch,batch_size,eta)
			results = self.Evaluate(data)
			print('Epoch {0} complete'.format(epoch))
			print('Avg. {:.3f}'.format(results[0]))
			print('Std. {:.3f}'.format(results[1]))
			print('!!!!!!!!!!!!!!!!!')

	def UpdateBatch(self,batch,batch_size,eta):
		"""
		update the neural net's weights and biases by applying gradient descent
		using backpropagation to a single batch

		:batch: list of [(features,output)] where features is a list
		:batch_size: size of the batch
		:eta: the training rate
		"""
		nabla_b = [np.zeros(self.sizes[layer]) for layer \
			in range(1,self.layers)]
		nabla_w = [np.zeros([self.sizes[layer],self.sizes[layer-1]]) for layer \
			in range(1,self.layers)]
		for a,output in batch:
			delta_nabla_b,delta_nabla_w = self.BackPropagation(a,output)
			nabla_b = [np.add(nabla_b[layer],delta_nabla_b[layer]) for layer \
				in range(self.layers-1)]
			nabla_w = [np.add(nabla_w[layer],delta_nabla_w[layer]) for layer \
				in range(self.layers-1)]
		self.biases = [np.subtract(self.biases[layer],np.multiply(eta\
			/batch_size,nabla_b[layer])) for layer in range(self.layers-1)]
		self.weights = [np.subtract(self.weights[layer],np.multiply(eta\
			/batch_size,nabla_w[layer])) for layer in range(self.layers-1)]

	def BackPropagation(self,a,output):
		"""
		returns the gradient of the cost function

		:a: input features vector
		:output: the y value corresponding to the input vector a

		:returns: tuple (nabla_b,nabla_w)
		"""
		nabla_b = [np.zeros(self.sizes[layer]) for layer \
			in range(1,self.layers)]
		nabla_w = [np.zeros([self.sizes[layer],self.sizes[layer-1]]) for layer \
			in range(1,self.layers)]
		# feedforward
		activations = [a] # list to store activations, layer by layer
		zs = [a] # list to store the a vectors, layer by layer
		for layer in range(1,self.layers):
			z = np.add(np.dot(self.weights[layer-1]\
				,np.transpose(activations[layer-1])),self.biases[layer-1])
			zs.append(z)
			activations.append(self.Sigmoid(z))
		activations[self.layers-1] = self.Final(activations[self.layers-1])
		nabla_b[self.layers-2] = np.multiply(np.subtract(activations\
			[self.layers-1],output),self.SigmoidPrime(zs[self.layers-1]))
		nabla_w[self.layers-2] = np.array([np.multiply(nabla_b[self.layers-2]\
			[node],activations[self.layers-2]) for node \
			in range(self.sizes[self.layers-1])])
		# in the loop below, l=2 represents the second to last layer of neurons,
		# l=3 is the thir-last, et cetera
		for l in range(3,self.layers+1):
			nabla_b[self.layers-l] = np.multiply(np.dot(nabla_b[self.layers-l\
				+1],self.weights[self.layers-l+1]),self.SigmoidPrime(zs\
				[self.layers-l+1]))
			nabla_w[self.layers-l] = np.array([np.multiply(nabla_b[self.layers\
				-l][node],activations[self.layers-l]) for node \
				in range(self.sizes[self.layers-l+1])])
		return (nabla_b,nabla_w)