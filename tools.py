"""
Author: Kevin Caleb Eades
Summer 2017
"""


# imports from standard modules
import numpy as np


class Error(Exception):
	""" Base class for exceptions """
	pass


class OrderError(Error):
	""" Exception raised when the order of things are done incorrectly """
	def __init__(self,expression,message):
		"""
		constructor

		:expression: input expression in which the error occured
		:message: explanation of the error
		"""
		self.expression = expression
		self.message = message


class fitter(object):
	"""
	Class object for a fitter: this will generate the coefficients for fitting
	a vector to an arbitrary set of other vectors
	"""
	def __init__(self):
		""" constructor """
		self.base_vecs = None
		self.inv_vecs = None

	def PrepFit(self,base_vecs):
		"""
		sets up the inverse matrix with base_vecs for use in fitting

		:base_vecs: list of vectors all of the same length
		"""
		num_vecs = len(base_vecs)

		coeff_arr = np.zeros((num_vecs,num_vecs))
		for i in range(num_vecs):
			for j in range(i):
				coeff_arr[i][j] = coeff_arr[j][i]
			for j in range(i,num_vecs):
				coeff_arr[i][j] = np.dot(base_vecs[i],base_vecs[j])

		self.inv_vecs = np.linalg.inv(coeff_arr)
		self.base_vecs = base_vecs

	def Fit(self,vec):
		"""
		does the actual fitting and returns the coefficients

		:vec: vector of the same length as all vectors in self.base_vecs

		:returns: the coefficients of the fit
		"""
		if self.base_vecs is None or self.inv_vecs is None:
			raise OrderError('Fit','Invalid Order: Run fitter.PrepFit before' \
				+ ' running fitter.Fit')

		final = [np.dot(vec,base_vec) for base_vec in self.base_vecs]

		return np.transpose(np.dot(self.inv_vecs,np.transpose(final)))

	def Reduce(self,vecs):
		"""
		returns the reduced vectors of vecs after being fit to self.base_vecs

		:vecs: list of vectors

		:returns: list of coefficient vectors
		"""
		if self.base_vecs is None or self.inv_vecs is None:
			raise OrderError('Fit','Invalid Order: Run fitter.PrepFit before' \
				+ ' running fitter.Reduce')

		r_arr = []
		for vec in vecs:
			coeffs = self.Fit(vec)
			r_arr.append(np.subtract(vec,np.dot(coeffs,self.base_vecs)))

		return r_arr


def ExclusiveIn(clist,masterlist):
	"""
	checks if every element in clist is in masterlist

	:clist: list of elements
	:masterlist: list of elements

	:returns: True or False
	"""
	if clist==[]:
		return True
	else:
		for elem in clist:
			if elem not in masterlist:
				return False
		return True

def RemoveDash(string):
	""" 
	removes the dash from the end of a string if one is there

	:string: an input string

	:returns: a string
	"""
	if string[-1]=='-':
		return string[:-1]
	else:
		return string

def ClosestMatch(num,num_list):
	"""
	returns the closest number to num in num_list

	:num: a number, integer or float
	:num_list: a list of numbers, again integers or floats
	"""
	diffs = np.abs(np.subtract(num,num_list))
	return num_list[np.argmin(diffs)]

def FitRMS(a,b,length):
	"""
	returns the RMS of the difference between vectors a and b with len length

	:a: vector
	:b: vector
	:length: integer, the length of a and b
	"""
	return np.sqrt(np.sum(np.power(np.subtract(a,b),2))/length)

def RMS(a,length):
	"""
	returns the RMS of a vector

	:a: vector
	:length: integer, the length of the vector
	"""
	return np.sqrt(np.sum(a)/length)

def FitChi(fit,data,errs,dof):
	"""
	returns the chi-square when fitting data with fit while having
	uncertainties errs and dof degrees of freedom (note fit, data and errs
	all have to be of uniform length)

	:fit: vector
	:data: vector
	:errs: vector
	:dof: integer, the number of degrees of freedom
	"""
	squared_err = np.power(np.subtract(fit,data),2)
	return np.sum(np.divide(squared_err,np.power(errs,2)))/dof