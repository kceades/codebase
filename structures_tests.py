"""
Author: Kevin Caleb Eades
Created: Spring 2018
"""

# importing the structures file I wrote
import structures

# importing the test case framework
import unittest
# useful import for testing
import random

class LinkedListTests(unittest.TestCase):
	def test_creation(self):
		test = structures.LinkedList([1,2,3,4,5])
		self.assertEqual([1,2,3,4,5],[val for val in test])

	def test_deletion(self):
		test = structures.LinkedList([1,2,3,4,5])
		test.remove(4)
		self.assertEqual([1,2,3,5],[val for val in test])

	def test_reversal(self):
		test = structures.LinkedList([1,2,3,4,5])
		test.reverse()
		self.assertEqual([5,4,3,2,1],[val for val in test])

	def test_itemgetter(self):
		test = structures.LinkedList([1,2,3,4,5])
		self.assertEqual(test[0],1)

class MinHeapTests(unittest.TestCase):
	def test_creation(self):
		testing = [i for i in range(100)]
		random.shuffle(testing)
		test = structures.MinHeap(testing)
		self.assertEqual([i for i in range(100)],[val for val in test])

	def test_keyed(self):
		testing = [('b'*i,i) for i in range(10)]
		random.shuffle(testing)
		test = structures.MinHeap(testing,1)
		self.assertEqual([('b'*i,i) for i in range(10)],[val for val in test])

class MaxHeapTests(unittest.TestCase):
	def test_creation(self):
		testing = [i for i in range(100)]
		random.shuffle(testing)
		test = structures.MaxHeap(testing)
		self.assertEqual([i for i in range(100)][::-1],[val for val in test])

	def test_keyed(self):
		testing = [('b'*i,i) for i in range(10)]
		random.shuffle(testing)
		test = structures.MaxHeap(testing,1)
		self.assertEqual([('b'*i,i) for i in range(10)][::-1],[val \
			for val in test])

if __name__=='__main__':
	unittest.main()