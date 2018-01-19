"""
Author: Kevin Caleb Eades
Created: Fall 2017

Available classes:
	Base classes:
		-- Node
		-- DoubleNode
		-- BSTNode
		-- AVLNode
		-- TrieNode
	Data structures:
		-- DoubleLinkedList
		-- LinkedList
		-- Stack
		-- Queue
		-- PriorityQueue
		-- Trie
		-- BST
		-- AVL
		-- MinHeap
		-- MaxHeap
		-- HashTable
"""

class Node(object):
	"""
	General Node object which can be used for linked lists and simply stores
	some information as well as a pointer to the next node in the list.
	"""
	def __init__(self,value=None,**kwargs):
		"""
		constructor

		:value: (any object) what to store in the node, defaults to None
		**kwargs -- available are:
			'val' - object to store in the node
			'next' - pointer to the next Node
		"""
		self.value = value
		self.next = None
		for key,val in kwargs.items():
			if key=='val':
				self.value = val
			elif key=='next':
				self.next = val

class DoubleNode(object):
	"""
	Doubly linked Node object for use with doubly linked lists.
	"""
	def __init__(self,value=None,**kwargs):
		"""
		constructor

		:value: (any object) what to store in the node, defaults to None
		**kwargs -- available are:
			'val' - object to store in the node
			'prev' - pointer to the previous DoubleNode
			'next' - pointer to the next DoubleNode
		"""
		self.value = value
		self.prev = None
		self.next = None
		for key,val in kwargs.items():
			if key=='val':
				self.value = val
			elif key=='prev':
				self.prev = val
			elif key=='next':
				self.next = val

class BSTNode(object):
	"""
	Binary Search Tree Node object for use with binary search trees.
	"""
	def __init__(self,value=None,**kwargs):
		"""
		constructor

		:value: (any object) what to store in the node
		**kwargs -- available are:
			'val' - object to store in the node
			'left' - pointer to the left BSTNode
			'right' - pointer to the right BSTNode
		"""
		self.value = value
		self.left = None
		self.right = None
		for key,val in kwargs.items():
			if key=='val':
				self.value = val
			elif key=='left':
				self.left = val
			elif key=='right':
				self.right = val

class AVLNode(object):
	"""
	Adelson-Velskii and Landis (AVL) Node object for use in an AVL tree.
	"""
	def __init__(self,value=None,**kwargs):
		"""
		constructor

		:value: (any object) what to store in the node
		**kwargs -- available are:
			'val' - object to store in the node
			'left' - pointer to the left AVLNode
			'right' - pointer to the right AVLNode
			'height' - height of the node
		"""
		self.value = value
		self.left = None
		self.right = None
		self.height = 0
		for key,val in kwargs.items():
			if key=='val':
				self.value = val
			elif key=='left':
				self.left = val
			elif key=='right':
				self.right = val
			elif key=='height':
				self.height = val

class TrieNode(object):
	"""
	Trie tree Node for use with Trie trees.
	"""
	def __init__(self,value=0,**kwargs):
		"""
		constructor

		:value: (integer) the number of strings that end at this node
		**kwargs -- available are:
			'val' - (integer) the number of strings that end at this node
		"""
		self.value = value
		self.children = {}
		for key,val in kwargs.items():
			if key=='val':
				self.value = val



################################################################################
################################################################################
################################################################################

class DoubleLinkedList(object):
	def __init__(self,values=None):
		self.head = None
		self.tail = None
		self._stored = 0
		if isinstance(values,list):
			for val in values:
				self.insert(val)

	def __str__(self):
		if self._stored==0:
			return ''
		else:
			x = self.head
			string = '{} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{} '.format(x.value)
			return string

	def __iter__(self):
		self._current = self.head
		return self

	def __next__(self):
		if self._current is None:
			raise StopIteration
		x = self._current
		self._current = self._current.next
		return x.value

	def __len__(self):
		return self._stored

	def __contains__(self,key):
		x = self.head
		while x is not None:
			if x.value==key:
				return True
			x = x.next
		return False

	def __getitem__(self,key):
		if isinstance(key,int):
			if key>=self._stored or key<0:
				raise IndexError('{} not a valid index'.format(key))
			else:
				count = 0
				x = self.head
				while count<key:
					x = x.next
				return x.value
		else:
			raise KeyError('{} not an integer to index the list by. ' \
				+ 'Use "{} in LinkedList" to see if an element is in it.'\
				.format(key,key))

	def isempty(self):
		return self._stored==0

	def insert(self,value):
		if self._stored==0:
			self.head = DoubleNode(value)
		elif self._stored==1:
			self.tail = DoubleNode(value,prev=self.head)
			self.head.next = self.tail
		else:
			self.tail.next = DoubleNode(value,prev=self.tail)
			self.tail = self.tail.next
		self._stored += 1

	def remove(self,value):
		x = self.head
		while x is not None:
			if x.value==value:
				break
			x = x.next
		if x is None:
			return
		elif self._stored==1:
			self.head = None
		elif x.prev is None:
			self.head = x.next
		else:
			x.prev.next = x.next
			x.next.prev = x.prev
		if self._stored==2:
			self.tail = None
		self._stored -= 1

	def reverse(self):
		if self._stored<=1:
			return
		self.tail = self.head
		previous_node = None
		current_node = self.head
		while current_node.next is not None:
			next_node = current_node.next
			current_node.next = previous_node
			current_node.prev = next_node
			previous_node = current_node
			current_node = next_node
		self.head = current_node
		self.head.next = previous_node
		self.head.prev = None

################################################################################
################################################################################
################################################################################

class LinkedList(object):
	def __init__(self,values=None):
		self.head = None
		self.tail = None
		self._stored = 0
		if isinstance(values,list):
			for val in values:
				self.insert(val)

	def __str__(self):
		if self._stored==0:
			return ''
		else:
			x = self.head
			string = '{} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{} '.format(x.value)
			return string

	def __iter__(self):
		self._current = self.head
		return self

	def __next__(self):
		if self._current is None:
			raise StopIteration
		x = self._current
		self._current = self._current.next
		return x.value

	def __len__(self):
		return self._stored

	def __contains__(self,key):
		x = self.head
		while x is not None:
			if x.value==key:
				return True
			x = x.next
		return False

	def __getitem__(self,key):
		if isinstance(key,int):
			if key>=self._stored or key<0:
				raise IndexError('{} not a valid index'.format(key))
			else:
				count = 0
				x = self.head
				while count<key:
					x = x.next
				return x.value
		else:
			raise KeyError('{} not an integer to index the list by. ' \
				+ 'Use "{} in LinkedList" to see if an element is in it.'\
				.format(key,key))

	def isempty(self):
		return self._stored==0

	def insert(self,value):
		if self._stored==0:
			self.head = Node(value)
		elif self._stored==1:
			self.tail = Node(value)
			self.head.next = self.tail
		else:
			self.tail.next = Node(value)
			self.tail = self.tail.next
		self._stored += 1

	def remove(self,value):
		prev = None
		x = self.head
		while x is not None:
			if x.value==value:
				break
			prev = x
			x = x.next
		if x is None:
			raise KeyError('{} not in LinkedList'.format(value))
		elif self._stored==1:
			self.head = None
		elif prev is None:
			self.head = x
		else:
			prev.next = x.next
		if self._stored==2:
			self.tail = None
		self._stored -= 1

	def reverse(self):
		if self._stored<=1:
			return
		self.tail = self.head
		previous_node = None
		current_node = self.head
		while current_node.next is not None:
			next_node = current_node.next
			current_node.next = previous_node
			previous_node = current_node
			current_node = next_node
		self.head = current_node
		self.head.next = previous_node

################################################################################
################################################################################
################################################################################

class Stack(object):
	def __init__(self,values=None):
		self.contents = []
		if isinstance(values,list):
			self.contents = values

	def __len__(self):
		return len(self.contents)

	def isempty(self):
		return self.contents==[]

	def pop(self):
		if self.contents==[]:
			raise IndexError('Unable to pop: empty stack.')
		self.contents.pop()

	def push(self,value):
		self.contents.append(value)

	def reverse(self):
		self.contents = self.contents[::-1]

################################################################################
################################################################################
################################################################################

class Queue(object):
	def __init__(self,values=None):
		self.contents = None
		if isinstance(values,list):
			self.contents = DoubleLinkedList(values)
		else:
			self.contents = DoubleLilnkedList()

	def __len__(self):
		return len(self.contents)

	def isempty(self):
		return self.contents.isempty()

	def enqueue(self,value):
		self.contents.insert(value)

	def dequeue(self):
		dequeued = self.contents.head.value
		self.contents.remove(self.contents[0])
		return dequeued

	def reverse(self):
		self.contents.reverse()

################################################################################
################################################################################
################################################################################

class PriorityQueue(object):
	def __init__(self,values=None,key=None):
		self.contents = MaxHeap()
		self.key = 0
		if isinstance(key,int):
			self.key = key
		if isinstance(values,list):
			if isinstance(values[0],tuple):
				if self.key<0 or self.key>=len(values[0]):
					self.key = 0
				if isinstance(values[0][self.key],(int,float,str)):
					for i in range(len(values)):
						if type(values[i][self.key])!=type(values[0][self.key]):
							raise TypeError('Inconsistent key typing.')
					self.contents.heapify(values,key)
				else:
					raise TypeError('Priority key must be of type int, float ' \
						+ 'or str.')
			else:
				raise TypeError('Must insert tuple or list of tuples into ' \
					+ 'priority queue.')
		elif isinstance(values,tuple):
			if self.key<0 or self.key>=len(values):
				self.key = 0
			self.contents.key = self.key
			if isinstance(values[self.key],(int,float,str)):
				self.contents.push(values)
			else:
				raise TypeError('Priority key must be of type int, float ' \
					+ 'or str.')
		else:
			if values is not None:
				raise TypeError('Must insert tuple or list of tuples into ' \
					+ 'priority queue.')

	def __len__(self):
		return len(self.contents)

	def isempty(self):
		return self.contents.isempty()

	def enqueue(self,value):
		if isinstance(value,tuple) and len(value)>self.key:
			if len(self.contents)>0 and type(self.contents.peek()[self.key])\
				!=type(value[self.key]):
				raise TypeError('Inconsistent key typing.')
			else:
				self.contents.push(value)
		else:
			if len(value)<self.key:
				raise TypeError('Inconsistent tuple length, unable to process.')
			else:
				raise TypeError('Must insert tuples into the priority queue.')

	def dequeue(self):
		return self.contents.pop()


################################################################################
################################################################################
################################################################################

class Trie(object):
	def __init__(self,strings=None):
		self.root = None
		if strings is not None:
			self.insert(strings)

	def isempty(self):
		return self.root.children=={}

	def wipe(self):
		self.root = TrieNode()

	def insert(self,strings):
		if isinstance(strings,list):
			for string in strings:
				self.insert(string)
		elif isinstance(strings,str) and len(strings)>=1:
			self._insertstring(strings)
		else:
			raise TypeError('Must insert strings or lists of strings into ' \
				+ 'the Trie tree.')

	def _insertstring(self,string):
		if self.root is None:
			self.root = TrieNode()
		x = self.root
		for char in string:
			if char not in x.children:
				x.children[char] = TrieNode()
			x = x.children[char]
		x.value += 1

	def topwords(self,partial='',k=3):
		x = self.root
		suggestions = []
		for char in partial:
			if char not in x.children:
				return suggestions
			x = x.children[char]
		queued = Queue()
		queued.enqueue((x,partial))
		minned = MinHeap()
		while queued.isempty() is False:
			current = queued.dequeue()
			if current[0].value>0 and len(minned)<k:
				minned.push((current[0].value,current[1]))
			elif current[0].value>0 and current[0].value>minned.peek()[0]:
				minned.pop()
				minned.push((current[0],value,current[1]))
			for char in current[0].children:
				queued.enqueue((current[0].children[char],current[1]+char))
		for i in range(k):
			suggestions.append(minned.pop()[1])
		return suggestions[::-1]

################################################################################
################################################################################
################################################################################

class BST(object):
	def __init__(self,values=None):
		self.root = None
		self._stored = 0
		if values is not None:
			self.insert(values)

	def __len__(self):
		return self._stored

	def __str__(self):
		return ' '.join(self.inorder())

	def __getitem__(self,key):
		if not isinstance(key,int):
			raise KeyError('Invalid key: input an integer tree index.')
		elif key>=self._stored or key<0:
			raise IndexError('Index out of range.')
		else:
			return self.inorder()[key]

	def __iter__(self):
		self._iteratingvalues = self.inorder()
		self._currentiteration = 0
		return self

	def __next__(self):
		if self._currentiteration>=self._stored:
			raise StopIteration
		self._currentiteration += 1
		return self._iteratingvalues[self._currentiteration-1]

	def postorder(self):
		traversal = []
		self._postorderhelper(self.root,traversal)
		return traversal

	def _postorderhelper(self,node,visited):
		if node is None:
			return
		self._postorderhelper(node.left,visited)
		self._postorderhelper(node.right,visited)
		visited.append(node.value)

	def preorder(self):
		traversal = []
		self._preorderhelper(self.root,traversal)
		return traversal

	def _preorderhelper(self,node,visited):
		if node is None:
			return
		visited.append(node.value)
		self._preorderhelper(node.left,visited)
		self._preorderhelper(node.right,visited)

	def inorder(self):
		traversal = []
		self._inorderhelper(self.root,traversal)
		return traversal

	def _inorderhelper(self,node,visited):
		if node is None:
			return
		self._inorderhelper(node.left,visited)
		visited.append(node.value)
		self._inorderhelper(node.right,visited)

	def breadthfirst(self):
		visited = []
		if self._stored==0:
			return visited
		queued = Queue()
		queued.enqueue(self.root)
		while queued.isempty() is False:
			popped = queued.dequeue()
			visited.append(popped.value)
			if popped.left is not None:
				queued.enqueue(popped.left)
			if popped.right is not None:
				queued.enqueue(popped.right)
		return visited

	def isempty(self):
		return self._stored==0

	def insert(self,values):
		if isinstance(values,list):
			self._sortedinsert(values)
		else:
			self._singleinsert(values)

	def _singleinsert(self,value):
		if not isinstance(value,(int,float)):
			raise TypeError('Must insert integers or floats into BST')
		elif self.root is None:
			self.root = BSTNode(value)
		else:
			x = self.root
			while True:
				if value<=x.value:
					if x.left is None:
						x.left = BSTNode(value)
						break
					else:
						x = x.left
				else:
					if x.right is None:
						x.right = BSTNode(value)
					else:
						x = x.right
		self._stored += 1

	def _sortedinsert(self,values):
		self._sortedinserthelper(values,0,len(values)-1)

	def _sortedinserthelper(self,values,start,end):
		mid = int((start+end)/2)
		self._singleinsert(values[mid])
		if start<=(mid-1):
			self._sortedinserthelper(values,start,mid-1)
		if (mid+1)<=end:
			self._sortedinserthelper(values,mid+1,end)

	def remove(self,value,head=None,prev=None,path=-1):
		if self._stored==0:
			raise KeyError('{} not in empty BST.'.format(value))
		x = self.head
		prev = prev
		if head is not None:
			x = head
		while x is not None:
			if x.value==value:
				if x.left is None and x.right is None:
					if path==-1:
						self.root = None
					elif path==0:
						prev.left = None
					else:
						prev.right = None
				elif x.left is None:
					if path==-1:
						self.root = x.right
					elif path==0:
						prev.left = x.right
					else:
						prev.right = x.right
				elif x.right is None:
					if path==-1:
						self.root = x.left
					elif path==0:
						prev.left = x.left
					else:
						prev.right = x.left
				else:
					temp = self._largest(x.left)
					self.remove(temp,x.left,x,0)
					x.value = temp
					self._stored += 1
				self._stored -= 1
				return
			prev = x
			if value<x.value:
				x = x.left
				path = 0
			else:
				x = x.right
				path = 1
		raise KeyError('{} not in BST.'.format(value))

	def _largest(self,head):
		x = head
		while x.right is not None:
			x = x.right
		return x.value

################################################################################
################################################################################
################################################################################

class AVL(object):
	def __init__(self,values=None):
		self.root = None
		self._stored = 0
		if values is not None:
			self.insert(values)

	def __len__(self):
		return self._stored

	def __str__(self):
		return ' '.join(self.inorder())

	def __getitem__(self,key):
		if not isinstance(key,int):
			raise KeyError('Invalid key: input an integer tree index.')
		elif key>=self._stored or key<0:
			raise IndexError('Index out of range.')
		else:
			return self.inorder()[key]

	def __iter__(self):
		self._iteratingvalues = self.inorder()
		self._currentiteration = 0
		return self

	def __next__(self):
		if self._currentiteration>=self._stored:
			raise StopIteration
		self._currentiteration += 1
		return self._iteratingvalues[self._currentiteration-1]

	def postorder(self):
		traversal = []
		self._postorderhelper(self.root,traversal)
		return traversal

	def _postorderhelper(self,node,visited):
		if node is None:
			return
		self._postorderhelper(node.left,visited)
		self._postorderhelper(node.right,visited)
		visited.append(node.value)

	def preorder(self):
		traversal = []
		self._preorderhelper(self.root,traversal)
		return traversal

	def _preorderhelper(self,node,visited):
		if node is None:
			return
		visited.append(node.value)
		self._preorderhelper(node.left,visited)
		self._preorderhelper(node.right,visited)

	def inorder(self):
		traversal = []
		self._inorderhelper(self.root,traversal)
		return traversal

	def _inorderhelper(self,node,visited):
		if node is None:
			return
		self._inorderhelper(node.left,visited)
		visited.append(node.value)
		self._inorderhelper(node.right,visited)

	def breadthfirst(self):
		visited = []
		if self._stored==0:
			return visited
		queued = Queue()
		queued.enqueue(self.root)
		while queued.isempty() is False:
			popped = queued.dequeue()
			visited.append(popped.value)
			if popped.left is not None:
				queued.enqueue(popped.left)
			if popped.right is not None:
				queued.enqueue(popped.right)
		return visited

	def isempty(self):
		return self._stored==0

	def insert(self,values):
		if isinstance(values,list):
			for val in values:
				self._singleinsert(val)
		else:
			self._singleinsert(values)

	def _singleinsert(self,value):
		if not isinstance(value,(int,float)):
			raise TypeError('Must insert integers or floats into AVL tree.')
		elif self.root is None:
			self.root = AVLNode(value)
			self._stored += 1
		else:
			x = self.root
			path = [-1]
			prev = None
			while x is not None:
				prev = x
				if value<=x.value:
					x = x.left
					path.append(0)
				else:
					x = x.right
					path.append(1)
			if path[-1]==0:
				prev.left = AVLNode(value)
				self._fix(path)
			else:
				prev.right = AVLNode(value)
				self._fix(path)
			self._stored += 1

	def _fix(self,steps):
		stacked = Stack()
		stacked.push(None)
		length = len(steps)
		node = self.root
		stacked.push((node,steps[0]))
		node.height = max(node.height,length)
		for trip in range(1,length):
			if steps[trip]==0:
				node = node.left
			else:
				node = node.right
			node.height = max(length-trip,node.height)
			stacked.push((node,steps[trip]))

		child = None
		while stacked.isempty() is False:
			parent = stacked.pop()
			if parent is None:
				break
			balance = self._getbalance(parent[0])
			if abs(balance)>1:
				grandparent = stacked.pop()
				if grandparent is not None:
					grandparent = grandparent[0]
				if balance<-1:
					if self._getbalance(child[0])>0:
						self._rotateleft(parent[0],child[0],child[0].right\
							,child[1])
					self._rotateright(grandparent,parent[0],child[0],parent[1])
				else:
					if self._getbalance(child[0])<0:
						self._rotateright(parent[0],child[0],child[0].left\
							,child[1])
					self._rotateleft(grandparent,parent[0],child[0],parent[1])
				break
			child = parent

	def _rotateright(self,grandparent,parent,child,step):
		if grandparent is None:
			self.head = child
			temp = self.head.right
			self.head.right = parent
			parent.left = temp
			self._calcheight(parent)
			self._calcheight(self.head.left)
			self._calcheight(self.head)
		else:
			if step==0:
				grandparent.left = child
			else:
				grandparent.right = child
			temp = child.right
			child.right = parent
			parent.left = temp
			self._calcheight(parent)
			self._calcheight(child)
			self._calcheight(grandparent)

	def _rotateleft(self,grandparent,parent,child,step):
		if grandparent is None:
			self.head = child
			temp = self.head.left
			self.head.left = parent
			parent.right = temp
			self._calcheight(parent)
			self._calcheight(self.head.left)
			self._calcheight(self.head)
		else:
			if step==0:
				grandparent.left = child
			else:
				grandparent.right = child
			temp = child.left
			child.left = parent
			parent.right = temp
			self._calcheight(parent)
			self._calcheight(child)
			self._calcheight(grandparent)

	def remove(self,value,head=None,prev=None,path=[-1]):
		if not isinstance(value,(int,float)):
			raise TypeError('{} not a valid type: only ints and floats can be' \
				+ ' stored in the AVL tree.')
		if self._stored==0:
			raise KeyError('{} not in empty AVL tree.'.format(value))
		x = self.head
		prev = prev
		if head is not None:
			x = head
		while x is not None:
			if x.value==value:
				if x.left is None and x.right is None:
					if path[-1]==-1:
						self.head = None
					elif path[-1]==0:
						prev.left = None
					else:
						prev.right = None
				elif x.left is None:
					if path[-1]==-1:
						self.head = x.right
					elif path[-1]==0:
						prev.left = x.right
					else:
						prev.right = x.right
				elif x.right is None:
					if path[-1]==-1:
						self.head = x.left
					elif path[-1]==0:
						prev.left = x.left
					else:
						prev.right = x.left
				else:
					temp = self._largest(x.left)
					path.append(0)
					self.remove(temp,x.left,x,path)
					x.value = temp
					self._stored += 1
				self._stored -= 1
				path.pop()
				self._fixremove(path)
				return
			prev = x
			if value<x.value:
				x = x.left
				path.append(0)
			else:
				x = x.right
				path.append(1)
		raise KeyError('{} not in AVL tree.'.format(value))

	def _largest(self,head):
		x = head
		while x.right is not None:
			x = x.right
		return x.value

	def _fixremove(self,steps):
		stacked = Stack()
		stacked.push(None)
		length = len(steps)
		node = self.head
		stacked.push((node,steps[0]))
		for trip in range(1,length):
			if steps[trip]==0:
				node = node.left
			else:
				node = node.right
				stacked.push((node,steps[trip]))

		child = None
		while stacked.isempty() is False:
			parent = stacked.pop()
			if parent is None:
				break
			self._calcheight(parent[0])
			balance = self._getbalance(parent[0])
			if abs(balance)>1:
				grandparent = stacked.pop()
				if grandparent is not None:
					grandparent = grandparent[0]
					self._calcheight(grandparent)
				if balance<-1:
					if self._getbalance(child[0])>0:
						self._rotateleft(parent[0],child[0],child[0].right\
							,child[1])
					self._rotateright(grandparent,parent[0],child[0],parent[1])
				else:
					if self._getbalance(child[0])<0:
						self._rotateright(parent[0],child[0],child[0].left\
							,child[1])
					self._rotateleft(grandparent,parent[0],child[0],parent[1])
					break
			child = parent
		while stacked.isempty() is False:
			current = stacked.pop()
			if current is not None:
				self._calcheight(current[0])

	def _getheight(self,node):
		if node is None:
			return 0
		return node.height

	def _calcheight(self,node):
		node.height = max(1+self._getheight(node.left)\
			,1+self._getheight(node.right))

	def _getbalance(self,node):
		if node is None:
			return 0
		return self._getheight(node.right) - self._getheight(node.left)

################################################################################
################################################################################
################################################################################

class MinHeap(object):
	def __init__(self,values=None,key=None):
		self._stored = 0
		self.contents = []
		self.key = None
		if values is not None:
			self.heapify(values,key)

	def __len__(self):
		return self._stored

	def __iter__(self):
		self._temp = [x for x in self.contents]
		self._tempstored = self._stored
		return self

	def __next__(self):
		if self.contents==[]:
			self.contents = [x for x in self._temp]
			self._stored = self._tempstored
			raise StopIteration
		else:
			return self.pop()

	def _log2(self,num):
		power = 0
		while 2**power<=num:
			power += 1
		return power-1

	def heapify(self,arr,key=None):
		self._stored = len(arr)
		if isinstance(arr[0],(tuple,list)):
			self.key = 0
			if key is not None:
				if isinstance(key,int) and key>=0 and key<len(arr[0]):
					self.key = key
		self.contents = [None for x in range(self._stored)]
		power = self._log2(self._stored)
		while power>=0:
			for i in range(2**power - 1,min(2**(power + 1) - 1,self._stored)):
				self.contents[i] = arr[i]
				if self.key is not None:
					self._siftdownkeyed(i)
				else:
					self._siftdown(i)
			power -= 1

	def _siftdown(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self._stored:
			if self.contents[parent]>self.contents[left] and self.contents\
				[left]<=self.contents[right]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[left]
				self.contents[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.contents[parent]>self.contents[right]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[right]
				self.contents[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self._stored and self.contents[parent]>self.contents[left]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[left]
			self.contents[left] = temp

	def _siftdownkeyed(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self._stored:
			if self.contents[parent][self.key]>self.contents[left][self.key] \
				and self.contents[left][self.key]\
				<=self.contents[right][self.key]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[left]
				self.contents[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.contents[parent][self.key]>self.contents[right][self.key]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[right]
				self.contents[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self._stored and self.contents[parent][self.key]\
			>self.contents[left][self.key]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[left]
			self.contents[left] = temp

	def pop(self):
		if self._stored==0:
			raise IndexError('Unable to pop from empty heap.')
		min_element = self.contents[0]
		self.contents[0] = self.contents[-1]
		self.contents.pop()
		self._stored -= 1
		if self.key is not None:
			self._siftdownkeyed(0)
		else:
			self._siftdown(0)
		return min_element

	def push(self,element,key=None):
		if isinstance(element,(list,tuple)):
			self.key = 0
			if key is not None:
				self.key = key
		if self.key is not None:
			self._pushkeyed(element)
		else:
			self.contents.append(element)
			self._stored += 1
			child = self._stored-1
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2
			while parent>=0 and self.contents[parent]>self.contents[child]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[child]
				self.contents[child] = temp
				child = parent
				if child%2==0:
					parent = (child-2)//2
				else:
					parent = (child-1)//2

	def _pushkeyed(self,element):
		self.contents.append(element)
		self._stored += 1
		child = self._stored-1
		if child%2==0:
			parent = (child-2)//2
		else:
			parent = (child-1)//2
		while parent>=0 and self.contents[parent][self.key]\
			>self.contents[child][self.key]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[child]
			self.contents[child] = temp
			child = parent
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2

	def peek(self):
		if self._stored==0:
			return None
		return self.contents[0]

	def isempty(self):
		return self._stored==0

################################################################################
################################################################################
################################################################################

class MaxHeap(object):
	def __init__(self,values=None,key=None):
		self._stored = 0
		self.contents = []
		self.key = None
		if values is not None:
			self.heapify(values,key)

	def __len__(self):
		return self._stored

	def __iter__(self):
		self._temp = [x for x in self.contents]
		return self

	def __next__(self):
		if self.contents==[]:
			self.contents = [x for x in self._temp]
			raise StopIteration
		else:
			return self.pop()

	def _log2(self,num):
		power = 0
		while 2**power<=num:
			power += 1
		return power-1

	def heapify(self,arr,key=None):
		self._stored = len(arr)
		if isinstance(arr[0],(tuple,list)):
			self.key = 0
			if key is not None:
				if isinstance(key,int) and key>=0 and key<len(arr[0]):
					self.key = key
		self.contents = [None for x in range(self._stored)]
		power = self._log2(self._stored)
		while power>=0:
			for i in range(2**power - 1,min(2**(power + 1) - 1,self._stored)):
				self.contents[i] = arr[i]
				if self.key is not None:
					self._siftdownkeyed(i)
				else:
					self._siftdown(i)
			power -= 1

	def _siftdown(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self._stored:
			if self.contents[parent]<self.contents[left] and self.contents\
				[left]>=self.contents[right]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[left]
				self.contents[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.contents[parent]<self.contents[right]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[right]
				self.contents[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self._stored and self.contents[parent]<self.contents[left]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[left]
			self.contents[left] = temp

	def _siftdownkeyed(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self._stored:
			if self.contents[parent][self.key]<self.contents[left][self.key] \
				and self.contents[left][self.key]\
				>=self.contents[right][self.key]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[left]
				self.contents[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.contents[parent][self.key]<self.contents[right][self.key]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[right]
				self.contents[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self._stored and self.contents[parent][self.key]\
			<self.contents[left][self.key]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[left]
			self.contents[left] = temp

	def pop(self):
		if self._stored==0:
			raise IndexError('Unable to pop from empty heap.')
		max_element = self.contents[0]
		self.contents[0] = self.contents[-1]
		self.contents.pop()
		self._stored -= 1
		if self.key is not None:
			self._siftdownkeyed(0)
		else:
			self._siftdown(0)
		return max_element

	def push(self,element,key=None):
		if isinstance(element,(list,tuple)):
			self.key = 0
			if key is not None:
				self.key = key
		if self.key is not None:
			self._pushkeyed(element)
		else:
			self.contents.append(element)
			self._stored += 1
			child = self._stored-1
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2
			while parent>=0 and self.contents[parent]<self.contents[child]:
				temp = self.contents[parent]
				self.contents[parent] = self.contents[child]
				self.contents[child] = temp
				child = parent
				if child%2==0:
					parent = (child-2)//2
				else:
					parent = (child-1)//2

	def _pushkeyed(self,element):
		self.contents.append(element)
		self._stored += 1
		child = self._stored-1
		if child%2==0:
			parent = (child-2)//2
		else:
			parent = (child-1)//2
		while parent>=0 and self.contents[parent][self.key]\
			<self.contents[child][self.key]:
			temp = self.contents[parent]
			self.contents[parent] = self.contents[child]
			self.contents[child] = temp
			child = parent
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2

	def peek(self):
		if self._stored==0:
			return None
		return self.contents[0]

	def isempty(self):
		return self._stored==0

################################################################################
################################################################################
################################################################################

class HashTable(object):
	def __init__(self,size=10):
		self._size = size
		self._stored = 0
		self.contents = [LinkedList() for i in range(self._size)]

	def __getitem__(self,key):
		temp = self._search(key)
		if temp is None:
			raise KeyError('{} not in hash table.'.format(key))
		else:
			return temp

	def __delitem__(self,key):
		self.remove(key)

	def __setitem__(self,key,value):
		self.insert(key,value)

	def __contains__(self,key):
		temp = self._search(key)
		return temp is not None

	def __del__(self):
		self.clear()

	def __iter__(self):
		self._found = 0
		self._last_ind = 0
		self._last_key = None
		return self

	def __next__(self):
		if self._found==self._stored:
			raise StopIteration
		self._found += 1
		if self._found==1:
			while self.contents[self._last_ind].head is None:
				self._last_ind += 1
			self._last_key = self.contents[self._last_ind].head.value[0]
			return self._last_key
		x = self.contents[self._last_ind].head
		while x.value[0]!=self._last_key:
			x = x.next
		x = x.next
		if x is not None:
			self._last_key = x.value[0]
			return self._last_key
		else:
			self._last_ind += 1
			while self.contents[self._last_ind].head is None:
				self._last_ind += 1
			self._last_key = self.contents[self._last_ind].head.value[0]
			return self._last_key

	def _hash(self,key):
		hashed = 0
		if isinstance(key,(int,float)):
			hashed = int(key)
			return hashed%self._size
		elif isinstance(key,str):
			for i in range(len(key)):
				hashed += 256**i*ord(key[i])
			return hashed%self._size
		else:
			raise TypeError('Unhashable type: use float, int or str.')

	def _rehash(self,hashed):
		return hashed%self._size

	def _rescale(self):
		for i in range(self._size):
			self.contents.append(LinkedList())
		self._size = int(self._size*2)
		for i in range(int(self._size/2)):
			temp_list = []
			for x in self.contents[i]:
				key = x[0]
				value = x[1]
				temp_list.append((key,value))
			self.contents[i] = LinkedList()
			for elem in temp_list:
				self.insert(elem[0],elem[1])

	def _search(self,key):
		x = self.contents[self._hash(key)[1]].head
		while x is not None:
			if x.value[0]==key:
				return x.value[1]
			x = x.next
		return x

	def insert(self,key,value):
		if self._search(key) is not None:
			raise KeyError('{} already a key in the hash table.'.format(key))
		self.contents[self._hash(key)].insert((key,value))
		self._stored += 1
		if self._stored > int(3*self._size/4):
			self._rescale()

	def remove(self,key):
		key_content = self._search(key)
		if key_content is not None:
			self.contents[self._hash(key)].remove(key_content)
			self._stored -= 1

	def clear(self):
		self._stored = 0
		self._size = 10
		self.contents = [LinkedList() for i in range(self._size)]