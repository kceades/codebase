"""
node, doublynode, and bstnode classes
"""

class node(object):
	def __init__(self,value=None):
		self.value = value
		self.next = None



class doublynode(object):
	def __init__(self,value=None,prev=None):
		self.value = value
		self.prev = prev
		self.next = None



class bstnode(object):
	def __init__(self,value=None):
		self.value = value
		self.left = None
		self.right = None



class avlnode(object):
	def __init__(self,value=None):
		self.value = value
		self.left = None
		self.right = None
		self.height = 0



class trienode(object):
	def __init__(self):
		self.children = {}
		self.value = 0



"""
linked lists
"""

class doublylinkedlist(object):
	def __init__(self):
		self.head = None
		self.tail = None
		self.stored = 0

	def __str__(self):
		if self.IsEmpty():
			return ''
		else:
			x = self.head
			string = '{} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{} '.format(x.value)
			return string

	def __iter__(self):
		self.current = self.head
		return self

	def __next__(self):
		if self.current is None:
			raise StopIteration
		x = self.current
		self.current = self.current.next
		return x

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored

	def Insert(self,value):
		if self.IsEmpty():
			self.head = doublynode(value)
		elif self.stored==1:
			self.tail = doublynode(value,self.head)
			self.head.next = self.tail
		else:
			x = doublynode(value,self.tail)
			self.tail.next = x
			self.tail = x
			x = self.head
		self.stored += 1

	def Remove(self,value):
		if self.IsEmpty():
			return
		x = self.head
		while x is not None:
			if x.value==value:
				break
			x = x.next
		if x is None:
			return
		elif self.stored==1:
			self.head = None
		elif self.stored==2:
			self.tail = None
		elif x.next is None:
			x.prev.next = None
		elif x.prev is None:
			x = x.next
		else:
			x.prev.next = x.next
			x.next.prev = x.prev
		self.stored -= 1

	def Reverse(self):
		if self.stored<=1:
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



class linkedlist(object):
	def __init__(self):
		self.head = None
		self.stored = 0

	def __str__(self):
		if self.IsEmpty():
			return ''
		else:
			x = self.head
			string = '{} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{} '.format(x.value)
			return string

	def __iter__(self):
		self.current = self.head
		return self

	def __next__(self):
		if self.current is None:
			raise StopIteration
		x = self.current
		self.current = self.current.next
		return x

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored

	def Insert(self,value):
		if self.IsEmpty():
			self.head = node(value)
		else:
			x = self.head
			while x.next is not None:
				x = x.next
			x.next = node(value)
		self.stored += 1

	def Remove(self,value):
		if self.IsEmpty():
			return
		prev = None
		x = self.head
		while x is not None:
			if x.value==value:
				break
			prev = x
			x = x.next
		if x is None:
			return
		elif self.stored==1:
			self.head = None
		else:
			prev.next = x.next
		self.stored -= 1

	def Reverse(self):
		if self.stored<=1:
			return
		previous_node = None
		current_node = self.head
		while current_node.next is not None:
			next_node = current_node.next
			current_node.next = previous_node
			previous_node = current_node
			current_node = next_node
		self.head = current_node
		self.head.next = previous_node



"""
heaps
"""

class minheap(object):
	def __init__(self):
		self.stored = 0
		self.values = []
		self.key = None

	def _Log2(self,num):
		power = 0
		while 2**power<=num:
			power += 1
		return power-1

	def Heapify(self,arr,key=None):
		self.stored = len(arr)
		if isinstance(arr[0],(tuple,list)):
			self.key = 0
		if key is not None:
			self.key = key
		self.values = [None for x in range(self.stored)]
		power = self._Log2(self.stored)
		while power>=0:
			for i in range(2**power - 1,min(2**(power + 1) - 1,self.stored)):
				self.values[i] = arr[i]
				if self.key is not None:
					self._SiftDownKeyed(i)
				else:
					self._SiftDown(i)
			power -= 1

	def _SiftDown(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self.stored:
			if self.values[parent]>self.values[left] and self.values[left]\
				<=self.values[right]:
				temp = self.values[parent]
				self.values[parent] = self.values[left]
				self.values[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.values[parent]>self.values[right]:
				temp = self.values[parent]
				self.values[parent] = self.values[right]
				self.values[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self.stored and self.values[parent]>self.values[left]:
			temp = self.values[parent]
			self.values[parent] = self.values[left]
			self.values[left] = temp

	def _SiftDownKeyed(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self.stored:
			if self.values[parent][self.key]>self.values[left][self.key] \
				and self.values[left][self.key]<=self.values[right][self.key]:
				temp = self.values[parent]
				self.values[parent] = self.values[left]
				self.values[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.values[parent][self.key]>self.values[right][self.key]:
				temp = self.values[parent]
				self.values[parent] = self.values[right]
				self.values[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self.stored and self.values[parent][self.key]\
			>self.values[left][self.key]:
			temp = self.values[parent]
			self.values[parent] = self.values[left]
			self.values[left] = temp

	def Pop(self):
		if self.IsEmpty():
			return None
		min_element = self.values[0]
		self.values[0] = self.values[-1]
		self.values.pop()
		self.stored -= 1
		if self.key is not None:
			self._SiftDownKeyed(0)
		else:
			self._SiftDown(0)
		return min_element

	def Push(self,element,key=None):
		if isinstance(element,(list,tuple)):
			self.key = 0
			if key is not None:
				self.key = key
		if self.key is not None:
			self._PushKeyed(element)
		else:
			self.values.append(element)
			self.stored += 1
			child = self.stored-1
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2
			while parent>=0 and self.values[parent]>self.values[child]:
				temp = self.values[parent]
				self.values[parent] = self.values[child]
				self.values[child] = temp
				child = parent
				if child%2==0:
					parent = (child-2)//2
				else:
					parent = (child-1)//2

	def _PushKeyed(self,element):
		self.values.append(element)
		self.stored += 1
		child = self.stored-1
		if child%2==0:
			parent = (child-2)//2
		else:
			parent = (child-1)//2
		while parent>=0 and self.values[parent][self.key]\
			>self.values[child][self.key]:
			temp = self.values[parent]
			self.values[parent] = self.values[child]
			self.values[child] = temp
			child = parent
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2

	def Peek(self):
		if self.stored==0:
			return None
		return self.values[0]

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored



class maxheap(object):
	def __init__(self):
		self.stored = 0
		self.values = []
		self.key = None

	def _Log2(self,num):
		power = 0
		while 2**power<=num:
			power += 1
		return power-1

	def Heapify(self,arr,key=None):
		self.stored = len(arr)
		if isinstance(arr[0],(tuple,list)):
			self.key = 0
		if key is not None:
			self.key = key
		self.values = [None for x in range(self.stored)]
		power = self._Log2(self.stored)
		while power>=0:
			for i in range(2**power - 1,min(2**(power + 1) - 1,self.stored)):
				self.values[i] = arr[i]
				if self.key is not None:
					self._SiftDownKeyed(i)
				else:
					self._SiftDown(i)
			power -= 1

	def _SiftDown(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self.stored:
			if self.values[parent]<self.values[left] and self.values[left]\
				>=self.values[right]:
				temp = self.values[parent]
				self.values[parent] = self.values[left]
				self.values[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.values[parent]<self.values[right]:
				temp = self.values[parent]
				self.values[parent] = self.values[right]
				self.values[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self.stored and self.values[parent]<self.values[left]:
			temp = self.values[parent]
			self.values[parent] = self.values[left]
			self.values[left] = temp

	def _SiftDownKeyed(self,index):
		parent = index
		left = parent*2+1
		right = parent*2+2
		while right<self.stored:
			if self.values[parent][self.key]<self.values[left][self.key] \
				and self.values[left][self.key]>=self.values[right][self.key]:
				temp = self.values[parent]
				self.values[parent] = self.values[left]
				self.values[left] = temp
				parent = left
				left = 2*parent + 1
				right = 2*parent + 2
			elif self.values[parent][self.key]<self.values[right][self.key]:
				temp = self.values[parent]
				self.values[parent] = self.values[right]
				self.values[right] = temp
				parent = right
				left = 2*parent + 1
				right = 2*parent + 2
			else:
				break
		if left<self.stored and self.values[parent][self.key]\
			<self.values[left][self.key]:
			temp = self.values[parent]
			self.values[parent] = self.values[left]
			self.values[left] = temp

	def Pop(self):
		if self.IsEmpty():
			return None
		min_element = self.values[0]
		self.values[0] = self.values[-1]
		self.values.pop()
		self.stored -= 1
		if self.key is not None:
			self._SiftDownKeyed(0)
		else:
			self._SiftDown(0)
		return min_element

	def Push(self,element):
		if self.key is not None:
			self._PushKeyed(element)
		else:
			self.values.append(element)
			self.stored += 1
			child = self.stored-1
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2
			while parent>=0 and self.values[parent]<self.values[child]:
				temp = self.values[parent]
				self.values[parent] = self.values[child]
				self.values[child] = temp
				child = parent
				if child%2==0:
					parent = (child-2)//2
				else:
					parent = (child-1)//2

	def _PushKeyed(self,element):
		self.values.append(element)
		self.stored += 1
		child = self.stored-1
		if child%2==0:
			parent = (child-2)//2
		else:
			parent = (child-1)//2
		while parent>=0 and self.values[parent][self.key]\
			<self.values[child][self.key]:
			temp = self.values[parent]
			self.values[parent] = self.values[child]
			self.values[child] = temp
			child = parent
			if child%2==0:
				parent = (child-2)//2
			else:
				parent = (child-1)//2

	def Peek(self):
		if self.stored==0:
			return None
		return self.values[0]

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored



"""
hash table
"""

class hashtable(object):
	def __init__(self,size=1000):
		self.size = size
		self.stored = 0
		self.values = [linkedlist() for i in range(self.size)]

	def __iter__(self):
		self.found = 0
		self.last_ind = 0
		self.last_key = None
		return self

	def __next__(self):
		if self.found==self.stored:
			raise StopIteration
		self.found += 1
		if self.found==1:
			while self.values[self.last_ind].head is None:
				self.last_ind += 1
			self.last_key = self.values[self.last_ind].head.value[0]
			return self.last_key
		x = self.values[self.last_ind].head
		while x.value[0]!=self.last_key:
			x = x.next
		if x.next is not None:
			self.last_key = x.value[0]
			return self.last_key
		else:
			self.last_ind += 1
			while self.values[self.last_ind].head is None:
				self.last_ind += 1
			self.last_key = self.values[self.last_ind].head.value[0]
			return self.last_key

	def __getitem__(self,key):
		return self.Search(key)

	def _Hash(self,key):
		hashed = 0
		for i in range(len(key)):
			hashed += 256**i*ord(key[i])
		return hashed%self.size

	def Insert(self,key,value):
		self.values[self._Hash(key)].Insert((key,value))
		self.stored += 1

	def Search(self,key):
		x = self.values[self._Hash(key)].head
		while x is not None:
			if x.value[0]==key:
				return x.value[1]
			x = x.next
		return x

	def Remove(self,key):
		x = self.values[self._Hash(key)].head
		prev = None
		while x is not None:
			if x.value[0]==key:
				break
			prev = x
			x = x.next
		if x is None:
			return
		if prev is None and x.next is None:
			x = None
		elif prev is None:
			x = x.next
		else:
			prev.next = x.next
		self.stored -= 1

	def Clear(self):
		self.stored = 0
		for i in range(self.size):
			self.values[i].head = None



"""
stack
"""

class stack(object):
	def __init__(self):
		self.top = None
		self.stored = 0

	def __str__(self):
		if self.IsEmpty():
			return ''
		else:
			x = self.top
			string = '{:} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{:} '.format(x.value)
			return string

	def Push(self,value):
		if self.IsEmpty():
			self.top = node(value)
		else:
			x = node(value)
			x.next = self.top
			self.top = x
		self.stored += 1

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored

	def Pop(self):
		if self.IsEmpty():
			return None
		popped = self.top.value
		if self.stored==1:
			self.top = None
		else:
			self.top = self.top.next
		self.stored -= 1
		return popped

	def Reverse(self):
		if self.stored<=1:
			return
		prev = None
		x = self.top
		while x.next is not None:
			next_node = x.next
			x.next = prev
			prev = x
			x = next_node
		x.next = prev
		self.top = x



"""
queue
"""

class queue(object):
	def __init__(self):
		self.first = None
		self.last = None
		self.stored = 0

	def __str__(self):
		if self.stored==0:
			return ''
		else:
			x = self.first
			string = '{:} '.format(x.value)
			while x.next is not None:
				x = x.next
				string += '{:} '.format(x.value)
			return string

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored

	def Enqueue(self,value):
		if self.IsEmpty():
			self.first = node(value)
		elif self.stored==1:
			self.last = node(value)
			self.first.next = self.last
		else:
			self.last.next = node(value)
			self.last = self.last.next
		self.stored += 1

	def Dequeue(self):
		if self.IsEmpty():
			return None
		self.stored -= 1
		dequeued = self.first.value
		if self.stored==0:
			self.first = None
		elif self.stored==1:
			self.first = self.last
			self.last = None
		else:
			self.first = self.first.next
		return dequeued

	def Reverse(self):
		if self.stored<=1:
			return
		prev = None
		x = self.first
		while x.next is not None:
			next_node = x.next
			x.next = prev
			prev = x
			x = next_node
		x.next = prev
		temp = self.first
		self.first = self.last
		self.last = self.first



"""
trees
"""

class bst(object):
	def __init__(self):
		self.head = None
		self.stored = 0

	def __str__(self):
		return self.BreadthFirst()
	
	def PostOrder(self,node=False):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return self.PostOrder(node.left) + self.PostOrder(node.right) \
			+ '{} '.format(node.value)

	def PreOrder(self,node=False):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return '{} '.format(node.value) + self.PreOrder(node.left) \
			+ self.PreOrder(node.right)

	def InOrder(self,node):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return self.InOrder(node.left) + '{} '.format(node.value) \
			+ self.InOrder(node.right)

	def BreadthFirst(self):
		if self.IsEmpty():
			return ''
		queued = queue()
		queued.Enqueue(self.head)
		string = ''
		while queued.IsEmpty() is False:
			popped = queued.Dequeue()
			string += '{} '.format(popped.value)
			if popped.left is not None:
				queued.Enqueue(popped.left)
			if popped.right is not None:
				queued.Enqueue(popped.right)
		return string

	def IsEmpty(self):
		return self.stored==0

	def Size(self):
		return self.stored

	def Insert(self,value):
		if isinstance(value,list):
			self.SortedInsert(value)
			return
		if self.IsEmpty():
			self.head = bstnode(value)
		else:
			x = self.head
			prev = None
			path = -1
			while x is not None:
				prev = x
				if value<=x.value:
					path = 0
					x = x.left
				else:
					path = 1
					x = x.right
			if path==0:
				prev.left = bstnode(value)
			else:
				prev.right = bstnode(value)
		self.stored += 1

	def SortedInsert(self,values):
		self._SortedInsertHelper(values,0,len(values)-1)

	def _SortedInsertHelper(self,values,start,end):
		mid = int((start+end)/2)
		self.Insert(values[mid])
		if start<=(mid-1):
			self._SortedInsertHelper(values,start,mid-1)
		if (mid+1)<=end:
			self._SortedInsertHelper(values,mid+1,end)

	def Remove(self,value,head=None,prev=None,path=-1):
		if self.IsEmpty():
			return
		x = self.head
		prev = prev
		if head is not None:
			x = head
		while x is not None:
			if x.value==value:
				if x.left is None and x.right is None:
					if path==-1:
						self.head = None
					elif path==0:
						prev.left = None
					else:
						prev.right = None
				elif x.left is None:
					if path==-1:
						self.head = x.right
					elif path==0:
						prev.left = x.right
					else:
						prev.right = x.right
				elif x.right is None:
					if path==-1:
						self.head = x.left
					elif path==0:
						prev.left = x.left
					else:
						prev.right = x.left
				else:
					temp = self._Largest(x.left)
					self.Remove(temp,x.left,x,0)
					x.value = temp
					self.stored += 1
				self.stored -= 1
				return
			prev = x
			if value<x.value:
				x = x.left
				path = 0
			else:
				x = x.right
				path = 1

	def _Largest(self,head):
		x = head
		while x.right is not None:
			x = x.right
		return x.value



class trie(object):
	def __init__(self):
		self.head = None

	def Insert(self,string):
		if self.head is None:
			self.head = trienode()
		x = self.head
		for char in string:
			if char not in x.children:
				x.children[char] = trienode()
			x = x.children[char]
		x.value += 1

	def Wipe(self):
		self.head = trienode()

	def IsEmpty(self):
		return self.head.children=={}

	def TopWords(self,partial='',k=3):
		x = self.head
		suggestions = []
		for char in partial:
			if char not in x.children:
				return None
			x = x.children[char]
		queued = queue()
		queued.Enqueue((x,partial))
		maxxed = maxheap()
		while queued.IsEmpty() is False:
			current = queued.Dequeue()
			if current[0].value>0:
				maxxed.Push((current[0].value,current[1]))
			for char in current[0].children:
				queued.Enqueue((current[0].children[char],current[1]+char))
		for i in range(k):
			suggestions.append(maxxed.Pop()[1])
		return suggestions



class avl(object):
	def __init__(self):
		self.head = None
		self.stored = 0

	def IsEmpty(self):
		return self.stored==0

	def __str__(self):
		return self.BreadthFirst()
	
	def PostOrder(self,node=False):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return self.PostOrder(node.left) + self.PostOrder(node.right) \
			+ '{} '.format(node.value)

	def PreOrder(self,node=False):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return '{} '.format(node.value) + self.PreOrder(node.left) \
			+ self.PreOrder(node.right)

	def InOrder(self,node=False):
		if node is False:
			node = self.head
		if node is None:
			return ''
		return self.InOrder(node.left) + '{} '.format(node.value) \
			+ self.InOrder(node.right)

	def BreadthFirst(self):
		if self.IsEmpty():
			return ''
		queued = queue()
		queued.Enqueue(self.head)
		string = ''
		while queued.IsEmpty() is False:
			popped = queued.Dequeue()
			string += '{} '.format(popped.value)
			if popped.left is not None:
				queued.Enqueue(popped.left)
			if popped.right is not None:
				queued.Enqueue(popped.right)
		return string

	def Insert(self,value):
		x = self.head
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
		if path[-1]==-1:
			self.head = avlnode(value)
		elif path[-1]==0:
			prev.left = avlnode(value)
			self._Fix(path)
		else:
			prev.right = avlnode(value)
			self._Fix(path)
		self.stored += 1

	def Remove(self,value,head=None,prev=None,path=[-1]):
		if self.IsEmpty():
			return
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
					temp = self._Largest(x.left)
					path.append(0)
					self.Remove(temp,x.left,x,path)
					x.value = temp
					self.stored += 1
				self.stored -= 1
				path.pop()
				self._FixRemove(path)
				return
			prev = x
			if value<x.value:
				x = x.left
				path.append(0)
			else:
				x = x.right
				path.append(1)

	def _Largest(self,head):
		x = head
		while x.right is not None:
			x = x.right
		return x.value

	def _RotateRight(self,grandparent,parent,child,step):
		if grandparent is None:
			self.head = child
			temp = self.head.right
			self.head.right = parent
			parent.left = temp
			self._CalcHeight(parent)
			self._CalcHeight(self.head.left)
			self._CalcHeight(self.head)
		else:
			if step==0:
				grandparent.left = child
			else:
				grandparent.right = child
			temp = child.right
			child.right = parent
			parent.left = temp
			self._CalcHeight(parent)
			self._CalcHeight(child)
			self._CalcHeight(grandparent)

	def _RotateLeft(self,grandparent,parent,child,step):
		if grandparent is None:
			self.head = child
			temp = self.head.left
			self.head.left = parent
			parent.right = temp
			self._CalcHeight(parent)
			self._CalcHeight(self.head.left)
			self._CalcHeight(self.head)
		else:
			if step==0:
				grandparent.left = child
			else:
				grandparent.right = child
			temp = child.left
			child.left = parent
			parent.right = temp
			self._CalcHeight(parent)
			self._CalcHeight(child)
			self._CalcHeight(grandparent)

	def _FixRemove(self,steps):
		stacked = stack()
		stacked.Push(None)
		length = len(steps)
		node = self.head
		stacked.Push((node,steps[0]))
		for trip in range(1,length):
			if steps[trip]==0:
				node = node.left
			else:
				node = node.right
				stacked.Push((node,steps[trip]))

		child = None
		while stacked.IsEmpty() is False:
			parent = stacked.Pop()
			if parent is None:
				break
			self._CalcHeight(parent[0])
			balance = self._GetBalance(parent[0])
			if abs(balance)>1:
				grandparent = stacked.Pop()
				if grandparent is not None:
					grandparent = grandparent[0]
					self._CalcHeight(grandparent)
				if balance<-1:
					if self._GetBalance(child[0])>0:
						self._RotateLeft(parent[0],child[0],child[0].right\
							,child[1])
					self._RotateRight(grandparent,parent[0],child[0],parent[1])
				else:
					if self._GetBalance(child[0])<0:
						self._RotateRight(parent[0],child[0],child[0].left\
							,child[1])
					self._RotateLeft(grandparent,parent[0],child[0],parent[1])
					break
			child = parent
		while stacked.IsEmpty() is False:
			current = stacked.Pop()
			if current is not None:
				self._CalcHeight(current[0])

	def _Fix(self,steps):
		stacked = stack()
		stacked.Push(None)
		length = len(steps)
		node = self.head
		stacked.Push((node,steps[0]))
		node.height = max(node.height,length)
		for trip in range(1,length):
			if steps[trip]==0:
				node = node.left
			else:
				node = node.right
			node.height = max(length-trip,node.height)
			stacked.Push((node,steps[trip]))

		child = None
		while stacked.IsEmpty() is False:
			parent = stacked.Pop()
			if parent is None:
				break
			balance = self._GetBalance(parent[0])
			if abs(balance)>1:
				grandparent = stacked.Pop()
				if grandparent is not None:
					grandparent = grandparent[0]
				if balance<-1:
					if self._GetBalance(child[0])>0:
						self._RotateLeft(parent[0],child[0],child[0].right\
							,child[1])
					self._RotateRight(grandparent,parent[0],child[0],parent[1])
				else:
					if self._GetBalance(child[0])<0:
						self._RotateRight(parent[0],child[0],child[0].left\
							,child[1])
					self._RotateLeft(grandparent,parent[0],child[0],parent[1])
				break
			child = parent

	def _GetHeight(self,node):
		if node is None:
			return 0
		return node.height

	def _CalcHeight(self,node):
		node.height = max(1+self._GetHeight(node.left)\
			,1+self._GetHeight(node.right))

	def _GetBalance(self,node):
		if node is None:
			return 0
		return self._GetHeight(node.right) - self._GetHeight(node.left)