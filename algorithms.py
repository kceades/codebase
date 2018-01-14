import datastructures

import random
import time



"""
sorting
"""

def InsertionSort(arr):
	for i in range(len(arr)):
		current = arr[i]
		j = i-1
		while j>=0 and arr[j]>current:
			arr[j+1] = arr[j]
			j -= 1
		arr[j+1] = current



def BubbleSort(arr):
	unswapped = True
	while unswapped:
		unswapped = False
		for index in range(len(arr)-1):
			if arr[index+1]<arr[index]:
				temp = arr[index+1]
				arr[index+1] = arr[index]
				arr[index] = temp
				unswapped = True



def MergeSort(arr):
	if len(arr)>1:
		mid = len(arr)//2
		left = arr[:mid]
		right = arr[mid:]
		MergeSort(left)
		MergeSort(right)

		i = 0
		j = 0
		k = 0
		while i<mid and j<(len(arr)-mid):
			if left[i]<=right[j]:
				arr[k] = left[i]
				i += 1
			else:
				arr[k] = right[j]
				j += 1
			k += 1
		while i<mid:
			arr[k] = left[i]
			k += 1
			i += 1
		while j<(len(arr)-mid):
			arr[k] = right[j]
			k += 1
			j += 1



def QuickSort(arr):
	QuickSortHelper(arr,0,len(arr)-1)

def QuickSortHelper(arr,start,end):
	split = Partition(arr,start,end)
	if split-1>start:
		QuickSortHelper(arr,start,split-1)
	if split+1<end:
		QuickSortHelper(arr,split+1,end)

def Partition(arr,start,end):
	pivot = arr[start]
	left = start + 1
	right = end
	while right>=left:
		if arr[left]>pivot and arr[right]<pivot:
			temp = arr[left]
			arr[left] = arr[right]
			arr[right] = temp
			left += 1
			right -= 1
		elif arr[left]>pivot:
			right -= 1
		else:
			left += 1
	temp = arr[right]
	arr[right] = pivot
	arr[start] = temp
	return right





def HeapSort(arr):
	length = len(arr)
	power = Log2(length)
	while power>=0:
		for i in range(2**power - 1,min(2**(power + 1) - 1,length)):
			SiftDown(arr,i,length)
		power -= 1
	while length>1:
		temp = arr[length-1]
		arr[length-1] = arr[0]
		arr[0] = temp
		length -= 1
		SiftDown(arr,0,length)

def SiftDown(arr,index,length):
	parent = index
	left = parent*2+1
	right = parent*2+2
	while right<length:
		if arr[parent]<arr[left] and arr[left]>=arr[right]:
			temp = arr[parent]
			arr[parent] = arr[left]
			arr[left] = temp
			parent = left
			left = 2*parent + 1
			right = 2*parent + 2
		elif arr[parent]<arr[right]:
			temp = arr[parent]
			arr[parent] = arr[right]
			arr[right] = temp
			parent = right
			left = 2*parent + 1
			right = 2*parent + 2
		else:
			break
	if left<length and arr[parent]<arr[left]:
		temp = arr[parent]
		arr[parent] = arr[left]
		arr[left] = temp

def Log2(num):
	power = 0
	while 2**power<=num:
		power += 1
	return power-1



"""
Graph algorithms
"""

class dijkstranode(object):
	def __init__(self,length=0):
		self.length = length

def Dijkstra(graph,start,end):
	heapped = datastructures.minheap()
	nodes = {start:dijkstranode()}
	for edge in graph[start]:
		if edge[1]==end:
			return edge[0]
		nodes[edge[1]] = dijkstranode(edge[0])
		heapped.Push((edge[0],edge[1]))
	while heapped.IsEmpty() is False:
		current = heapped.Pop()
		for edge in graph[current[1]]:
			total_length = nodes[current[1]].length + edge[0]
			if edge[1]==end:
				return total_length
			if edge[1] in nodes:
				nodes[edge[1]].length = min(nodes[edge[1]].length,total_length)
			else:
				nodes[edge[1]] = dijkstranode(total_length)
				heapped.Push((total_length,edge[1]))
	return -1