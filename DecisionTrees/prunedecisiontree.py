import numpy as np
import csv
import math
import statistics 
import matplotlib.pyplot as plt
import copy

class Node(object):
	def __init__(self, data):
		self.data = data
		self.children = []
	def add_child(self, obj):
		self.children.append(obj)
	def remove_child(self, obj):
		self.children.remove(obj)	
	def height(self):
		if self.children==[]:
			return 0
		else:
			return 1+max([x.height() for x in self.children])	

list_train = []
with open('dtree_data/train.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		y = [x.strip() for x in row]
		list_train.append(y)

list_test = []
with open('dtree_data/test.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		y = [x.strip() for x in row]
		list_test.append(y)

list_valid = []
with open('dtree_data/valid.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		y = [x.strip() for x in row]
		list_valid.append(y)	

list_attribute = []
with open('attr.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
		y = [x.strip() for x in row]
		list_attribute.append(y)

# Preprocessing step:
# Choosing the numerical attributes
numerical_attribute = []
for attribute in list_attribute:
	if attribute[1].strip()=='Numerical':
		numerical_attribute.append(attribute[0])

numerical_positions = []
count=0
for i in list_train[0]:
	if i.strip() in numerical_attribute:
		numerical_positions.append(count)
	count+=1	



# Median segregation for numerical attributes
train_medians = []
for j in numerical_positions:
	values = [i[j] for i in list_train[1:-1]]
	med = statistics.median(values)
	train_medians.append(med)

for i in list_train[1:-1]:
	count=0
	for j in numerical_positions:
		if i[j]>train_medians[count]:
			i[j]=1
		else:
			i[j]=0	
		count+=1		

test_medians = []
for j in numerical_positions:
	values = [i[j] for i in list_test[1:-1]]
	med = statistics.median(values)
	test_medians.append(med)

for i in list_test[1:-1]:
	count=0
	for j in numerical_positions:
		if i[j]>test_medians[count]:
			i[j]=1
		else:
			i[j]=0	
		count+=1

validation_medians = []
for j in numerical_positions:
	values = [i[j] for i in list_valid[1:-1]]
	med = statistics.median(values)
	validation_medians.append(med)

for i in list_valid[1:-1]:
	count=0
	for j in numerical_positions:
		if i[j]>validation_medians[count]:
			i[j]=1
		else:
			i[j]=0	
		count+=1

# Now build the tree
attributes = [i[0] for i in list_attribute[1:-1]]
num_attributes = len(attributes)


def shouldsplit(position,values,trainlist):
	train_examples = []
	# print(position,values)
	# print(trainlist)
	for nn in trainlist:
		i = list_train[nn]
		truth = True
		for j,k in zip(position,values):
			if not (str(i[j]).strip()==str(k)):
				truth=False
				break
		if truth:
			train_examples.append(i)
	hy0 = 0
	hy1 = 0	
	for i in train_examples:
		if (i[-1])=='0':
			hy0+=1
		else :
			hy1+=1
	if (hy0==0):
		#pure subset
		return (False,1)
	elif (hy1==0):
		return (False,0)	
	elif (hy0>hy1):
		return (True,0)
	else : 
		return (True,1)		



def splitonattribute(position,values,trainlist):
	train_examples = []
	for nn in trainlist:
		i = list_train[nn]
		truth = True
		for j,k in zip(position,values):
			if not (str(i[j]).strip()==str(k)):
				truth=False
				break
		if truth:
			train_examples.append(i)
	hy0 = 0
	hy1 = 0	
	for i in train_examples:
		if (i[-1])=='0':
			hy0+=1
		else :
			hy1+=1
	total = hy0+hy1		
	py0 = hy0 / total
	py1 = hy1 / total
	sy = -(py0*math.log(py0,2))-(py1*math.log(py1,2))
	maxgain = 0
	attribute_picked = -1
	for i in range(0,len(attributes)):
		if i not in position:
			#calculate information gain for that attribute
			# first split on the attribute - discrete multi split and numerical single split
			split_attribute_prob={}
			total = 0
			for j in train_examples:
				p = j[i]
				if p not in split_attribute_prob:
					split_attribute_prob[p]={}
					if (j[-1])=='0':
						m= split_attribute_prob[p]
						m[0]=1
					else:
						m= split_attribute_prob[p]
						m[1]=1	 
				else:
					m=split_attribute_prob[p]
					if (j[-1])=='0' and 0 in m:
						m[0]+=1
					elif (j[-1])=='0' and 0 not in m:
						m[0]=1
					elif (j[-1])=='1' and 1 in m:
						m[1]+=1
					elif (j[-1])=='1' and 1 not in m:
						m[1]=1			
				total+=1
			summation = 0	
			for m in split_attribute_prob:
				n = split_attribute_prob[m]
				prob_of_occuring = sum(n.values())/total
				if 0 not in n:
					n[0]=0
				if 1 not in n:
					n[1]=0	
				p0 = n[0]/(n[0]+n[1])
				p1 = n[1]/(n[0]+n[1])
				if p0==0:
					hsv = -(p1*math.log(p1,2))
				elif p1==0:
					hsv = -(p0*math.log(p0,2))
				else:		
					hsv = -(p0*math.log(p0,2))-(p1*math.log(p1,2))
				summation += hsv* prob_of_occuring
			if (sy-summation>0 and sy-summation>maxgain):
				maxgain = sy-summation
				#pick that attribute
				attribute_picked = i
	# print(attribute_picked)
	# print(maxgain,attribute_picked)			
	return attribute_picked


trainhelper = [i for i in range(0,len(list_train[1:-1]))]
root = Node([[],[],trainhelper,-1])
counter=0
# correct = 0 
# total = 0
train_accuracy=[]
test_accuracy= []
validation_accuracy = []
train_total = len(list_train[1:-1])
test_total = len(list_test[1:-1])
valid_total = len(list_valid[1:-1])
print(train_total,test_total,valid_total)
numbernodes=[]

def makeTree(node):
	# print(node.data[0],node.data[1])
	kk = shouldsplit(node.data[0],node.data[1],node.data[2])
	global counter
	counter+=1
	node.data[3]=kk[1]	
	if (kk[0]):
		attribute_pick = splitonattribute(node.data[0],node.data[1],node.data[2])
		if not attribute_pick==-1:
			branches = list(set([i[attribute_pick] for i in list_train[1:-1]]))
			branches.sort()
			for i in branches:
				th = [x for x in node.data[2] if list_train[x][attribute_pick]==i]
				p = Node([node.data[0]+[attribute_pick],node.data[1]+[i],th,-1])
				node.add_child(p)
				makeTree(p)



				

# Build tree
makeTree(root)	
# print([x.data[1] for x in root.children[0].children[1].children])
#Decision Tree is built
print('Decision tree is built!')

correct=0
current_accuracy = 0
for j in list_valid[1:-1]:
	thenode = copy.copy(root)
	level = 0
	while not (thenode.children==[]):
		positions = thenode.children[0].data[0]
		attr = positions[level]
		trainingattribute = j[attr]
		truth=False
		for l in thenode.children:
			values= l.data[1]
			if (str(values[level])==trainingattribute):
				thenode = l
				truth=True
				break
		if not truth:
			break
		level+=1
	if (str(thenode.data[3])==j[-1]):
		correct+=1

current_accuracy = correct/valid_total
print('Current accuracy of tree is:',current_accuracy)
print('now start pruning:')		


prunethisnode = copy.copy(root)
ii =0 

jj=0
def printTree(node):
	global root
	global jj
	jj+=1
	for p in node.children:
		printTree(p)

# printTree(root)
def getNode(node):
	global valid_total
	global current_accuracy
	global prunethisnode
	global root
	global ii
	# print(node.data[1])
	childrens = node.children
	for p in childrens:
		correct=0
		for j in list_valid[1:-1]:
			thenode = copy.copy(root)
			level = 0
			while not (thenode.children==[]):
				positions = thenode.children[0].data[0]
				attr = positions[level]
				trainingattribute = j[attr]
				truth=False
				for l in thenode.children:
					values= l.data[1]
					if (str(values[level])==trainingattribute and not l.data[1]==p.data[1]):
						thenode = l
						truth=True
						break
				if not truth:
					break
				level+=1
			if (str(thenode.data[3])==j[-1]):
				correct+=1
		# print(ii,correct/valid_total,p.data[1])	
		ii+=1	
		if (correct/valid_total>current_accuracy):
			current_accuracy = correct/valid_total
			prunethisnode = p
			break
		getNode(p)		


# getNode(root)
# print(current_accuracy)
# print(prunethisnode.data[0],prunethisnode.data[1])
# print(len(root.children))
					
training_accuracy=[]
numbernodes=[]


def pruneTree(node):
	global root
	global prunethisnode
	global current_accuracy
	global jj
	print(root.data[0],root.data[1],len(root.children))

	getNode(node)
	n=prunethisnode
	printTree(root)
	diff=jj
	print(diff)
	index=0
	while not (n==Node([])) and index<10:
		jj=0	
		printTree(root)
		print('jj val:',jj)
		numbernodes.append(jj)
		correct=0	
		for j in list_train[1:-1]:
			thenode = root
			level = 0
			while not (thenode.children==[]):
				positions = thenode.children[0].data[0]
				attr = positions[level]
				trainingattribute = j[attr]
				truth=False
				for l in thenode.children:
					values= l.data[1]
					if (str(values[level])==trainingattribute):
						thenode = l
						truth=True
						break
				if not truth:
					break
				level+=1
			if (str(thenode.data[3])==j[-1]):
				correct+=1
		train_accuracy.append(correct/train_total)
		print('train accuracy:',correct/train_total)

		correct=0	
		for j in list_test[1:-1]:
			thenode = root
			level = 0
			while not (thenode.children==[]):
				positions = thenode.children[0].data[0]
				attr = positions[level]
				trainingattribute = j[attr]
				truth=False
				for l in thenode.children:
					values= l.data[1]
					if (str(values[level])==trainingattribute):
						thenode = l
						truth=True
						break
				if not truth:
					break
				level+=1
			if (str(thenode.data[3])==j[-1]):
				correct+=1
		test_accuracy.append(correct/test_total)
		print('test accuracy:',correct/test_total)


		correct=0	
		for j in list_valid[1:-1]:
			thenode = root
			level = 0
			while not (thenode.children==[]):
				positions = thenode.children[0].data[0]
				attr = positions[level]
				trainingattribute = j[attr]
				truth=False
				for l in thenode.children:
					values= l.data[1]
					if (str(values[level])==trainingattribute):
						thenode = l
						truth=True
						break
				if not truth:
					break
				level+=1
			if (str(thenode.data[3])==j[-1]):
				correct+=1
		validation_accuracy.append(correct/valid_total)		
		print('validation set accuracy:',correct/valid_total)


		print(prunethisnode.data[1],current_accuracy)
		print('index:',index)
		print('.....')
		level=0
		thenode=root
		parent = Node([])
		while not thenode==n:
			truth=False
			# print(len(thenode.children))
			for p in thenode.children:
				val=p.data[1][level]
				if (n.data[1][level]==val):
					truth=True
					parent=thenode
					thenode=p 
					break
			if not truth:
				break
			level+=1
		# print('parent:',parent.data[1])
		# print('remove:',thenode.data[1])	
		parent.remove_child(thenode)	
		prunethisnode=Node([])
		getNode(node)	
		n=prunethisnode	
		index+=1




pruneTree(root)

print('Decision tree is pruned!')

plt.plot(numbernodes,train_accuracy, color='red')
plt.plot(numbernodes,test_accuracy, color='green')
plt.plot(numbernodes,validation_accuracy, color='blue')
plt.xlabel("Number of nodes")
plt.ylabel("Accuracy")
plt.show()


# print(train_accuracy)
# print(test_accuracy)
# print(validation_accuracy)
# Now to test for train accuracy:
# maxlevel = (root.height())
# maxlevel =2
# plt.plot(numbernodes,train_accuracy)
# plt.plot(numbernodes,test_accuracy)
# plt.plot(numbernodes,validation_accuracy)
# plt.show()
