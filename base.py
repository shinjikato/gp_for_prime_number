import numpy as np
import random
from copy import deepcopy

class Tree(list):
	def __init__(self, content):
		list.__init__(self, content)
		self.fitness = None
		self.errors = None
	@property
	def height(self):
		stack = [0]
		max_depth = 0
		for elem in self:
			depth = stack.pop()
			max_depth = max(max_depth, depth)
			stack.extend([depth+1]*node[2])
		return max_depth
	def subtree(self, begin):
		end = begin + 1
		total = self[begin][2]
		while total > 0:
			total += self[end][2] - 1
			end += 1
		return slice(begin, end)
	def __str__(self):
		stack = []
		for node in self[::-1]:
			name,func,arity,const = node
			stack.append(name +  "("+ ",".join([stack.pop() for _ in range(arity)])+")" if arity!=0 else name)
		return stack[0]
	def run(X):
		x = X.T
		stack = []
		for node in self[::-1]:
			name,func,arity,const = node
			if name == "x":
				func_ret = x
			elif arity == 0:
				func_ret = np.ones(len(x)) * const
			else:
				func_ret = func(*[stack.pop() for _ in range(arity)])
			stack.append(func_ret)
		return stack[0]

def createTree(nodeSet,leafSet):
	height = random.randint(5,10)
	items = []
	stack = [0]
	while len(stack) != 0:
		depth = stack.pop()
		if depth == height:
			node = random.choice(leafSet)
		else:
			node = random.choice(nodeSet)
			stack.extend([depth+1]*node[2])
		items.append(node)
	tree = Tree(items)
	return tree

def swap_tree(treeA, treeB, limit):
	if len(treeA) == 1 or len(treeB) ==0:
		return treeA, treeB
	for _ in range(100):#safty loop
		s_A = random.randrange(1,len(treeA))
		s_B = random.randrange(1,len(treeB))
		e_A = treeA.subtree(s_A)
		e_B = treeA.subtree(s_B)
		treeA[s_A:e_A],treeB[s_B:e_B] = treeB[s_B:e_B],treeA[s_A:e_A]
		if treeA.height <= limit and treeB.height <= limit:
			break
		treeA[s_A:s_A+(e_B-s_B)],treeB[s_B:s_B+(e_A-s_A)] =treeB[s_B:s_B+(e_A-s_A)],treeA[s_A:s_A+(e_B-s_B)]
	return treeA, treeB

def point_change(tree, nodeSet, leafSet, MUTPB):
	for i,node in enumerate(tree):
		if random.random() < MUTPB:
			name,func,arity,const = node
			if arity == 0:
				tree[i] = random.choice(leafSet)
			else:
				tree[i] = random.choice([_node for _node in nodeSet if _node[2]==arity])
	return tree

def makeNodeSet():
	nodeSet = []
	leafSet = []
	nodeSet.append(("add",	np.add,			2,	None))
	nodeSet.append(("sub",	np.subtract,	2,	None))
	nodeSet.append(("mul",	np.multiply,	2,	None))
	nodeSet.append(("div",	np.divide,		2,	None))
	nodeSet.append(("sin",	np.sin,			1,	None))
	nodeSet.append(("cos",	np.cos,			1,	None))
	nodeSet.append(("tan",	np.tan,			1,	None))
	nodeSet.append(("log",	np.log,			1,	None))
	nodeSet.append(("exp",	np.exp,			1,	None))
	nodeSet.append(("sqrt",	np.sqrt,		1,	None))

	leafSet.append(("-1",	None,			0,	-1))
	leafSet.append(("0",	None,			0,	0))
	leafSet.append(("1",	None,			0,	1))
	leafSet.append(("0.5",	None,			0,	0.5))
	leafSet.append(("pi",	None,			0,	np.pi))
	leafSet.append(("e",	None,			0,	np.e))
	leafSet.append(("x",	None,			0,	None))

	return nodeSet,leafSet

def initial_population(N, nodeSet, leafSet):
	return [createTree(nodeSet,leafSet) for _ in range(N)]

def evaluation(pop, X, y):
	for tree in pop:
		if tree.fitness == None:
			_y = tree.run(X,y)
			errors = (y-_y)**2
			fitness = np.sum(errors)/len(y)
			if np.isfinite(fitness):
				tree.fitness = fitness
				tree.errors = errors
			else:
				tree.fitness = None
	return pop

def selection(pop,elite, batch_size=8, tour_size=64):
	pop_size = len(pop)
	pop = [tree for tree in pop if tree.fitness != None]
	best = min(pop, key=lambda tree:tree.fitness)
	if elite.fitness == None or best.fitness < elite.fitness:
		elite = deepcopy(best)
	T = np.arange(len(pop[0].errors))
	np.random.shuffle(T)
	t_start = 0
	next_pop = []
	for _ in range(pop_size):
		best = None
		best_ave = 0
		subpop = random.sample(pop,tour_size) if len(pop) > tour_size else pop
		t_end = t_start + batch_size if (t_start + batch_size) < len(T) else len(T)
		for i,tree in enumerate(subpop):
			ave = sum([tree.errors[t] for t in T[t_start:t_end]])/batch_size
			if best == None or ave < best_ave:
				best = tree
				best_ave = ave
		next_pop.append(deepcopy(best))
		t_start = t_end
		if t_start == len(T):
			t_start = 0
	pop.append(deepcopy(elite))
	return pop, elite


def crossover(pop, CXPB, limit):
	random.shuffle(pop)
	for i in range(int(len(pop)/2)):
		if random.random() < CXPB:
			pop[i*2], pop[i*2+1] = swap_tree(pop[i*2], pop[i*2+1],limit)
			pop[i*2].fitness = None
			pop[i*2+1].fitness = None
	return pop

def mutation(pop, MUTPB, nodeSet, leafSet):
	random.shuffle(pop)
	for i in range(int(len(pop)/2)):
		pop[i] = point_change(pop[i], nodeSet, leafSet, MUTPB)
		pop[i].fitness = None
	return pop
