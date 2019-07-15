import numpy as np
import base
import prime_number
import random

class GPRegressor:
	def __init__(self, max_generation=200, pop_size = 500,
		tour_size=64, batch_size=8, CXPB=0.8, MUTPB=0.1, limit=18, printlog=False):
		self.max_generation = max_generation
		self.pop_size = pop_size
		self.tour_size = tour_size
		self.batch_size = batch_size
		self.CXPB = CXPB
		self.MUTPB = MUTPB
		self.limit = limit
		self.printlog = printlog
	def geneinfo(self,pop,g):
		fits = [tree.fitness for tree in pop if tree.fitness != None]
		max_fit = max(fits)
		min_fit = min(fits)
		ave_fit = sum(fits)/len(fits)
		return "{:0=6} : max {:.6f}, ave {:.6f}, min {:.6f}".format(g,max_fit,ave_fit,min_fit)

	def fit(self, X, y):
		self.nodeSet, self.leafSet  = base.makeNodeSet()
		pop = base.initial_population(self.pop_size, self.nodeSet, self.leafSet)
		elite = None
		for g in range(self.max_generation):
			pop = base.evaluation(pop, X, y)
			if self.printlog: print(self.geneinfo(pop,g))
			pop,elite = base.selection(pop, elite, batch_size=8, tour_size=64)


			pop = base.crossover(pop, self.CXPB, self.limit)
			pop = base.mutation(pop, self.MUTPB, self.nodeSet, self.leafSet)
		pop = base.evaluation(pop, X, y)
		pop = [tree for tree in pop if tree.fitness != None]
		elite = min(pop, key = lambda tree:tree.fitness)
		self.tree = elite
		return self

	def predict(self, X):
		return self.tree.run(X)
	def score(self, X, y):
		_y = self.predict(X)
		s = np.sum(np.abs(y-_y))/len(y)
		return -1*s

	def get_params(self, deep=True):
		return {"max_generation":self.max_generation,
				"pop_size":self.pop_size,
				"tour_size":self.tour_size,
				"batch_size":self.batch_size,
				"CXPB":self.CXPB,
				"MUTPB":self.MUTPB,
				"limit":self.limit,
				"printlog":self.printlog}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self.params,parameter, value)
		return self

if __name__ == "__main__":# test code
	random.seed(0)
	np.random.seed(0)
	N = 10000

	X, y = prime_number.create(N=int(N/5),start_index=1)
	test_X, test_y = prime_number.create(N=N,start_index=1)
	model = GPRegressor(pop_size=1000,printlog=True,max_generation=1000)
	model.fit(X, y)
	print(model.tree.toTexText())
	print(model.score(X, y))
	print(model.score(test_X, test_y))

	from sympy import sympify, simplify, latex
	f = sympify(model.tree.toSympyInputText())
	print(latex(f))
	f_ = simplify(f)
	print(latex(f_))

	import matplotlib.pyplot as plt
	x = test_X.flatten()
	plt.plot(x,test_y,label="true prime number")
	_y = model.predict(test_X)
	plt.plot(x,_y,label="test  predict")
	_y = model.predict(X)
	x = X.flatten()
	plt.plot(x,_y,label="train predict")
	plt.legend(loc="upper left")
	#plt.xticks(np.arange(1,))
	plt.xlim([1,N])
	plt.xlabel("x = index")
	plt.ylabel("y = f(x) : f is prime number function")
	plt.savefig("predict_graph.pdf")



