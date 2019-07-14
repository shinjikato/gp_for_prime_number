import numpy as np
import base

class GPRegressor:
	def __init__(self, max_generation=200, pop_size = 500,
		tour_size=64, batch_size=8, CXPB=0.8, MUTPB=0.1, limit=18):
		self.max_generation = max_generation
		self.pop_size = pop_size
		self.tour_size = tour_size
		self.batch_size = batch_size
		self.CXPB = CXPB
		self.MUTPB = MUTPB
		self.limit = limit
	def fit(X,y):
		self.leafSet, self.nodeSet = base.makeNodeSet()
		pop = base.initial_population(self.pop_size, self.nodeSet, self.leafSet)
		elite = None
		for g in range(self.max_generation):
			pop = base.evaluation(pop, X, y)
			pop = base.selection(pop, elite, batch_size=8, tour_size=64)
			pop = base.crossover(pop, self.CXPB, self.limit)
			pop = base.mutation(pop, self.MUTPB, self.nodeSet, self.leafSet)
		pop = base.evaluation(pop, X, y)
		pop = [tree for tree in pop if tree.fitness != None]
		elite = min(pop, key = lambda tree:tree.fitness)
		self.tree = elite
		return self

	def predict(X):
		return self.tree.run(X)
	def score(X,y):
		_y = self.predict(X)
		s = np.sum((y-_y)**2)/len(y)
		return -1*s

	def get_params(self, deep=True):
		return {"max_generation":self.max_generation,
				"pop_size":self.pop_size,
				"tour_size":self.tour_size,
				"batch_size":self.batch_size,
				"CXPB":self.CXPB,
				"MUTPB":self.MUTPB,
				"limit":self.limit}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self.params,parameter, value)
		return self
