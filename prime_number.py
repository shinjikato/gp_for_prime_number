import time
import numpy as np

def create(N=100,start_index=0):
	primes = [2, 3]
	n = 5
	while len(primes) < N:
		isprime = True
		for p in primes:
			if n % p == 0:
				isprime = False
				break
		if isprime:
			primes.append(n)
		n += 2
	X = np.arange(N).reshape(N,1) + start_index
	y = np.array(primes)
	return X,y
