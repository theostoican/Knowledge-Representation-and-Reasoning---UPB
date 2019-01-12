import itertools
import random
import math

def sigmoid(x):
	#print(x)
	return 1 / (1 + math.exp(-x))

net = {}
net['A'] = []
net['B'] = []
net['C'] = ['A']
net['D'] = ['A', 'B']
net['E'] = ['C', 'D']
net['F'] = ['E']
net['G'] = ['B']
net['H'] = []
net['I'] = ['G', 'H']
net['J'] = ['I']
net['K'] = ['F', 'J']
net['L'] = []
net['M'] = []
net['N'] = ['K', 'L']
net['O'] = ['L', 'M']

if __name__ == "__main__":


	# Initialize theta with random values
	theta = {}
	
	for var in net:
		numPa = len(net[var])
		combs = list(itertools.product([0, 1], repeat=numPa))
		#print(var)

		for comb in combs:
			key = ""
			
			for valPa in comb:
				key += str(valPa)
			
			if var not in theta:
				theta[var] = {}

			theta[var][key] = random.randint(1, 2)
	#print(theta)
	

	for it in range(0, 15):
		with open('samples_bn1', 'r') as fin:

			cnt = 0
			lr = 0.01
			vars = None

			# For each sample
			for line in fin:
				if cnt == 0:
					vars = line.split()
				else:
					vals = line.split()
					pairs = {}

					# Association between parents variables and values
					for var, val in zip(vars, vals):
						pairs[var] = val

					# For each variable
					for var in net:

						# Compute the key based on the parents' values
						parVals = ""
						for parent in net[var]:
							parVals += pairs[parent]

						# Gradient descent
						#print(cnt)
						#if cnt == 50:
						#	print(theta[var][parVals])
						
						theta[var][parVals] = theta[var][parVals] + lr * (int(pairs[var]) - sigmoid(theta[var][parVals]))
						#if theta[var][parVals] < 0:
							#print(theta[var])

				cnt += 1
		
	for var in theta:
		print(var)
		for comb in theta[var]:
			print(sigmoid(theta[var][comb]))
	#print(theta)