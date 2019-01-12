import itertools

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


countOnes = {}
count = {}

estimPar = {}

if __name__ == "__main__":
	with open('samples_bn1', 'r') as fin:

		cnt = 0
		vars = None

		for line in fin:
			if cnt == 0:
				vars = line.split()
			else:
				vals = line.split()
				pairs = {}

				for var, val in zip(vars, vals):
					pairs[var] = val

				#print(pairs)
				for var in net:
					parVals = ""
					for parent in net[var]:
						parVals += pairs[parent]

					if var not in count:
						count[var] = {}
						countOnes[var] = {}

					if parVals not in count[var]:
						count[var][parVals] = 0
						countOnes[var][parVals] = 0

					count[var][parVals] += 1
					if pairs[var] == '1':
						countOnes[var][parVals] += 1

			cnt += 1

		for var in net:
			numPa = len(net[var])
			combs = list(itertools.product([0, 1], repeat=numPa))

			probs = []

			print(var)

			for comb in combs:
				key = ""
				
				for valPa in comb:
					key += str(valPa)

				cntOne = 0
				cnt = 0

				if var in count and key in count[var]:
					cnt = count[var][key]
					cntOne = countOnes[var][key]

				smooth = float(cntOne + 1) / float(cnt + 2)


				probs.append(smooth)

			print(probs)