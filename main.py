import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain, combinations
import pdb
from read_file import *
from indicators import *
import time

np.random.seed(0)
numInstance=0
n=100
p=2

for n in range(100, 800, 100):
	for numInstance in range(0, 10):
		w=np.zeros(n,dtype=int)
		v=np.zeros((n,p),dtype=int)
		filename = "Data/"+str(n)+"_items/2KP"+str(n)+"-TA-"+str(numInstance)+".dat"
		with open("tableau_result.txt", "a") as f:
				f.write(filename[:-4]+"\n")

		capacity, weights, values =readFile(filename,w,v)

		#Lecture des point non-dominées

		filename = "Data/"+str(n)+"_items/2KP"+str(n)+"-TA-"+str(numInstance)+".eff"
		YN=readPoints(filename,p) # YN est la matrice des points non-dominées (la vraie frontiere)

		plt.grid()
		plt.scatter(YN[:,0],YN[:,1],color='blue')

		YND=[]  #YND est la liste des solutions non-dominées (approximation)


		##################################################
		# Naive Random Sampling :
		##################################################
		#Génération de m solutions aléatoires : 


		m=100
		xStart = None
		YND=[]  
		random_feasible_solutions = []
		for i in range(m):
			xStart=np.zeros(n,dtype=int) 
			arr = np.arange(n)  
			np.random.shuffle(arr) 
			wTotal=0 
			vStart=np.zeros(p,dtype=int) 
			for i in range(n): 	
				if wTotal+w[arr[i]]<=capacity: 
					xStart[arr[i]]=1    
					wTotal=wTotal+w[arr[i]] 
					for j in range(p):
						vStart[j]=vStart[j]+v[arr[i],j] 
						v1,v2 = vStart[0], vStart[1]
			solution = [xStart,vStart]
			random_feasible_solutions.append(solution)
			dominates = miseAJour(YND,[xStart,vStart])
			

		"""
		
		# print(YND)
		# print(YN)

		for sol in YND:
			plt.scatter(sol[1][0],sol[1][1],color='red')

		# plt.show()

		#Calcule de la proportion

		print("Proportion = ",proportion(YN,YND))

		#Calcule de la distance DM

		print("DM =",DM(YN,YND,p))

		"""



		##################################################
		# PLS 1 :
		##################################################

		# # initial population based on performance ratio:
		"""
		print("PLS 1:")
		with open("tableau_result.txt", "a") as f:
			f.write("PLS 1:"+"\n")

		m=100
		p0 = []
		YND=[] 
		time_start = time.time() 	
		for q_val in np.linspace(0.0, 1.0, m):
			performance_scores = (q_val*values[:,0] + (1 - q_val)*values[:,1]) / weights # performance ratio R(i)
			order = np.argsort(-performance_scores)  
			x=np.zeros(n,dtype=int)
			wTotal=0
			v=np.zeros(p,dtype=int)
			for i in range(n):
				idx = order[i]
				if wTotal + weights[idx] <= capacity:
					x[idx] = 1
					wTotal += weights[idx]
					for j in range(p):
						v[j] += values[idx,j]
			p0.append([x, v])
			dominates = miseAJour(YND,[xStart,v])

		def voisins_1_1_faisables(xStart, vStart, w, v, capacity):
			n = len(w)
			wt = int(np.dot(w, xStart))
			ones  = np.where(xStart == 1)[0]
			zeros = np.where(xStart == 0)[0]

			for i in ones:          # take out i
				for j in zeros:     # add  in j
					new_wt = wt - w[i] + w[j]  # we change two items (remove one and add one) because we want the neighbors  (hence 1-1)
					if new_wt <= capacity:
						x2 = xStart.copy()
						x2[i] = 0
						x2[j] = 1
						v2 = vStart.copy()
						v2[0] = v2[0] - v[i,0] + v[j,0]
						v2[1] = v2[1] - v[i,1] + v[j,1]
						yield x2, v2
						
		XE = [ [x.copy(), v.copy()] for (x, v) in p0 ]
		P  = [ [x.copy(), v.copy()] for (x, v) in p0 ]
		pa = []	
		while len(P) > 0:	
			for solution in P:
				x1, v1 = solution
				for solution_prime in voisins_1_1_faisables(x1, v1, weights, values, capacity):
					x2, v2 = solution_prime 
					if (v1[0] >= v2[0] and v1[1] >= v2[1]) and (v1[0] > v2[0] or v1[1] > v2[1]): 
						new_solution = [x2, v2]
						dominates =  miseAJour(XE, [x2, v2])
						if dominates:
							miseAJour(pa, [x2, v2])
			
			P = pa
			pa = []	
		time_end = time.time() 
		"""              
		"""
		plt.figure()
		plt.grid()
		plt.scatter(YN[:,0], YN[:,1], color='blue', s=10, label='True Pareto (YN)')
		for (xsol, vsol) in XE:
			plt.scatter(vsol[0], vsol[1], color='red', s=12)
		plt.legend()
		plt.show()
  
		"""
		"""
		print("Proportion =", proportion(YN, XE))
		print("DM =", DM(YN, XE, p))
		print("Archive size (XE):", len(XE))
		print("Time taken (PLS 1):", time_end - time_start, "seconds")
		with open("tableau_result.txt", "a") as f:
			f.write("Proportion ="+ str(proportion(YN, XE)) +" ")
			f.write("DM ="+ str(DM(YN, XE, p)) + " ")
			f.write("Archive size (XE):" + str(len(XE)) + " ")
			f.write("Time taken (PLS 1):"+ str(time_end - time_start)+ "seconds\n")
		"""
		##################################################
		# PLS 2 :
		##################################################
		print("PLS 2:")
		with open("tableau_result.txt", "a") as f:
			f.write("PLS 2:"+"\n")

		def voisins_1_1_faisables(xStart, vStart, w, v, capacity):
			n = len(w)
			wt = int(np.dot(w, xStart))
			ones  = np.where(xStart == 1)[0]
			zeros = np.where(xStart == 0)[0]

			for i in ones:          # take out i
				for j in zeros:     # add  in j
					new_wt = wt - w[i] + w[j]  # we change two items (remove one and add one) because we want the neighbors  (hence 1-1)
					if new_wt <= capacity:
						x2 = xStart.copy()
						x2[i] = 0
						x2[j] = 1
						v2 = vStart.copy()
						v2[0] = v2[0] - v[i,0] + v[j,0]
						v2[1] = v2[1] - v[i,1] + v[j,1]
						yield x2, v2
		# initial population based on performance ratio:
		m=100
		p0 = []
		XE=[]  	
		time_start = time.time()
		for i in range(m):
			q_val=np.random.rand()
			performance_scores = ((q_val*values[:,0] + (1 - q_val)*values[:,1]) / weights) # performance ratio R(i)
			order_idx_descending = np.argsort(-performance_scores) # descending order
			x_=np.zeros(n,dtype=int)
			wTotal=0
			v_=np.zeros(p,dtype=int)
			for i in range(n):
				idx = order_idx_descending[i]
				if wTotal + weights[idx] <= capacity:
					x_[idx] = 1
					wTotal += weights[idx]
					for j in range(p):
						v_[j] += values[idx,j]
			p0.append([x_, v_])
			dominates = firstObj_miseAJour(XE,[x_,v_])

		P  = [ [x.copy(), v.copy()] for (x, v) in p0 ]
		pa = []	
		while len(P) > 0:	
			for solution in P:
				x1, v1 = solution
				for solution_prime in voisins_1_1_faisables(x1, v1, weights, values, capacity):
					x2, v2 = solution_prime 
					if not ((v1[0] >= v2[0] and v1[1] >= v2[1]) and (v1[0] > v2[0] or v1[1] > v2[1])): 
						new_solution = [x2, v2]
						dominates =  firstObj_miseAJour(XE, [x2, v2])
						if dominates:
							firstObj_miseAJour(pa, [x2, v2])
			
			P = pa
			pa = []	
		time_end = time.time()
		       
		"""
		plt.figure()
		plt.grid()
		plt.scatter(YN[:,0], YN[:,1], color='blue', s=10, label='True Pareto (YN)')
		for (xsol, vsol) in XE:
			plt.scatter(vsol[0], vsol[1], color='red', s=12)
		plt.legend()
		plt.show()
		"""
		
		print("Proportion =", proportion(YN, XE))
		print("DM =", DM(YN, XE, p))
		print("Archive size (XE):", len(XE))
		print("Time taken (PLS 2):", time_end - time_start, "seconds")
		"""
		with open("tableau_result.txt", "a") as f:
			f.write("Proportion ="+ str(proportion(YN, XE)) +" ")
			f.write("DM ="+ str(DM(YN, XE, p)) + " ")
			f.write("Archive size (XE):" + str(len(XE)) + " ")
			f.write("Time taken (PLS 2):"+ str(time_end - time_start)+ "seconds\n")
		"""
		
		##################################################
		# PLS 3 :
		##################################################

		def powerset(iterable):
			"""Renvoie tous les sous-ensembles d'un iterable"""
			s = list(iterable)
			return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

		# initial population based on performance ratio:
		print("PLS 3:")
		with open("tableau_result.txt", "a") as f:
			f.write("PLS 3 :\n")
		m=100
		# using the previous p0 from PLS 2
		def voisins_faisables_L1l2(xStart, weights, values, capacity, q_val=0.5, L=4): # worst and best L items 
			performance_scores = (q_val*values[:,0] + (1 - q_val)*values[:,1]) / weights # performance ratio R(i)
			order_idx_descending = np.argsort(-performance_scores)
			in_idx = np.where(xStart == 1)[0]
			out_idx = np.where(xStart == 0)[0]	

			ratio_in  = (q_val*values[in_idx,0]  + (1-q_val)*values[in_idx,1])  / weights[in_idx] # all items inside
			ratio_out = (q_val*values[out_idx,0] + (1-q_val)*values[out_idx,1]) / weights[out_idx] # all items outside
			L1 = in_idx[np.argsort(ratio_in)[:min(L, len(in_idx))]] # worst L items inside
			L2 = out_idx[np.argsort(-ratio_out)[:min(L, len(out_idx))]] # best L items outside

			n = len(weights)
			S = L1 + L2
			S_set = set(S)

			F = [i for i in range(n) if (i not in S_set and xStart[i] == 1)]
			W_base = int(np.sum(weights[F])) if F else 0
			V_base = values[F].sum(axis=0) if F else np.zeros(2, dtype=int)
			Wprime = capacity - W_base
			neighbors = []
			# Map S indices to positions in bit vector
			k = len(S)
			# all_bits = list(itertools.product([0,1], repeat=k))
			# # for combo in sample(all_bits, min(20, len(all_bits))):
			wt0 = int(np.dot(weights, xStart))
			v0  = values.T @ xStart

			for i in L1:
				for j in L2:
					new_wt = wt0 - weights[i] + weights[j]
					if new_wt <= capacity:
						x2 = xStart.copy()
						x2[i] = 0
						x2[j] = 1
						v2 = v0 - values[i] + values[j]
						yield x2, v2


						
						
		time_start = time.time()             
		XE = [ [x.copy(), v.copy()] for (x, v) in p0 ]
		P  = [ [x.copy(), v.copy()] for (x, v) in p0 ]
		q_val = 0.6
		L = 4
		while len(P) > 0:	
			pa = []
			for solution in P:
				x1, v1 = solution
				for solution_prime in voisins_faisables_L1l2(x1, weights, values, capacity, q_val, L)	:

					x2, v2 = solution_prime 
					if not ((v1[0] >= v2[0] and v1[1] >= v2[1]) and (v1[0] > v2[0] or v1[1] > v2[1])): 
						dominates =  firstObj_miseAJour(XE, [x2, v2])
						if dominates:
							firstObj_miseAJour(pa, [x2, v2])
			
			P = pa
			pa = []	
		time_end = time.time() 
		"""              
		plt.figure()
		plt.grid()
		plt.scatter(YN[:,0], YN[:,1], color='blue', s=10, label='True Pareto (YN)')
		for (xsol, vsol) in XE:
			plt.scatter(vsol[0], vsol[1], color='red', s=12)
		plt.legend()
		plt.show()
		"""
		print("Proportion =", proportion(YN, XE))
		print("DM =", DM(YN, XE, p))
		print("Archive size (XE):", len(XE))
		print("Time taken (PLS 3):", time_end - time_start, "seconds")
		"""
		with open("tableau_result.txt", "a") as f:
			f.write("Proportion ="+ str(proportion(YN, XE)) +" ")
			f.write("DM ="+ str(DM(YN, XE, p)) + " ")
			f.write("Archive size (XE):" + str(len(XE)) + " ")
			f.write("Time taken (PLS 3):"+ str(time_end - time_start)+ "seconds\n")
		"""




