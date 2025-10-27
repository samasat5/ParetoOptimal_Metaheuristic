import math
import numpy as np
import pdb


def miseAJour(YND,solution): 
	new_sol = []
	[xStart,vStart] = solution
	dominates=True
	i=0
	while i < len(YND) and dominates:
		sol=YND[i] 
		if (vStart[0] >= sol[1][0] and vStart[1] >= sol[1][1]) and (vStart[0] > sol[1][0] or vStart[1] > sol[1][1]):
			# print("POP! la solution courante domine la solution sol")
			YND.pop(i) 
		else: 
			if (sol[1][0] >= vStart[0] and sol[1][1] >= vStart[1]) and (sol[1][0] > vStart[0] or sol[1][1] > vStart[1]):
				# la solution sol domine la solution courante
				dominates=False
			else:
				i=i+1
	if dominates:
		YND.append(solution)
		new_sol.append(vStart)
	return dominates
			
from bisect import bisect_left

def firstObj_miseAJour(YND, solution):
    x_new, v_new = solution
    a = int(v_new[0])  # f1
    b = int(v_new[1])  # f2

    # empty archive → insert
    if not YND:
        YND.append([x_new.copy(), v_new.copy()])
        return True 
    # locate by f1 using binary search
    f1_list = [int(sol[1][0]) for sol in YND]
    idx = bisect_left(f1_list, a)   # first position with f1 >= a 
    # dominated by the right neighbor (or equal point)? reject
    if idx < len(YND):
        f1_r = int(YND[idx][1][0]); f2_r = int(YND[idx][1][1])
        if (f1_r > a and f2_r >= b) or (f1_r == a and f2_r >= b):
            return False  # dominated or duplicate with better/equal f2

    #  remove the dominated block to the LEFT (those with f2 <= b)
    k = idx - 1
    while k >= 0 and int(YND[k][1][1]) <= b:
        YND.pop(k)
        idx -= 1
        k   -= 1

    # guard against exact duplicate at insertion point
    if idx < len(YND):
        if int(YND[idx][1][0]) == a and int(YND[idx][1][1]) == b:
            return False  # exact duplicate
    YND.insert(idx, [x_new.copy(), v_new.copy()])
    return True

	






def secondObj_miseAJour(YND,solution): 
	# solution est de la forme [x,v] avec x la solution binaire et v le vecteur des valeurs
	# print("YND before",YND)
 
	new_sol = []
	[xStart,vStart] = solution
	dominates=True
	i=0
	while i < len(YND) and dominates:
	
		sol=YND[i] 
		if (vStart[1] >= sol[1][1] ) and (vStart[1] > sol[1][1]):
			# print("POP! la solution courante domine la solution sol")
			YND.pop(i) 
	
		else: 
			if (sol[1][0] >= vStart[0]) and (sol[1][0] > vStart[0]):
				# la solution sol domine la solution courante
				dominates=False
			else:
				i=i+1
		
	if dominates:
		YND.append(solution)
		new_sol.append(vStart)
	# print("solutions added:", new_sol)

	return dominates
   
   
    
    
    



def proportion(YN,YApprox):
	cpt=0
	for y in YN:	
		for sol in YApprox:
			if np.array_equal(y,sol[1][:]):
				cpt=cpt+1
				break
	return cpt/YN.shape[0]
			

def distanceEuclidienne(y1,y2,poids,p):
	d=0
	for j in range(p):
		d = d + math.sqrt(poids[j] * (y1[j] - y2[j])**2)
	return d 
    
def dprime(YApprox,y,poids,p):    
	minV = 9999999	
	for sol in YApprox:
		dist=distanceEuclidienne(y,sol[1][:],poids,p)
		if dist < minV:
			minV=dist
	return minV
	
	
def DM(YN,YApprox,p):
	Nadir = np.zeros(p,dtype=int)
	Ideal = np.zeros(p,dtype=int)
	for j in range(p):
		Nadir[j]=min(YN[:,j])
	for j in range(p):
		Ideal[j]=max(YN[:,j])
		
	poids = np.zeros(p)
	for j in range(p):
		poids[j] = 1/abs(Ideal[j]-Nadir[j])
	
	d=0
	for y in YN:
		d=d+dprime(YApprox,y,poids,p)
	
	return d/YN.shape[0]


	
