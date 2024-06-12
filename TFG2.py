# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:14:41 2024

@author: Usuario
"""
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

@njit
def build_supra_adjacency_matrix(A_1, A_2, p, gamma):
    n = A_1.shape[0]

    # Create the off-diagonal blocks
    identity_block = np.eye(n)
    off_diagonal_block1 = p * identity_block
    off_diagonal_block2 = p * identity_block

    # Construct the final 2Nx2N matrix
    top_row = np.hstack((A_1, off_diagonal_block1))
    bottom_row = np.hstack((off_diagonal_block2, A_2))
    result_matrix = np.vstack((top_row, bottom_row))

    return result_matrix

def build_supra_contact_matrix(A_1, A_2, p, gamma):
    n = A_1.shape[0]

    # Calculate row sums
    k_1 = np.sum(A_1, axis=1)
    k_2 = np.sum(A_2, axis=1)
    #print(k_1)
    #print(k_2)

    # Calculate diagonal blocks
    R_1 = 1 - np.power(1 - A_1 / k_1[:, np.newaxis], gamma)
    R_2 = 1 - np.power(1 - A_2 / k_2[:, np.newaxis], gamma)

    # Create the off-diagonal blocks
    identity_block = np.eye(n)
    off_diagonal_block1 = p * identity_block
    off_diagonal_block2 = p * identity_block

    # Construct the final 2Nx2N matrix
    top_row = np.hstack((R_1, off_diagonal_block1))
    bottom_row = np.hstack((off_diagonal_block2, R_2))
    result_matrix = np.vstack((top_row, bottom_row))

    return result_matrix

def leading_eigenvector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the leading eigenvalue
    leading_index = np.argmax(eigenvalues)

    # Extract the leading eigenvector
    leading_eigenvector = eigenvectors[:, leading_index]

    return leading_eigenvector, eigenvalues[leading_index]


def IPR(vector):
    N = len(vector) // 2
    IPR1 = sum(entry ** 4 for entry in vector[:N])
    IPR2 = sum(entry ** 4 for entry in vector[N:])
    return IPR1, IPR2

def build_diferent_gammas_matrix(A_1, A_2, p, gamma1,gamma2):
    n = A_1.shape[0]

    #matrix 1
    if gamma1=="inf":
      R_1=A_1
    else:
      # Calculate row sums
      k_1 = np.sum(A_1, axis=1)
      # Calculate diagonal blocks
      R_1 = 1 - np.power(1 - A_1 / k_1[:, np.newaxis], gamma1)

    if gamma2=="inf":
      R_2=A_2
    else:
      # Calculate row sums
      k_2 = np.sum(A_2, axis=1)
      # Calculate diagonal blocks
      R_2 = 1 - np.power(1 - A_2 / k_2[:, np.newaxis], gamma2)


    # Create the off-diagonal blocks
    identity_block = np.eye(n)
    off_diagonal_block1 = p * identity_block
    off_diagonal_block2 = p * identity_block

    # Construct the final 2Nx2N matrix
    top_row = np.hstack((R_1, off_diagonal_block1))
    bottom_row = np.hstack((off_diagonal_block2, R_2))
    result_matrix = np.vstack((top_row, bottom_row))

    return result_matrix

def QSsimulation2(N,gamma,eta,InPr,k1,k2):
    dim = 2 
    
    #Defining the probabilities that define the evolution of the system
    if eta*InPr<=1:
        nu = 1
        InPr=InPr*nu
        ExPr = eta * InPr
    else :
        ExPr = 1
        nu=1/(eta * InPr)
        InPr = 1/eta
       
    
    
    #Defining variables that control Quasistationary Simulation
    M = 1000
    Prep = 10**(-2)
    
    #Defining variables that control the density and the time-steps
    nsum=800
    Ftime = 3000

    #Generation of erdorenyi graphs with k1 and k2 espected value of neighbours
    G1_2=nx.erdos_renyi_graph(N,k1/N)
    G2_2=nx.erdos_renyi_graph(N,k2/N)
    while nx.is_connected(G1_2)!=True:
          G1_2=nx.erdos_renyi_graph(N,k1/N)
    while nx.is_connected(G2_2)!=True:
          G2_2=nx.erdos_renyi_graph(N,k2/N)

    # Defining adjacency matrix
    A_1_2 = nx.to_numpy_array(G1_2)
    A_2_2 = nx.to_numpy_array(G2_2)
    # Building supracontact probability matrix
    r = build_diferent_gammas_matrix(A_1_2, A_2_2, eta,'inf',gamma)

    #Defining the starting state 
    States=np.zeros((N,dim))
    SaveStates=np.zeros((M,N,dim))
    States[0,:]=1
    States_1=np.copy(States)
    t=0

    #Defining all of the densities
    rho = np.ones(Ftime)
    rho1 = np.ones(Ftime)
    rho2 = np.ones(Ftime)
    rhox2 = np.ones(Ftime)
    rhox21 = np.ones(Ftime)
    rhox22 = np.ones(Ftime)   

    x=0


    #Start of the loop
    while t<Ftime:
        States=np.copy(States_1)
        
        f=t%50
        
        #Write densities for each time
        rho[t] = np.sum(States) / (N * 2)
        rho1[t] = np.sum(States[:,0])/N
        rho2[t] = np.sum(States[:,1])/N
        rhox2[t]=(np.sum(States)/(N*2)) ** 2
        rhox21[t]=(np.sum(States[:,0])/N)**2
        rhox22[t]=(np.sum(States[:,1])/N)**2
        
        
        #Test to see if we are in the equilibrium
        if t>=200 and f==0 and Ftime-nsum>t:
            cs = np.sum(rho1[t-50:t])-np.sum(rho1[t-100:t-50])
            cs2 =  np.sum(rho2[t-50:t])-np.sum(rho2[t-100:t-50])
            
            # If we are at the equilibrium start taking measures simulate the lasts steps.
            if 0.05>cs and 0.05>cs2:
                t=Ftime-nsum-1
        
        #Quasi-stationary control
        #We save states
        
        if t<M:
            SaveStates[t]=np.copy(States)
        if t>=M and random.random()<=Prep:
            k=random.randint(0,M-1)
            SaveStates[k]=np.copy(States)
        #If we get to the absorbing state we change it for another past state
        if rho[t]==0:
            while x==0:
                k=random.randint(0,M-1)
                if np.sum(SaveStates[k,:,:]) != 0 :
                    States=np.copy(SaveStates[k])
                    x=1
            x=0
        
        #Denfining the random numbers to use
        rn_nu = np.random.random((N, dim))
        rn_ExPr = np.random.random((N, dim))
        rn_r = np.random.random((N, dim * N))
    
        
        
        
        # We set all state=1 to 0 with probability nu
        aS1 = States[:,0] == 1
        aS2 = States[:,1] == 1
        States_1[aS1,0] = (rn_nu[aS1,0] > nu)
        States_1[aS2,1] = (rn_nu[aS2,1] > nu)
            

        # Nodes contact himself on the other layer
        v1 =  aS1 & (ExPr >= rn_ExPr[:,0])
        States_1[v1,1]= 1
        
        v2 =  aS2 & (ExPr >= rn_ExPr[:,1])
        States_1[v2,0]=  1

        # We simulate contacting the neighborus
        for i in range(N):
            if States[i, 0] == 1:
                pb= r[0:N,i]*InPr >  rn_r[0:N,i]
                States_1[pb[:],0]=1
            if States[i, 1] == 1:
                pb= r[N:2*N,N+i]*InPr >  rn_r[0:N,N+i]
                States_1[pb[:],1]=1
        t=t+1
    
   # Calculation of all the final values
    rhoF=np.sum(rho[(Ftime-nsum):Ftime])/nsum
    rhoF1=np.sum(rho1[(Ftime-nsum):Ftime])/nsum
    rhoF2=np.sum(rho2[(Ftime-nsum):Ftime])/nsum
    sus=(np.sum(rhox2[(Ftime-nsum):Ftime])/nsum-rhoF**2)/rhoF
    sus1=(np.sum(rhox21[(Ftime-nsum):Ftime])/nsum-rhoF1**2)/rhoF1
    sus2=(np.sum(rhox22[(Ftime-nsum):Ftime])/nsum-rhoF2**2)/rhoF2

    return [rhoF,rhoF1,rhoF2,sus,sus1,sus2]






#%%

ti=time.time()

# We define the parameters of the simulations
sims=100
rep=1
saves=np.zeros((sims,6,rep))
N=1000
eta=0.1
k1=10
k2=30

gamma=13.5

# Start the simulations
for s in range(0,rep):
    k=0
    for i in np.logspace(-2,-0.5,sims):
       saves[k,:,s]=QSsimulation2(N,gamma,eta,i,k1,k2)    
       k=k+1

tf=time.time()

print(tf-ti)

#%%

# Plot the susceptibility. I save this arrays using spyder but you can use np.save

plt.plot(np.logspace(-2,-0.5,100),saves[:,4])
plt.plot(np.logspace(-2,-0.5,100),saves[:,5])
#plt.xscale('log')

