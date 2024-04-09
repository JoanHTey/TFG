# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:15:48 2024

@author: Usuario
"""
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

#%%

G1=nx.random_regular_graph(10,100)
G2=nx.random_regular_graph(8,100)
while nx.is_connected(G1)!=True:
  G1=nx.random_regular_graph(10,100)
while nx.is_connected(G2)!=True:
  G2=nx.random_regular_graph(8,100)

A_1=nx.to_numpy_array(G1)
A_2=nx.to_numpy_array(G2)


i1l={}
i2l={}

for gamma in np.logspace(0,2,20):
  i1l[gamma]=[]
  i2l[gamma]=[]
  
  for p in np.logspace(-1,2,100):
    G1=nx.random_regular_graph(10,100)
    G2=nx.random_regular_graph(8,100)
    while nx.is_connected(G1)!=True:
      G1=nx.random_regular_graph(10,100)
    while nx.is_connected(G2)!=True:
      G2=nx.random_regular_graph(8,100)

    A_1=nx.to_numpy_array(G1)
    A_2=nx.to_numpy_array(G2)

    r = build_supra_contact_matrix(A_1, A_2, p, gamma)
    w,v=leading_eigenvector(r)
    i1,i2=IPR(w)
    i1l[gamma].append(i1)
    i2l[gamma].append(i2)
    
    
    
for gamma in np.logspace(0,2,20):
    if gamma!=1:
        plt.plot(np.logspace(-5,2,100),i2l[gamma],label=gamma)

plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.xlabel("p'")
plt.ylabel("IPR(2)")

plt.show()
#%%
alpha=[]
c1=[]

for gamma in np.logspace(0,2,20):
    c,m=np.polyfit(np.log(np.logspace(-1,2,1000))[:100],np.log(i2l[gamma])[:100],1)
    alpha.append(c)
    c1.append(m)

#%%
i=1
x=np.linspace(0.1,1,100)
for gamma in np.logspace(0,2,20):
    if gamma!=1:
        plt.plot(np.logspace(-1,2,1000),i2l[gamma],label=gamma)
        plt.plot(x[:i*5],(x[:i*5])**(alpha[i])*np.exp(c1[i]))
        i=i+1
        
        plt.yscale('log') 
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel("p'")
        plt.ylabel("IPR(2)")
        plt.show()
        
#%%
plt.plot(np.logspace(0,2,20)[1:],alpha[1:])
plt.xlabel("gamma")
plt.ylabel("alpha")
plt.yscale('log')
plt.xscale('log')
plt.show()

#%%
i1sum={}
i2sum={}
for gamma in np.logspace(0,2,20):
    i1sum[gamma]=np.zeros(100)
    i2sum[gamma]=np.zeros(100)
  
for i in range(0,10):

    i1_2l={}
    i2_2l={}
    for gamma in np.logspace(0,2,20):
        i1_2l[gamma]=[]
        i2_2l[gamma]=[]
        
    
        for p in np.logspace(-5,2,100):
            G1_2=nx.erdos_renyi_graph(1000,0.1)
            G2_2=nx.erdos_renyi_graph(1000,0.08)
            while nx.is_connected(G1_2)!=True:
                G1_2=nx.erdos_renyi_graph(1000,0.1)
            while nx.is_connected(G2_2)!=True:
                G2_2=nx.erdos_renyi_graph(1000,0.08)

            A_1_2=nx.to_numpy_array(G1_2)
            A_2_2=nx.to_numpy_array(G2_2)
            r = build_supra_contact_matrix(A_1_2, A_2_2, p, gamma)
            w,v=leading_eigenvector(r)
            i1,i2=IPR(w)
            i1_2l[gamma].append(i1)
            i2_2l[gamma].append(i2)
        i1sum[gamma][:]=i1sum[gamma][:]+np.array(i1_2l[gamma])[:]/10
        i2sum[gamma][:]=i2sum[gamma][:]+np.array(i2_2l[gamma])[:]/10
#%%
print(np.array(i2sum[gamma]))



#%%
for gamma in np.logspace(0,2,20):
    if gamma!=1:
        plt.plot(np.logspace(-5,2,100),i2_2l[gamma],label=gamma)

plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.xlabel("p'")
plt.ylabel("IPR(2)")

plt.show()
#%%

alpha=[]
c1=[]
c2=[]


for gamma in np.logspace(0,2,20):
    c,m=np.polyfit(np.log(np.logspace(-5,2,100))[:10],np.log(i2_2l[gamma])[:10],1)
    cf=i2_2l[gamma][99]

    alpha.append(c)
    c1.append(m)
    c2.append(cf)
print(c1,c2)

#Obtaining p*
Pc=[]

for i in range(1,20):
  Pc.append((c2[i]/np.exp(c1[i]))**(1/alpha[i]))

print(Pc)
#%%

i=0
for gamma in np.logspace(0,2,20):
  if gamma!=1:
    plt.plot(np.logspace(-5,2,100)/Pc[i],i1_2l[gamma],label=gamma)
    plt.plot(np.logspace(-5,2,100)/Pc[i],i2_2l[gamma],label=gamma)
    #plt.plot(np.logspace(-5,2,100),1/100*(np.logspace(-5,2,100)/(10*(1-(1-1/10)**(gamma))-8*(1-(1-1/8)**(gamma))))**4,label=gamma)
    i=i+1
plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.xlabel("p'")
plt.ylabel("IPR(2)")

#%%
N=10000
dim=2
States=np.zeros((N,dim))
nu=0.2
InPr=0.1
ExPr=0.001
eta=ExPr/InPr

p=0.0001
gamma=3
M=1000
ProbGen=0.1
Ftime=5000
rho=np.ones(Ftime)
rho1=np.ones(Ftime)
rho2=np.ones(Ftime)
SaveStates=np.zeros((M,N,dim))
Prep=10**(-2)



for j in range(0,dim):
    for i in range(0,N):
        States[i,j]=int(random.random()<=ProbGen)

G1_2=nx.erdos_renyi_graph(N,0.01)
G2_2=nx.erdos_renyi_graph(N,0.001)
while nx.is_connected(G1_2)!=True:
      G1_2=nx.erdos_renyi_graph(N,0.003)
while nx.is_connected(G2_2)!=True:
      G2_2=nx.erdos_renyi_graph(N,0.001)

A_1_2=nx.to_numpy_array(G1_2)
A_2_2=nx.to_numpy_array(G2_2)
r = build_supra_contact_matrix(A_1_2, A_2_2, eta, gamma)

t=0
while t<Ftime:
    rho[t]=np.sum(States)/(N*2)
    rho1[t]=np.sum(States[:,0])/N
    rho2[t]=np.sum(States[:,1])/N
    
    
    if t<M:
        SaveStates[t,:,:]=States[:,:]
    if t>=M and random.random()<=Prep:
        k=random.randint(0,M)
        SaveStates[k,:,:]
    if rho[t]==0:
        k=random.randint(0,M)
        States[:,:]=SaveStates[k,:,:]

   
    
    for g in range(0,dim):
        for i in range(0,N):
            if States[i,g]==1:
                States[i,g]=int(random.random()>nu)
                if States[i,(g-1)**2]!=1:
                    States[i,(g-1)**2]=int(ExPr>=random.random())
                for j in range(0,N):
                    if r[j,i]>random.random():
                        States[j]=int(InPr>=random.random())

        
    t=t+1

#%%

plt.plot(rho[0:1000])
