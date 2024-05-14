# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:15:48 2024

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
numgamma=10
Nodes=100
Ngen=100

i1sum=np.zeros((10,100))
i2sum=np.zeros((10,100))

 
for k in range(0,Ngen):

    i1_2l=np.zeros(100)
    i2_2l=np.zeros(100)
   
    for i in range (0,numgamma):
        gamma=np.logspace(0,2,10)[i]
        
        for j in range (0,Nodes):
            p=np.logspace(-1,2,Nodes)[j]
            
            G1_2=nx.erdos_renyi_graph(Nodes,0.1)
            G2_2=nx.erdos_renyi_graph(Nodes,0.08)
            while nx.is_connected(G1_2)!=True:
                G1_2=nx.erdos_renyi_graph(Nodes,0.1)
            while nx.is_connected(G2_2)!=True:
                G2_2=nx.erdos_renyi_graph(Nodes,0.08)

            A_1_2=nx.to_numpy_array(G1_2)
            A_2_2=nx.to_numpy_array(G2_2)
            r = build_supra_contact_matrix(A_1_2, A_2_2, p, gamma)
            w,v=leading_eigenvector(r)
            i1,i2=IPR(w)
            i1_2l[j]=i1
            i2_2l[j]=i2
        i1sum[i][:]=i1sum[i][:]+i1_2l[:]/Ngen
        i2sum[i][:]=i2sum[i][:]+i2_2l[:]/Ngen


#%%
print(np.array(i2sum[gamma]))



#%%
i=0
for gamma in np.logspace(0,2,10):
    if gamma!=1:
        plt.plot(np.logspace(-1,2,100),i2sum[i][:],label=gamma)
    i=1+i
    

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
nu=0
InPr=1
ExPr=0
eta=ExPr/InPr

gamma=3
M=1000
SaveStates=np.zeros((M,N,dim))
Prep=10**(-2)



#for j in range(0,dim):
   # for i in range(0,N):
      #  States[i,j]=int(random.random()<=ProbGen)

G1_2=nx.erdos_renyi_graph(N,0.003)
G2_2=nx.erdos_renyi_graph(N,0.001)
while nx.is_connected(G1_2)!=True:
      G1_2=nx.erdos_renyi_graph(N,0.003)
while nx.is_connected(G2_2)!=True:
      G2_2=nx.erdos_renyi_graph(N,0.001)

A_1_2=nx.to_numpy_array(G1_2)
A_2_2=nx.to_numpy_array(G2_2)
r = build_supra_contact_matrix(A_1_2, A_2_2, eta, 100000)

States=np.zeros((N,dim))
SaveStates=np.zeros((M,N,dim))
States_1=np.zeros((N,dim))
States[0,1]=1
States[0,0]=1
t=0
States_1[:,:]=States[:,:]
Ftime=100
rho=np.ones(Ftime)
rho1=np.ones(Ftime)
rho2=np.ones(Ftime)
rhox2=np.ones(Ftime)
rhox21=np.ones(Ftime)
rhox22=np.ones(Ftime)
x=0

while t<Ftime:
    States[:,:]=States_1[:,:]
    print(t)
    rho[t]=np.sum(States)/(N*2)
    rho1[t]=np.sum(States[:,0])/N
    rho2[t]=np.sum(States[:,1])/N
    rhox2[t]=(np.sum(States)/(N*2))**2
    rhox21[t]=(np.sum(States[:,0])/N)**2
    rhox22[t]=(np.sum(States[:,1])/N)**2

    
    if t<M:
        SaveStates[t,:,:]=States[:,:]
    if t>=M and random.random()<=Prep:
        k=random.randint(0,M)
        SaveStates[k,:,:]=States[:,:]
    if rho[t]==0:
        while x==0:
            k=random.randint(0,M)
            if np.sum(SaveStates[k,:,:]) != 0 :
                States[:,:]=SaveStates[k,:,:]
                x=1
        x=0
    
    for g in range(0,dim):
        print(np.sum(States_1[:N,0]))
        for i in range(0,N):
            if States[i,g]==1:
                States_1[i,g]=int(random.random()>nu)
                if States[i,(g-1)**2] != 1 :
                    if ExPr>random.random():
                        States_1[i,(g-1)**2]=1
                for j in range(0,N):
                    if r[g*N+j,g*N+i]>random.random():
                        if InPr>=random.random():
                            States_1[j,g]=1

    t=t+1
#%%
x=np.ones((N,N))

x[:N,:N]=r[:N,:N]
print(nx.is_connected(G1_2))
#%%
print(np.sum(States[:,0]))


#%%

def QSsimulation(N,gamma,eta,InPr,k1,k2):
    dim=2
    States=np.zeros((N,dim))
    nu=1
    ExPr=eta*InPr
    M=1000
    SaveStates=np.zeros((M,N,dim))
    Prep=10**(-2)


    G1_2=nx.erdos_renyi_graph(N,k1/N)
    G2_2=nx.erdos_renyi_graph(N,k2/N)
    while nx.is_connected(G1_2)!=True:
          G1_2=nx.erdos_renyi_graph(N,k1/N)
    while nx.is_connected(G2_2)!=True:
          G2_2=nx.erdos_renyi_graph(N,k2/N)

    A_1_2=nx.to_numpy_array(G1_2)
    A_2_2=nx.to_numpy_array(G2_2)
    r = build_supra_contact_matrix(A_1_2, A_2_2, eta, gamma)


    States=np.zeros((N,dim))
    SaveStates=np.zeros((M,N,dim))
    States_1=np.zeros((N,dim))
    States[0,1]=1
    States[0,0]=1
    t=0
    States_1[:,:]=States[:,:]
    Ftime=400
    rho=np.ones(Ftime)
    rho1=np.ones(Ftime)
    rho2=np.ones(Ftime)
    rhox2=np.ones(Ftime)
    rhox21=np.ones(Ftime)
    rhox22=np.ones(Ftime)   
    x=0

    while t<Ftime:
        States[:,:]=States_1[:,:]
        print(t)
        rho[t]=np.sum(States)/(N*2)
        rho1[t]=np.sum(States[:,0])/N
        rho2[t]=np.sum(States[:,1])/N
        rhox2[t]=(np.sum(States)/(N*2))**2
        rhox21[t]=(np.sum(States[:,0])/N)**2
        rhox22[t]=(np.sum(States[:,1])/N)**2

        
        if t<M:
            SaveStates[t,:,:]=States[:,:]
        if t>=M and random.random()<=Prep:
            k=random.randint(0,M)
            SaveStates[k,:,:]=States[:,:]
        if rho[t]==0:
            while x==0:
                k=random.randint(0,M-1)
                if np.sum(SaveStates[k,:,:]) != 0 :
                    States[:,:]=SaveStates[k,:,:]
                    x=1
            x=0
        
        for g in range(0,dim):
            for i in range(0,N):
                if States[i,g]==1:
                    States_1[i,g]=int(random.random()>nu)
                    if States[i,(g-1)**2] != 1 :
                        if ExPr>random.random():
                            States_1[i,(g-1)**2]=1
                    for j in range(0,N):
                        if r[g*N+j,g*N+i]>random.random(): 
                            if InPr>=random.random():
                                States_1[j,g]=1
            
        t=t+1
    rhoF=np.sum(rho[300:Ftime])/100
    rhoF1=np.sum(rho1[300:Ftime])/100
    rhoF2=np.sum(rho2[300:Ftime])/100
    sus=(np.sum(rhox2[300:Ftime])/100-rhoF)/rhoF
    sus1=(np.sum(rhox21[300:Ftime])/100-rhoF1**2)/rhoF1
    sus2=(np.sum(rhox22[300:Ftime])/100-rhoF2**2)/rhoF2
    plt.plot(rho1)
    plt.plot(rho2)
    plt.plot(rho)
    return [rhoF,rhoF1,rhoF2,sus,sus1,sus2]
#%%
ti=time.time()
sims=100
rep=10
saves=np.zeros((sims,6,rep))
N=1000
eta=0.01
k1=30
k2=10

gamma=10000


for s in range(0,rep):
    k=0
    for i in np.logspace(-2,-0.7,sims):
        saves[k,:,s]=QSsimulation(N,gamma,eta,i,k1,k2)
        k=k+1
tf=time.time()

print(tf-ti)
#%%

savestrue=np.zeros((sims,6))
savestrue[:,:]=(saves[:,:,0]+saves[:,:,1]+saves[:,:,2]+saves[:,:,3]+saves[:,:,4]+saves[:,:,5]+saves[:,:,6]+saves[:,:,7]+saves[:,:,8]+saves[:,:,9])/10
#%%
plt.plot(np.logspace(-2,-0.7,100),savestrue[:,1])
plt.plot(np.logspace(-2,-0.7,100),savestrue[:,2])
plt.xscale('log')

