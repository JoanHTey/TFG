# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:35:44 2024

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt


eta0_01=np.load('eta_0.01_saves.npy')
eta15=np.load('eta_15_saves.npy')
eta50=np.load('eta_50_saves.npy')
gamma5=np.load('gamma_5_saves.npy')
gamma13=np.load('gamma_13.5_saves.npy')
gamma100=np.load('gamma_100_saves.npy')

plt.figure(figsize=[7,5])
plt.subplot(233)
plt.plot(np.logspace(-2,-0.5,100),gamma100[:,4],marker='o',linestyle=' ',markersize=3)
plt.plot(np.logspace(-2,-0.5,100),gamma100[:,5],marker='o',linestyle=' ',markersize=3)
plt.xscale('log')
plt.xticks([],[])
plt.yticks([],[])
plt.ylabel(r'$\chi$',fontsize=14)

plt.subplot(232)
plt.plot(np.logspace(-2,-0.5,100),gamma13[:,4],marker='o',linestyle=' ',markersize=3)
plt.plot(np.logspace(-2,-0.5,100),gamma13[:,5],marker='o',linestyle=' ',markersize=3)
plt.xscale('log')
plt.xticks([],[])
plt.yticks([],[])


plt.subplot(231)
plt.plot(np.logspace(-2,-0.5,100),gamma5[:,4],marker='o',linestyle=' ',markersize=3)
plt.plot(np.logspace(-2,-0.5,100),gamma5[:,5],marker='o',linestyle=' ',markersize=3)
plt.xscale('log')
plt.xticks([],[])
plt.yticks([],[])
plt.ylabel(r'$\chi$',fontsize=14)

plt.subplot(236)
plt.plot(np.logspace(-2,-0.5,100),eta50[:,4],marker='o',linestyle=' ',markersize=3,label='layer 1')
plt.plot(np.logspace(-2,-0.5,100),eta50[:,5],marker='o',linestyle=' ',markersize=3,label='layer 2')
plt.xscale('log')
plt.yticks([],[])
plt.legend(loc="upper right",fontsize=12)
plt.xlabel(r'$\beta/\mu$',fontsize=14)

plt.subplot(235)
plt.plot(np.logspace(-2,-0.5,100),eta15[:,4],marker='o',linestyle=' ',markersize=3)
plt.plot(np.logspace(-2,-0.5,100),eta15[:,5],marker='o',linestyle=' ',markersize=3)
plt.xscale('log')
plt.yticks([],[])
plt.xlabel(r'$\beta/\mu$',fontsize=14)

plt.subplot(234)
plt.plot(np.logspace(-2,-0.5,100),eta0_01[:,4],marker='o',linestyle=' ',markersize=3)
plt.plot(np.logspace(-2,-0.5,100),eta0_01[:,5],marker='o',linestyle=' ',markersize=3)
plt.xscale('log')
plt.yticks([],[])
plt.xlabel(r'$\beta/\mu$',fontsize=14)
plt.ylabel(r'$\chi$',fontsize=14)



plt.subplots_adjust(wspace=0, hspace=0)
#%%
import networkx as nx
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
#v1={}

for gamma in np.logspace(0,2,10):
  i1l[gamma]=[]
  i2l[gamma]=[]
  #v1[gamma]=[]
  for p in np.logspace(-2,2,100):
    r = build_supra_contact_matrix(A_1, A_2, p, gamma)
    w,v=leading_eigenvector(r)
    i1,i2=IPR(w)
    #v1[gamma].append(v)
    i1l[gamma].append(i1)
    i2l[gamma].append(i2)

#%%
colors=['red','blue','green','orange','pink','brown','gray','yellow','purple','lima']
i=0
ER1=np.load('ErdorenyiIPR1.npy')
ER2=np.load('ErdorenyiIPR2.npy')


plt.figure(figsize=[7,7])
plt.subplot(221)
for gamma in np.logspace(0,2,10):
  if gamma!=1:

    plt.plot(np.logspace(-2,2,100),i1l[gamma],color=colors[i],linestyle='--' )
    plt.plot(np.logspace(-2,2,100),i2l[gamma],label=r'$\gamma$'+'='+str(int(gamma*100)/100),color=colors[i])

    i=i+1


plt.yscale('log')
plt.xscale('log')
plt.xticks([],[]) 
plt.ylabel("IPR",fontsize=14)


i=0
plt.subplot(222)
for gamma in np.logspace(0,2,10):
  if gamma!=1:
    if i<10:
      plt.plot(np.logspace(-2,2,100)/((10*(1-(1-1/10)**(np.logspace(0,2,10)[i+1]))-8*(1-(1-1/8)**(np.logspace(0,2,10)[i+1])))),i1l[gamma],color=colors[i],linestyle='--' )
    
      plt.plot(np.logspace(-2,2,100)/((10*(1-(1-1/10)**(np.logspace(0,2,10)[i+1]))-8*(1-(1-1/8)**(np.logspace(0,2,10)[i+1])))),i2l[gamma],label=r'$\gamma$'+'='+str(int(gamma*100)/100),color=colors[i])
    if 10<i:
        plt.plot(np.logspace(-2,2,100)/((10*(1-(1-1/10)**(np.logspace(0,2,10)[i+1]))-8*(1-(1-1/8)**(np.logspace(0,2,10)[i+1])))),i1l[gamma],color=colors[i],linestyle='--' )
      
        plt.plot(np.logspace(-2,2,100)/((10*(1-(1-1/10)**(np.logspace(0,2,10)[i+1]))-8*(1-(1-1/8)**(np.logspace(0,2,10)[i+1])))),i2l[gamma],color=colors[i])
    i=i+1
plt.yscale('log')
plt.xscale('log')
plt.legend(loc="lower right",fontsize=8)
plt.xticks([],[])
plt.yticks([],[])

i=0
plt.subplot(223)
for gamma in np.logspace(0,2,10):
   if gamma!=1:
     plt.plot(np.logspace(-2,2,100),ER1[i+1],color=colors[i],linestyle='--' )
     plt.plot(np.logspace(-2,2,100),ER2[i+1],label=r'$\gamma$'+'='+str(int(gamma*100)/100),color=colors[i])  
     i=i+1

plt.yscale('log')
plt.xscale('log')  
plt.xlabel("p",fontsize=14)
plt.ylabel("IPR",fontsize=14)

i=0
plt.subplot(224)
for gamma in np.logspace(0,2,10):
   if gamma!=1:
     if i<5:  
       
       plt.plot(np.logspace(-2,2,100)/((100*(1-(1-1/100)**(np.logspace(0,2,10)[i+1]))-80*(1-(1-1/80)**(np.logspace(0,2,10)[i+1])))),ER1[i+1],color=colors[i],linestyle='--' )
       plt.plot(np.logspace(-2,2,100)/((100*(1-(1-1/100)**(np.logspace(0,2,10)[i+1]))-80*(1-(1-1/80)**(np.logspace(0,2,10)[i+1])))),ER2[i+1],color=colors[i])  
     if 4<i:
        plt.plot(np.logspace(-2,2,100)/((100*(1-(1-1/100)**(np.logspace(0,2,10)[i+1]))-80*(1-(1-1/80)**(np.logspace(0,2,10)[i+1])))),ER1[i+1],color=colors[i],linestyle='--' )
        plt.plot(np.logspace(-2,2,100)/((100*(1-(1-1/100)**(np.logspace(0,2,10)[i+1]))-80*(1-(1-1/80)**(np.logspace(0,2,10)[i+1])))),ER2[i+1],label=r'$\gamma$'+'='+str(int(gamma*100)/100),color=colors[i])
     i=i+1

    
     
plt.legend(loc="lower right")    

plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("p/p*",fontsize=14)
plt.yticks([],[]) 
plt.subplots_adjust(wspace=0, hspace=0)




