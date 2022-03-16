# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:44:07 2022

@author: Richard
"""

import numpy as np
from scipy.special import comb
import cvxpy as cp
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
import funciones as f
import scipy.optimize as optimize
from scipy.optimize import differential_evolution
from scipy.stats import unitary_group
import itertools

N = 2 # Número de qubits

# Matrices de Pauli:

I = np.array([[1,0],[0,1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1.j],[1.j,0]])
Z=np.array([[1,0],[0,-1]])
sigmas=[I,X,Y,Z]

n_out = 2  # nro de outputs de las variables
n_var = 4  # nro de variables aleatorias
# En esta lista cada entrada es la cantidad de outputs de cada variable,
# hay tantas entradas como variables aleatorias.
dimensiones_va = [2, 2, 2, 2]
# Esto es una lista de listas con los outputs posibles de cada variable.
outputs = np.array([[1, -1], [1, -1], [1, -1], [1,-1]])
# Aquí guardo los valores medios de las variables.
mean_val = np.array([0, 0, 0, 0])
# Genero un arreglo de ceros (las entradas de este arreglo serán las probabilidades
# a calcular) cuyas dimensiones son el producto de las entradas de la
# lista dimensiones_va.
# Ej: probab [1][2][1][1] será P(x1_output1, x2_output2, x3_output1, x4_output4).

probab = np.zeros(dimensiones_va)

# Aquí calculo el tamaño de la matriz de coeficientes, es decir, el número de ecuaciones.
# Por ahora asumo que tengo todos los valores medios, de las variables y de los productos.

def n_equations(n_var):
    sum = 0
    for i in range(2, n_var):
        sum = sum + comb(n_var, i, exact = True)  # el coso ese calcula el coeficiente binomial.
    a = n_var + sum + 1 #No cuento el caso del producto de todas las VA.
    return a

# Acá le metemos a mano el número de filas y columnas (por ahora).

n_filas = 9

dim = [n_filas, 16]

# Esta función arma una lista de listas ordenada con los índices de las probabilidades
# a calcular, es decir, [0,0,0][0,0,1][0,1,0][0,1,1].. para dimensiones_va=[2,2,2]

def indices_probab(dimensiones_va):
    lista_indices = [[]]
    for dim_actual in dimensiones_va:
        indices_sig = []
        for indices in lista_indices:
            for valor in range(dim_actual):
                indices_sig.append(indices + [valor])
        lista_indices = indices_sig
    return lista_indices

#Me arma la lista con todas las combinaciones posibles de productos y con las variables aleatorias
#solitas, en orden primero por longitud ascendente y luego lexicográfico.
def subconjuntos(cant_va):
  lista_subconjuntos = [[]]
  for va_actual in range(cant_va):
    lista_con_va = [subc + [va_actual] for subc in lista_subconjuntos]
    lista_subconjuntos += lista_con_va
  lista_subconjuntos.sort()  # sorts normally by alphabetical order
  lista_subconjuntos.sort(key=len)  # sorts by ascending length
  lista_subconjuntos = [[],[0],[1],[2],[3],[0, 2],[0, 3],[1, 2],[1, 3]]    
  return lista_subconjuntos


# dim es la dimensión de la matriz de coeficientes del sistema a resolver.
# A cada fila le corresponde el producto de un subconjunto de variables aleatorias
# (el orden está en la lista de listas que genero en subconjuntos´),
# entonces para cada fila se recorre la lista de los índices y se toma el producto
# de las variables aleatorias correspondientes en ese índice.

def matriz_coeficientes(dim, n_var):
    matriz = np.ones(dim)
    lista_indices = indices_probab(dimensiones_va)
    lista_productos = subconjuntos(n_var)
    for fila in range(dim[0]):
        for ind_lista, indices in enumerate(lista_indices):
            for variable_aleatoria in lista_productos[fila]:
                matriz[fila, ind_lista] *= outputs[variable_aleatoria][indices[variable_aleatoria]]
    return matriz


# a = matriz de coef, b = vector terminos independientes

a = matriz_coeficientes(dim, n_var)

# d1 = np.array([np.cos(t1),0,np.sin(t1)])

# d2 = np.array([np.cos(w1),0,np.sin(w1)])

# d_1 = np.array([np.cos(a1),0,np.sin(a1)])

# d_2 = np.array([np.cos(b1),0,np.sin(b1)])

# Armamos distintos tipos de estados:

IdNormN = (1/2**N)*(np.eye(2**N))

ListasN = f.Lists(sigmas,N)

# Armo un vector puro de un qubit. Sería el ket I0>, expresado como vector columna. También armo el I1> ('one').

Zero = np.array([[1],[0]])

One = np.array([[0],[1]])

# Ahora armo una lista que lo repite al vector I0> N veces:

V = list(itertools.repeat(Zero,N))

# Ahora hago el producto tensorial de la lista de ceros: I0> tensor I0> tensor I0> ..... tensor I0>.
#  Sería como el estado de N qubits en el que todos los qubits están en I0>. En el Hilbert, lo podés pensar
# como una flecha que apunta hacia el "norte" (en un sentido figurado). 

Vect = f.KroneckerMultiplyList(V)

V0 = np.kron(Zero,One)

V1 = np.kron(One,Zero)

Vect_cat = (1/np.sqrt(2))*(V0-V1)

Vect_cat_Dagger = Vect_cat.conj().T    

RHO_cat = np.outer(Vect_cat,Vect_cat_Dagger) 

Set_random = [RHO_cat]

Nro_Estados = 1000

for i in range(Nro_Estados):
    U = unitary_group.rvs(2**N)
    V = np.dot(U,Vect)
    W = V.conj().T
    RHO = np.outer(V,W) 
    Set_random.append(RHO)
    
Num_A = 100    

Set_A = []
    
for i in range(1,Num_A+1):        
    p = (0.5/Num_A)*i
    Psi_p = np.sqrt(p)*V0-np.sqrt(1-p)*V1 
    Psi_p_Dagger = Psi_p.conj().T
    RHO_p = np.outer(Psi_p,Psi_p_Dagger)    
    Set_A.append(RHO_p)
    
Set_B = []

for i in range(100):
    Local_Unitaries = []
    for j in range(N):
      U = unitary_group.rvs(2)
      Local_Unitaries.append(U)
    NU = f.KroneckerMultiplyList(Local_Unitaries)
    Vect = np.dot(NU,V0)
    Vect_Dagger = Vect.conj().T
    RHO_Prod_Rand = np.outer(Vect,Vect_Dagger) 
    Set_B.append(RHO_Prod_Rand)
    
Set_C = []

for i in range(100):
    Local_Unitaries_1 = []
    for j in range(N):
      U_1 = unitary_group.rvs(2)
      Local_Unitaries_1.append(U_1)
    NU_1 = f.KroneckerMultiplyList(Local_Unitaries_1)
    V = (1/np.sqrt(2))*(V0 + V1)
    Vect1 = np.dot(NU_1,V)
    Vect1_Dagger = Vect1.conj().T
    RHO_Ent = np.outer(Vect1,Vect1_Dagger) 
    Set_C.append(RHO_Ent)    


# Ahora hago el producto tensorial de la lista de ceros: I0> tensor I0> tensor I0> ..... tensor I0>.
#  Sería como el estado de N qubits en el que todos los qubits están en I0>. En el Hilbert, lo podés pensar
# como una flecha que apunta hacia el "norte" (en un sentido figurado). 

# n1 y n2 se van a usar abajo para calcular la matriz reducida (para medir entropía de
# entrelazamiento).

n1,n2 = 2,2**(N-1)

Ent_Values = []

Violations = []

Contextuality = []

for RHO in Set_C:
    # Esta función define, para cada RHO, la desigualdad CHSH
    def CHSH(params):
     theta1,phi1,theta2,phi2 = params # <-- for readability you may wish to assign names to the component variables
     O1 = np.cos(theta1)*X + np.sin(theta1)*Z
     V1 = np.cos(phi1)*X + np.sin(phi1)*Z
     O2 = np.cos(theta2)*X + np.sin(theta2)*Z
     V2 = np.cos(phi2)*X + np.sin(phi2)*Z
     O1_O2 = f.KroneckerMultiplyList([O1,O2])
     O1_V2 = f.KroneckerMultiplyList([O1,V2])
     V1_O2 = f.KroneckerMultiplyList([V1,O2])
     V1_V2 = f.KroneckerMultiplyList([V1,V2])
     Obs = O1_O2 - O1_V2 + V1_O2 + V1_V2
     CHSH = (-1)*np.sqrt((np.trace(RHO@Obs).astype(float))**2)
     return CHSH        
    # Ahora, para ese RHO, calculamos los ángulos para los cuales la violación es máxima.
    initial_guess = [0.5,45,22.5,67.5]
    bnds = [(0,2*np.pi),(0, 2*np.pi),(0, 2*np.pi),(0, 2*np.pi)]
    result = optimize.minimize(CHSH,initial_guess, bounds=bnds)
    # Si el resultado anda bien, usamos los ángulos obtenidos para calcular
    # la contextualidad de ese RHO respecto de los observables con esos ángulos.
    if result.success:
      Angles = result.x
      Violations.append(-CHSH(Angles))
      O11 = np.cos(Angles[0])*X + np.sin(Angles[0])*Z
      V11 = np.cos(Angles[1])*X + np.sin(Angles[1])*Z
      O22 = np.cos(Angles[2])*X + np.sin(Angles[2])*Z
      V22 = np.cos(Angles[3])*X + np.sin(Angles[3])*Z
      O11_O22 = f.KroneckerMultiplyList([O11,O22])
      O11_V22 = f.KroneckerMultiplyList([O11,V22])
      V11_O22 = f.KroneckerMultiplyList([V11,O22])
      V11_V22 = f.KroneckerMultiplyList([V11,V22])
      b = [np.trace(RHO),np.trace(RHO@(np.kron(O11,I))).astype(float),np.trace(RHO@(np.kron(V11,I))).astype(float),np.trace(RHO@(np.kron(I,O22))).astype(float),np.trace(RHO@(np.kron(I,V22))).astype(float),np.trace(RHO@O11_O22).astype(float),np.trace(RHO@O11_V22).astype(float),np.trace(RHO@V11_O22).astype(float),np.trace(RHO@V11_V22).astype(float)]
      y = cp.Variable(a.shape[1])
      constr1 = [a@y == b]
      contextuality = cp.norm(y, 1)
      obj = cp.Minimize(contextuality)
      prob = cp.Problem(obj, constr1)
      prob.solve(verbose=True)
      c = y.value 
      cont = (np.linalg.norm(y.value, ord=1))
      Contextuality.append(cont-1)
      # Ahora calculamos la traza parcial del estado, y la entropía del subsistema.
      # Eso nos da la entropía de entrelazamiento (para llevar registro de cuan entrelazado estaba).
      RHO_tensor_A = RHO.reshape([n1, n2, n1, n2])
      RHO_A = np.trace(RHO_tensor_A, axis1=1, axis2=3)
      VNE_A = f.von_neumann_entropyEig(RHO_A)
      Ent_Values.append(VNE_A)
    else:
        raise ValueError(result.message)
        
        
    
      # O11 = np.cos(0)*X + np.sin(0)*Z
      # V11 = np.cos(45)*X + np.sin(45)*Z
      # O22 = np.cos(22.5)*X + np.sin(22.5)*Z
      # V22 = np.cos(67.5)*X + np.sin(67.5)*Z
      # O11_O22 = f.KroneckerMultiplyList([O11,O22])
      # O11_V22 = f.KroneckerMultiplyList([O11,V22])
      # V11_O22 = f.KroneckerMultiplyList([V11,O22])
      # V11_V22 = f.KroneckerMultiplyList([V11,V22])
      # Obs = O11_O22 - O11_V22 + V11_O22 + V11_V22
      # CHSH = np.sqrt((np.trace(RHO@Obs).astype(float))**2)
      # b = [np.trace(RHO),np.trace(RHO@(np.kron(O11,I))).astype(float),np.trace(RHO@(np.kron(V11,I))).astype(float),np.trace(RHO@(np.kron(I,O22))).astype(float),np.trace(RHO@(np.kron(I,V22))).astype(float),np.trace(RHO@O11_O22).astype(float),np.trace(RHO@O11_V22).astype(float),np.trace(RHO@V11_O22).astype(float),np.trace(RHO@V11_V22).astype(float)]
      # y = cp.Variable(a.shape[1])
      # constr1 = [a@y == b]
      # contextuality = cp.norm(y, 1)
      # obj = -cp.Minimize(contextuality)
      # prob = cp.Problem(obj, constr1)
      # prob.solve(verbose=True)
      # c = y.value 
      # cont = (np.linalg.norm(y.value, ord=1))
      # Contextuality.append(cont-1)
    
    
print('Contextualidad')    
print(Contextuality)

print('Violations')    
print(Violations)

print('Entanglement')    
print(Ent_Values)

# Esto para plotear, si se usan esos estados, los estados tipo gato que dependen de p.

P = []

for i in range(1,Num_A+1):        
    k = (0.5/Num_A)*i
    P.append(k)
 

hist,bin_edges = np.histogram(Contextuality, bins=1000)

np.save('Figuras/Contextuality_'+str(int(N)), Contextuality)

np.save('Figuras/Violations_'+str(int(N)), Violations)

np.save('Figuras/Entanglement_'+str(int(N)), Ent_Values)

np.save('Figuras/Hist_'+str(int(N)), hist)

import matplotlib

import matplotlib.pyplot as plt

   
# plt.plot(P,Contextuality)

# plt.plot(P,Violations)

plt.show()

fig = plt.figure(figsize=[10,8])

plt.bar(bin_edges[:-1], hist, width = 0.05, color='#0504aa',alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Histograma de Contextualidad',fontsize=15)
plt.show()
fig.savefig('Figuras/Histogram_'+str(int(N))+'.png')



