# pvsk.py
# Por Lucas Pereira de Souza
# Programinha para Cálculo do Espectro de Potência da Matéria hoje (a=1) em função de k
# Com o modelo Lambda CDM

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def H(a, Ho, Omegam, Omegar): # Define H(a)
    return Ho*np.sqrt(1 - Omegam + Omegam/(a**3) + Omegar/(a**4))

def edosys(a, x, k, Ho, Omegam, Omegar): # Define o sistema de EDOs pra integrarmos
    deltam, deltar, thetam, thetar = x
    ddeltam = -thetam/(H(a,Ho, Omegam, Omegar)*a*a)
    ddeltar = -4.0*thetar/(3.0*H(a,Ho, Omegam, Omegar)*a*a)
    dthetam = -thetam/a - (3.0*Ho*Ho/(2.0*H(a,Ho, Omegam, Omegar)))*(Omegam*(a**-3)*deltam + Omegar*(a**-4)*deltar)
    dthetar = (k*k*deltar)/(4.0*H(a,Ho, Omegam, Omegar)*a*a) - (3.0*Ho*Ho/(2.0*H(a,Ho, Omegam, Omegar)))*(Omegam*(a**-3)*deltam + Omegar*(a**-4)*deltar)
    sistema = [ddeltam, ddeltar, dthetam, dthetar]
    return sistema
 
Omegam = 0.3 # Omega da matéria
Omegar = 8e-5 # Omega da radiação: Se quiser ver o espectro de potência da matéria
# sem influência da radiação, reescreva Omegar = 0
Ho = 70.0/3e5

k = np.linspace(0.01, 0.1, 100) #define k
logk = [np.log10(i) for i in k] #log de k
x0 = [[np.sqrt(i),np.sqrt(i),0,0] for i in k] # condições iniciais
p = [] # Vetor pra p

for i in range(len(k)): #Integra com rho_r
    sol = solve_ivp(edosys, (1e-8,1.0), x0[i], args=[k[i], Ho, Omegam, Omegar], method="LSODA", dense_output=True) # integra
    delta = (Omegam*sol.sol(1.0)[0] + Omegar*sol.sol(1.0)[1])/(Omegam + Omegar) # calcula o delta
    if Omegar == 0: # Se Omega da radiação for 0, vamos montar o gráfico p x k
        p.append(delta*delta)
    else: # Se o omega da radiação não for 0, vamos montar log p x log k
        p.append(np.log10(delta*delta))

if Omegar == 0: # Se Omega da radiação for 0, vamos montar o gráfico p x k
    plt.plot(k, p, 'r') # plota o gráfico
    plt.xlabel('k')
    plt.ylabel('P')
    plt.title('Gráfico do Espectro de Potência da Matéria hoje $(a = 1)$ em Função de $k$')
else: # Se o omega da radiação não for 0, vamos montar log p x log k
    plt.plot(logk, p, 'r') # plota o gráfico
    plt.xlabel('$\\log_{10}(k)$')
    plt.ylabel('$\\log_{10}(P)$')
    plt.title('Gráfico Log$\\times$Log do Espectro de Potência da Matéria hoje $(a = 1)$ em Função de $k$')
plt.show()
