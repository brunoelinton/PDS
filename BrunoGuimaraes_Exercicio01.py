# ATIVIDADE COMPLEMENTAR PYTHON LAB009
# Disciplina: Sinais e Sistemas
# Nome: Exercício 01
# Autor: Bruno Elinton Guimarães de Araújo
# Data: 09/12/2020
# Descrição:
'''
ESTE PROGRAMA SIMULA A APLICAÇÃO DE RETENTOR DE PRIMEIRA ORDEM (INTERPOLAÇÃO LINEAR) + FILTRAGEM RC = 0.35 NO SINAL x(t)
PARA SEQUÊNCIAS DE AMOSTRAS COM OS SEGUINTES PERÍODOS DE AMOSTRAGEM:
    > Ts = 0.001 s
    > Ts = 0.0025 s
O SINAL x(t) É DADO POR:
    x(t) = 100cos(2π∙100t)
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.fft import fft, ifft, fftfreq, fftshift
import scipy as sc
from scipy import signal
import scipy.interpolate

'''
A FUNÇÃO filtragemRC RECEBE COMO PARÂMETROS:
    x       -> O SINAL x(t)
    t       -> O VETOR DE TEMPO CONTÍNUO PARA REPRESENTAÇÃO DO SINAL
    T       -> O PERÍODO DE x(t)
    Tam     -> A TAXA DE AMOSTRAGEM
    RC      -> VALORES PARA O FILTRO RC 
'''
def filtragem_RC(x, t, T, Tam, RC = [0.1, 0.35, 1.2]):
    
    X = fft(x)/len(x)                   # FFT DO SINAL x
    W = fftfreq(len(t), d=(1/T)*Tam)    # VETOR DE FREQUÊNCIA

    for RCk in RC:
        H = (1/(1+((1j * W)*(RCk))))    # MODELAGEM DO FILTRO PASSA-BAIXAS
        Y = H * X                       # APLICANDO O FILTRO

        # RETORNANDO O SINAL PARA O DOMÍNIO DO TEMPO
        yt = ifft(Y) * len(x)
        yr = np.real(yt)                # IGNORANDO OS ERROS DE ARREDONDAMENTO

        # PLOTANDO O SINAL FILTRADO
        fig, ax1 = plt.subplots()
        ax1.plot(t, x, 'k-', lw=2, label="xr(t)")
        ax1.plot(t, yr, 'r--', lw=2, label="xr_filtrado(t)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("tempo [s]")
        ax1.grid(True)
        ax1.legend()
        ax1.set_title("xr(t) e xr_filtrado(t) para RC = " + str(RCk))
            
    return yr

'''
A FUNÇÃO analisadorEspectro RECEBE COMO PARÂMETROS:
    x       -> O SINAL x(t)
    t       -> O VETOR DE TEMPO CONTÍNUO PARA REPRESENTAÇÃO DO SINAL
    N       -> O TAMANHO DA FFT
    limitF  -> O LIMITE SUPERIOR DA FREQUÊNCIA
    titleT  -> O TÍTULO DO GRÁFICO x(t)
    titleF  -> O TÍTULO DO GRÁFICO |X(jω)|  
'''   
def analisadorEspectro(x, t, N, limitF, titleT, titleF):
    # CALCULANDO A FT DO SINAL x(t)
    Tam = t[1]-t[0]
    X = fft(x, N)*Tam
    w = fftfreq(len(X), d=Tam)
    # REPOSICIONANDO OS ÍNDICES DE FREQUÊNCIA E O ESPECTRO DE X
    Xd = fftshift(X)
    Wd = fftshift(w)
    # CALCULANDO O MÓDULO DO ESPECTRO
    ModX = np.abs(Xd)

    # PLOTANDO OS GRÁFICOS
    fig, ax = plt.subplots(2,1)
    ''' GRÁFICO x(t) '''
    ax[0].plot(t, x, 'k', lw=2, label="x(t)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("tempo [s]")
    ax[0].grid(True)
    ax[0].set_title(titleT)
    ''' GrÁFICO |X(jω)| '''
    ax[1].plot(Wd, ModX, 'c-', lw=2, label="x(t)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Freq. [Hz]")
    ax[1].grid(True)
    if(limitF != 0):
        ax[1].set_xlim(0, limitF)
    ax[1].set_title(titleF)

    fig.tight_layout()

    return ModX, Wd

def main():
    # INFORMAÇÕES DO SISNAL DE TEMPO CONTÍNUO x(t)
    f = 100         # FREQUÊNCIA DO SINAL
    wo = 2*pi*f     # FREQUÊNCIA ANGULAR DO SINAL
    T = 1/f         # PERÍODO DO SINAL
    Tamc = T/1000   # PERÍODO DE AMOSTRAGEM DO SINAL

    # VETOR DE TEMPO CONTÍNUO PARA REPRESENTAR O SINAL x(t) CORRESPONDENDO A 6 PERÍODOS
    t = np.arange(0, 6*T, Tamc)
    x = 100*np.cos(wo*t)    # CONSTRUINDO O SINAL x(t)
    ModX, Wd = analisadorEspectro(x, t, 2**23, 2500, 'x(t)', '|X(jω)|')
    
    # PREPARANDO A AMOSTRAGEM
    Ts1 = T/10       # PERÍODO  DE AMOSTRAGEM DA INTERPOLAÇÃO = 0.001 s
    
    t1 = np.arange(0,6*T+1, Ts1)    # 
    x1 = 100*np.cos(wo*t1)          # OBTENDO AS AMOSTRAS PARA O PERÍODO DE AMOSTRAGEM Ts = 0.001 s
    x1interp = np.interp(t, t1, x1) # APLICANDO O RETENTOR DE PRIMEIRA ORDEM (INTERPOLAÇÃO LINEAR)
    analisadorEspectro(x1interp, t, 2**23, 2500, 'xs(t) Ts = ' + str(Ts1), '|Xs(jω)|')
   
    # FILTRAGEM
    RC = [0.35]
    xrf = filtragem_RC(x1interp, t, T, Tamc, RC)
    analisadorEspectro(xrf, t, 2**23, 2500, 'xs(t) Ts = ' + str(Ts1) + ' filtrado com RC = ' + str(RC[0]), '|Xs(jω)|')
    
    ''' --------------------------------------------------------------------------------------------'''
    ''' ---> MESMO PROCEDIMENTO, MAS AGORA PARA Ts = 0.0025 s <--- '''
    
    Tamc = T/2004   # PERÍODO DE AMOSTRAGEM DO SINAL
    # VETOR DE TEMPO CONTÍNUO PARA REPRESENTAR O SINAL x(t) CORRESPONDENDO A 6 PERÍODOS
    t = np.arange(0, 6*T, Tamc)
    x = 100*np.cos(wo*t)    # CONSTRUINDO O SINAL x(t)
    ModX, Wd = analisadorEspectro(x, t, 2**23, 2500, 'x(t)', '|X(jω)|')
    
    # PREPARANDO A AMOSTRAGEM
    Ts2 = T/4                       # PERÍODO  DE AMOSTRAGEM DA INTERPOLAÇÃO = 0.001 s

    t2 = np.arange(0,6*T+1, Ts2)    # 
    x2 = 100*np.cos(wo*t2)          # OBTENDO AS AMOSTRAS PARA O PERÍODO DE AMOSTRAGEM Ts = 0.001 s
    x2interp = np.interp(t, t2, x2) # APLICANDO O RETENTOR DE PRIMEIRA ORDEM (INTERPOLAÇÃO LINEAR)   
    ModX2, Wd2 = analisadorEspectro(x2interp, t, 2**23, 2500, 'xs(t) Ts = ' + str(Ts2), '|Xs(jω)|')

    # FILTRAGEM
    RC = [0.35]
    xrf = filtragem_RC(x2interp, t, T, Tamc, RC)
    analisadorEspectro(xrf, t, 2**23, 2500, 'xs(t) Ts = ' + str(Ts2) + ' filtrado com RC = ' + str(RC[0]), '|Xs(jω)|')
    
    
    plt.show()

main()
