# coding: utf-8

# # Estatísticas de Controle de Processos

#libs
import pandas as pd
import numpy as np
import statsmodels.graphics.tsaplots as smt
import statsmodels.tsa.stattools as sta

import matplotlib.pyplot as plt

#dados
df = pd.read_csv("/home/wesley/discovery/local_files/testes/data/exemplo_nove.csv")


#Shewhart
def shewhart_chart(df,m):
    factors = pd.read_csv("factors_control_chart.csv")
    
    #Control Limits for the R Chart
    r_mean = df['Ampl'].mean()
    ucl_r = factors['D4'][m-2]*r_mean
    lcl_r = factors['D3'][m-2]*r_mean
    
    #Control Limits for the X Chart
    x_mean = df['Mean'].mean()
    ucl_x = x_mean + factors['A2'][m-2]*r_mean
    lcl_x = x_mean - factors['A2'][m-2]*r_mean
    
    return(ucl_r, ucl_x)

#CUSUM Tabular 
def cusum_tabular(xi, k=1/2):
    xi_mean_k_pos = np.mean(xi) + k
    xi_mean_k_neg = np.mean(xi) - k
    
    ci_pos = []
    ci_neg = []
    xi_dif_mk = []
    mk_dif_xi = []
    n_positivo = []
    n_negativo = []
    n_pos = 0
    n_neg = 0
    
    for i in xi:
        #Ci positivo
        if len(ci_pos) == 0:
            xi_dif_mk.append(i - xi_mean_k_pos)
            ci_pos.append(max(0, i - xi_mean_k_pos + 0))
            if ci_pos[-1] == 0:
                n_pos = 0
            elif ci_pos[-1] > 0:
                n_pos += 1
        else:
            xi_dif_mk.append(i - xi_mean_k_pos)
            ci_pos.append(max(0, i - xi_mean_k_pos + ci_pos[-1]))
            if ci_pos[-1] == 0:
                n_pos = 0
            elif ci_pos[-1] > 0:
                n_pos += 1
        
        #Ci negativo
        if len(ci_neg) == 0:
            mk_dif_xi.append(xi_mean_k_neg - i)
            ci_neg.append(max(0, xi_mean_k_neg - i + 0))
            if ci_neg[-1] == 0:
                n_neg = 0
            elif ci_neg[-1] > 0:
                n_neg += 1
        else:
            mk_dif_xi.append(xi_mean_k_neg - i)
            ci_neg.append(max(0, xi_mean_k_neg - i + ci_neg[-1]))
            if ci_neg[-1] == 0:
                n_neg = 0
            elif ci_neg[-1] > 0:
                n_neg += 1
                
        n_positivo.append(n_pos)
        n_negativo.append(n_neg)
    
    df_cusum = pd.DataFrame({'xi':xi, 'xi-(mean+k)':xi_dif_mk, 'ci_pos':ci_pos, 'n_pos':n_positivo,
                            '(mean+k)-xi':mk_dif_xi, 'ci_neg':ci_neg, 'n_neg':n_negativo})
    
    df_cusum = df_cusum[['xi', 'xi-(mean+k)', 'ci_pos', 'n_pos', '(mean+k)-xi', 'ci_neg', 'n_neg']]
    
    H = 5 *  np.std(xi)
    
    return df_cusum, H
    



#The Standardized Cusum
def cusum_standardized(xi, k=1/2):
    yi = (xi - np.mean(xi))/np.std(xi)
    
    ci_pos = []
    ci_neg = []
    yi_dif_mk = []
    mk_dif_yi = []
    n_positivo = []
    n_negativo = []
    n_pos = 0
    n_neg = 0
    
    for i in yi:
        #Ci positivo
        if len(ci_pos) == 0:
            yi_dif_mk.append(i - k)
            ci_pos.append(max(0, i - k + 0))
            if ci_pos[-1] == 0:
                n_pos = 0
            elif ci_pos[-1] > 0:
                n_pos += 1
        else:
            yi_dif_mk.append(i - k)
            ci_pos.append(max(0, i - k + ci_pos[-1]))
            if ci_pos[-1] == 0:
                n_pos = 0
            elif ci_pos[-1] > 0:
                n_pos += 1
        
        #Ci negativo
        if len(ci_neg) == 0:
            mk_dif_yi.append(-k - i)
            ci_neg.append(max(0, -k - i + 0))
            if ci_neg[-1] == 0:
                n_neg = 0
            elif ci_neg[-1] > 0:
                n_neg += 1
        else:
            mk_dif_yi.append(-k - i)
            ci_neg.append(max(0, -k - i + ci_neg[-1]))
            if ci_neg[-1] == 0:
                n_neg = 0
            elif ci_neg[-1] > 0:
                n_neg += 1
                
        n_positivo.append(n_pos)
        n_negativo.append(n_neg)
    
    df_cusum = pd.DataFrame({'yi':yi, 'yi-(mean+k)':yi_dif_mk, 'ci_pos':ci_pos, 'n_pos':n_positivo,
                            '(mean+k)-yi':mk_dif_yi, 'ci_neg':ci_neg, 'n_neg':n_negativo})
    
    df_cusum = df_cusum[['yi', 'yi-(mean+k)', 'ci_pos', 'n_pos', '(mean+k)-yi', 'ci_neg', 'n_neg']]
    limit_sup = 5*np.std(yi)
    
    return df_cusum, limit_sup 
    



#The Exponentially Weighted Moving Average Control Chart
def ewma_chart(xi, lambd = 0.1, l = 2.7):
    
    xi_mean = np.mean(xi)
    xi_std = np.std(xi)
    yi = []
    count = 1
    ucl = []
    lcl = [] 
    
    for i in xi:
        if len(yi) == 0:
            yi.append(lambd*i + (1-lambd)*xi_mean)
            ucl.append(xi_mean + l * xi_std * np.sqrt((lambd/(2-lambd))*(1-(1-lambd)**(2*count))))
            lcl.append(xi_mean - l * xi_std * np.sqrt((lambd/(2-lambd))*(1-(1-lambd)**(2*count))))
            count += 1
        else:
            yi.append(lambd*i + (1-lambd)*yi[-1])
            ucl.append(xi_mean + l * xi_std * np.sqrt((lambd/(2-lambd))*(1-(1-lambd)**(2*count))))
            lcl.append(xi_mean - l * xi_std * np.sqrt((lambd/(2-lambd))*(1-(1-lambd)**(2*count))))
            count += 1
    df_ewma = pd.DataFrame({'xi':xi, 'yi':yi, 'ucl':ucl, 'lcl':lcl})
    
    return df_ewma


# The Batch Means Control Chart
def batch_means(xi, b):
    # xi e um vetor contendo valores da variavel de interesse
    # b e o tamanho do batch, o melhor batch e aquele que reduz a autocorrelacao no lag 1
    
    paf = sta.pacf(xi, nlags=int(len(xi)-len(xi)*0.4), alpha=0.5)[0][1] #partial autocorrelation function
    
    if paf < 0.2:
        
        return xi
    
    else:
        count = 1
        while (paf >= 0.2) and (count <= 4):    
            yi = []
            n = len(xi)
            wind_inf = 0
            wind_sup = wind_inf + b 

            while wind_sup <= n:
                yi.append(np.mean(xi[wind_inf:wind_sup]))
                wind_inf += b
                wind_sup += b
            
            b = b * 2
            paf = sta.pacf(yi, nlags=int(len(yi)-len(yi)*0.4), alpha=0.5)[0][1]
            count += 1
    
    return yi