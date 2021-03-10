import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import math as mat
import statsmodels.api as sm
from scipy import stats
import pylab
import tabulate #from tabulate


#print("Definisci il limite commissionale annuo")
#Spesa_massima_annua=input()

data_inizio_analisi= dt.datetime(2019,12, 31)
data_fine_analisi= dt.datetime (2020, 12, 31)

# print("Quale è il primo asset da inserire in analisi")
# Ticker1 = input()
#
# print("Quale è il secondo asset da inserire in analisi")
# Ticker2 = input()
#
# print("Quale è il terzo asset da inserire in analisi")
# Ticker3 = input()
#
# print("Quale è il quarto asset da inserire in analisi")
# Ticker4 = input()
#
# print("Quale è il quinto asset da inserire in analisi")
# Ticker5 = input()
#
# dataset_Asset=[Ticker1, Ticker2, Ticker3, Ticker4, Ticker5]
dataset_Asset=['KO', 'TSLA', 'NIO', 'AAPL', 'GM']
print(dataset_Asset)

dati_Asset_1 = web.DataReader( dataset_Asset[0], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[0], dati_Asset_1.shape)
dati_Asset_2 = web.DataReader( dataset_Asset[1], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[1], dati_Asset_2.shape)
dati_Asset_3 = web.DataReader( dataset_Asset[2], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[2], dati_Asset_3.shape)
dati_Asset_4 = web.DataReader( dataset_Asset[3], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[3], dati_Asset_4.shape)
dati_Asset_5 = web.DataReader( dataset_Asset[4], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[4], dati_Asset_5.shape)

quotazioni_1 = dati_Asset_1.values
lista_chiusura_Asset_1 = quotazioni_1 [:,5]
quotazioni_2 = dati_Asset_2.values
lista_chiusura_Asset_2 = quotazioni_2 [:,5]
quotazioni_3 = dati_Asset_1.values
lista_chiusura_Asset_3 = quotazioni_3 [:,5]
quotazioni_4 = dati_Asset_1.values
lista_chiusura_Asset_4 = quotazioni_4 [:,5]
quotazioni_5 = dati_Asset_1.values
lista_chiusura_Asset_5 = quotazioni_5 [:,5]

lista_rendimenti_Asset_1= np.array([])
lista_rendimenti_Asset_2= np.array([])
lista_rendimenti_Asset_3= np.array([])
lista_rendimenti_Asset_4= np.array([])
lista_rendimenti_Asset_5= np.array([])

i=0
lunghezza = lista_chiusura_Asset_1.size
while (i<lunghezza-1):
    rendimento=np.log (lista_chiusura_Asset_1[i+1]/lista_chiusura_Asset_1[i])
    lista_rendimenti_Asset_1 = np.append (lista_rendimenti_Asset_1, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_2[i+1]/lista_chiusura_Asset_2[i])
    lista_rendimenti_Asset_2 = np.append (lista_rendimenti_Asset_2, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_3[i+1]/lista_chiusura_Asset_3[i])
    lista_rendimenti_Asset_3 = np.append (lista_rendimenti_Asset_3, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_4[i+1]/lista_chiusura_Asset_4[i])
    lista_rendimenti_Asset_4 = np.append (lista_rendimenti_Asset_4, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_5[i+1]/lista_chiusura_Asset_5[i])
    lista_rendimenti_Asset_5 = np.append (lista_rendimenti_Asset_5, [rendimento])

    i=i+1


# plt.hist (lista_rendimenti_Asset_1, bins=75, density=False)
# stringa = 'distribuzione rendimenti' + dataset_Asset[0]
# plt.title(stringa)
# plt.show()
#
# Shapiro_Wilk_Test= stats.shapiro (lista_rendimenti_Asset_1)
# p_value= Shapiro_Wilk_Test[1]
# print('il p-value è', p_value)
#
# if p_value <= 0.05 :
#     print('l ipotesi nulla di normalità è rifiutata')
# else:
#     print('l ipotesi nulla di normalità è accettata')
#
# sm.qqplot (lista_rendimenti_Asset_1='s', dist= stats.norm, fit=True)
# stringa = 'grafico QQ di' + dataset_Asset[0]
# plt.title(stringa)
# pylab.show ()

var_90=pd.DataFrame(lista_rendimenti_Asset_1)[0].quantile(0.1)
var_95=pd.DataFrame(lista_rendimenti_Asset_1)[0].quantile(0.05)
var_99=pd.DataFrame(lista_rendimenti_Asset_1)[0].quantile(0.01)

#print(tabulate([['90%', var_90],['95%',var_95],['99%',var_99]],
 #              headers= 'livello di confidenza', 'Value at risk' dataset_Asset[0]))
print(var_90)
print(var_95)
print(var_99)

vettore_rendimenti_medi = []
rendimenti_medi= lista_rendimenti_Asset_1.mean()
vettore_rendimenti_medi.append([rendimenti_medi])

rendimenti_medi= lista_rendimenti_Asset_2.mean()
vettore_rendimenti_medi.append([rendimenti_medi])

rendimenti_medi= lista_rendimenti_Asset_3.mean()
vettore_rendimenti_medi.append([rendimenti_medi])

rendimenti_medi= lista_rendimenti_Asset_4.mean()
vettore_rendimenti_medi.append([rendimenti_medi])

rendimenti_medi= lista_rendimenti_Asset_5.mean()
vettore_rendimenti_medi.append([rendimenti_medi])

vettore_varianze=[]
varianze= lista_rendimenti_Asset_1.var()
vettore_varianze.append([varianze])

varianze= lista_rendimenti_Asset_2.var()
vettore_varianze.append([varianze])

varianze= lista_rendimenti_Asset_3.var()
vettore_varianze.append([varianze])

varianze= lista_rendimenti_Asset_4.var()
vettore_varianze.append([varianze])

varianze= lista_rendimenti_Asset_5.var()
vettore_varianze.append([varianze])

fig, ax =plt.subplots(figsize=(8,8))
plt.rcParams ['lines.markersize'] = 12
colori = ['red', 'blue', 'green', 'black', 'orange']
simboli = ['o', '*','^','p','s']

ctr=0
while ctr<5:
    ax.scatter (vettore_rendimenti_medi[ctr], vettore_varianze[ctr], c=colori[ctr], marker= simboli[ctr], label=dataset_Asset[ctr])
    ctr= ctr +1

ax.set_xlabel ('Valore atteso')
ax.set_ylabel ('Varianza')
ax.legend()
plt.show()

data= {dataset_Asset[0]: [0,0,0,0],
      dataset_Asset[1]: [0,0,0,0],
      dataset_Asset[2]: [0,0,0,0],
      dataset_Asset[3]: [0,0,0,0],
      dataset_Asset[4]: [0,0,0,0]}

tab_cov= pd.DataFrame.from_dict(data, orient='index', columns=[dataset_Asset[0], dataset_Asset[1],dataset_Asset[2], dataset_Asset[3], dataset_Asset[4]])

x1=np.array(lista_rendimenti_Asset_1,lista_rendimenti_Asset_2)
tab=np.cov(x1)

tab_cov.iloc[0,0]=tab[0,0]
tab_cov.iloc[1,1]=tab[1,1]
tab_cov.iloc[0,1]=tab[0,1]
tab_cov.iloc[1,0]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_1,lista_rendimenti_Asset_3)
tab=np.cov(x1)

tab_cov.iloc[2,2]=tab[1,1]
tab_cov.iloc[0,2]=tab[0,1]
tab_cov.iloc[2,0]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_1,lista_rendimenti_Asset_4)
tab=np.cov(x1)

tab_cov.iloc[3,3]=tab[1,1]
tab_cov.iloc[0,3]=tab[0,1]
tab_cov.iloc[3,0]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_1,lista_rendimenti_Asset_5)
tab=np.cov(x1)

tab_cov.iloc[4,4]=tab[1,1]
tab_cov.iloc[0,4]=tab[0,1]
tab_cov.iloc[4,0]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_2,lista_rendimenti_Asset_3)
tab=np.cov(x1)

tab_cov.iloc[1,2]=tab[0,1]
tab_cov.iloc[2,1]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_2,lista_rendimenti_Asset_4)
tab=np.cov(x1)

tab_cov.iloc[1,3]=tab[0,1]
tab_cov.iloc[3,1]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_2,lista_rendimenti_Asset_5)
tab=np.cov(x1)

tab_cov.iloc[1,4]=tab[0,1]
tab_cov.iloc[4,1]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_3,lista_rendimenti_Asset_4)
tab=np.cov(x1)

tab_cov.iloc[2,3]=tab[0,1]
tab_cov.iloc[3,2]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_3,lista_rendimenti_Asset_5)
tab=np.cov(x1)

tab_cov.iloc[2,4]=tab[0,1]
tab_cov.iloc[4,2]=tab[1,0]

x1=np.array(lista_rendimenti_Asset_4,lista_rendimenti_Asset_5)
tab=np.cov(x1)

tab_cov.iloc[3,4]=tab[0,1]
tab_cov.iloc[4,3]=tab[1,0]

tab_corr=tab_cov.copy()
i=0
j=0
while i<5:
    j=0
    while j<5:
        prodotto_varianze= tab_cov.iloc[i,i]*tab_cov[j,j]
        tab_corr.iloc[i,j]=tab_cov.iloc[i,j]/mat.sqrt(prodotto_varianze)
        j=j+1
    i=i+1

tab_cov.style.set_caption('<b><i>matrice covarianze')
tab_corr.style.set_caption(('<b><i>matrice indici correlazioni'))

rendimenti=np.empty((5,0), float)

rendimenti= np.append(rendimenti, [(lista_rendimenti_Asset_1.mean()*252)])
rendimenti= np.append(rendimenti, [(lista_rendimenti_Asset_2.mean()*252)])
rendimenti= np.append(rendimenti, [(lista_rendimenti_Asset_3.mean()*252)])
rendimenti= np.append(rendimenti, [(lista_rendimenti_Asset_4.mean()*252)])
rendimenti= np.append(rendimenti, [(lista_rendimenti_Asset_5.mean()*252)])

num_portafogli= 100000
risultati = np.zeros((4+len(dataset_Asset)-1, num_portafogli))
for i in range(num_portafogli):
    pesi=np.array(np.random.random(5))
    pesi /= np.sum(pesi)

    rendimento_portafoglio = np.sum(rendimenti*pesi)
    std_dev_portafoglio= np.sqrt(np.dot(pesi.T, np.dot(tab_cov, pesi)))*np.sqrt(252)

    risultati[0,i] = rendimento_portafoglio
    risultati[1,i] = std_dev_portafoglio
    risultati[2,i] = risultati [0,i] / risultati [1,i]

    for j in range(len(pesi)):
        risultati[j+3,i] = pesi [j]


frame_risultati = pd.DataFrame(risultati.T, columns=['ret','stdev','sharpe', dataset_Asset[0], dataset_Asset[1], dataset_Asset[2], dataset_Asset[3],dataset_Asset[4]])

max_sharpe_port = frame_risultati.iloc[frame_risultati['sharpe'].idxmax()]
min_var_port = frame_risultati.iloc[frame_risultati['stdev'].idxmin()]

plt.figure (figsize=(10,6))
plt.scatter ( frame_risultati.stdev, frame_risultati.ret, c= frame_risultati.sharpe , marker='.', cmap='coolwarm')

plt.xlabel ('deviazione standard')
plt.ylabel ('rendimento atteso')
plt.colorbar (label= 'Sharpe ratio')
plt.title ('frontiera portafogli')

plt.scatter (max_sharpe_port[1],max_sharpe_port[0], marker=(5,1,0), color='r', s=300)
plt.scatter(min_var_port[1], min_var_port[0], marker=(5,1,0), color='g', s=300)