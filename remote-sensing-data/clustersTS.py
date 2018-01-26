# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from random import randint
#import math
import csv
import datetime

import os
##############
os.chdir("C:\\Users\\Raul Vazquez\\Desktop\\NNproj\\remote-sensing\\remote-sensing-data")
#%matplotlib auto
##############
import utils_loc



path = './arrays/*.p'
tss = glob.glob(path)
path2 = './clusters/*.p'
clusts = glob.glob(path2)

# read the data for municpality (i+1)
i = 6 # remember: we are reading municipality i+1
corte  = 2 # define the clusterization cut-value desired
im = pd.read_pickle(tss[i]) # read the municipality NDVI/EVI time series values
clust = pd.read_pickle(clusts[(i*5 + corte)]).astype('int16') # read the cluster "map"

cve_mpo = tss[i][(tss[i].find('\\')+1):tss[i].find('_')]
# visualize the TS
utils_loc.gridPlotsTS(im)


def read_agricolaDB(init_date, end_date):
    '''
    load the database that contains the objective value 
        y=superficie_Cosechada/superficie_Sembrada
    INPUT:
        - init_date:(datetime.date) Date of NDVI´s Time Series first observation
        - end_date: (datetime.date) Date of NDVI´s Time Series last observation
    OUTPUT:
        y_TPV: (DataFrame) 'temporal' & 'primavera-verano' y_variable, indexed over the dates
        y_RPV: (DataFrame) 'riego' & 'primavera-verano' y_variable, indexed over the dates
        y_TOI: (DataFrame) 'temporal' & 'otono-invierno' y_variable, indexed over the dates
        y_ROI: (DataFrame) 'temporal' & 'otono-invierno' y_variable, indexed over the dates
    '''
    path_y = './avance_agricola/agricola_201712.csv'
    with open(path_y,'r') as f:
        reader = csv.reader(f, skipinitialspace=False)
        colnames = np.array(next(reader))
        db1 = np.array([(x) for x in reader if x[1] == cve_mpo]) # only interested on the municipality (i+1)
    '''
    NOT SURE OF THIS ANYMORE:
    # the y variable ig going to be 4-months lagged
    from dateutil.relativedelta import relativedelta
    init_date += relativedelta(months = 4)
    end_date  += relativedelta(months = 4) 
    '''
    df = pd.DataFrame(db1, columns = colnames, dtype = str ) # see values EXAMPLE: df.loc[1,:]
    '''PREPROCESS observed y_ variable'''
    # transform desired variables from strings to floats
    numeric_vars = ['mes','anio','sup_sembrada', 'sup_cosechada', 'sup_siniestrada',
                'produccion', 'rendimiento', 'mes_agricola', 'anio_agricola']
    for var in numeric_vars:
        df[var] = df[var].map(lambda x: x.replace(',',''))

    df[numeric_vars] = df[numeric_vars].apply(pd.to_numeric)


    #df[(df['sup_sembrada'] == df['sup_cosechada'] + df['sup_siniestrada'])]
    df['date'] = [datetime.datetime(x,y,1) for x,y in zip(df['anio'],df['mes'])] 
    df['date_agricola'] = [datetime.datetime(x,y,1) for x,y in zip(df['anio_agricola'],df['mes_agricola'])] 

    # choose only the dates we are interested in (i.e., the ones for which we have the ndvi TS)
    df = df.loc[(df['date'] >= init_date) & (df['date'] <= end_date)]

    # sort in ascending date values
    df = df.sort_values(by = 'date', ascending = True)

    # generate observed value harvested_surface/planted_surface
    df['prop_cosecha'] = df['sup_cosechada']/df['sup_sembrada']
    
    # divide y variable by agricultural cycle ("ciclo") and watering method ("modalidad hidrica")
    y_TPV = df.loc[(df['moda_hidr'] == 'T') & (df['ciclo'] == 'PV')][['date','prop_cosecha']]
    y_RPV = df.loc[(df['moda_hidr'] == 'R') & (df['ciclo'] == 'PV')][['date','prop_cosecha']]
    y_TOI = df.loc[(df['moda_hidr'] == 'T') & (df['ciclo'] == 'OI')][['date','prop_cosecha']]
    y_ROI = df.loc[(df['moda_hidr'] == 'R') & (df['ciclo'] == 'OI')][['date','prop_cosecha']]
    return (y_TPV.set_index('date'),
            y_RPV.set_index('date'),
            y_TOI.set_index('date'),
            y_ROI.set_index('date'))



# load the database that contains the objective value (y variable)
path_y = './avance_agricola/agricola_201712.csv'


with open(path_y,'r') as f:
    reader = csv.reader(f, skipinitialspace=False)
    colnames = np.array(next(reader))
    db1 = np.array([(x) for x in reader if x[1] == cve_mpo]) # only interested on the municipality (i+1)

# use only the values in observed ndvi time series 
init_date = [int(x) for x in (tss[i][(1+tss[i].find('_')):tss[i].find('-')]).split(sep = '.') ]
init_date = datetime.date(init_date[0], init_date[1], init_date[2])
end_date = [int(x) for x in (tss[i][(1+tss[i].find('-')):-2]).split(sep='.')]
end_date = datetime.date(end_date[0], end_date[1], end_date[2])
'''
NOT SURE OF THIS ANYMORE:
# the y variable ig going to be 4-months lagged
from dateutil.relativedelta import relativedelta
init_date += relativedelta(months = 4)
end_date  += relativedelta(months = 4) 
'''

df = pd.DataFrame(db1, columns = colnames, dtype = str ) # see values EXAMPLE: df.loc[1,:]

# ========================================================
# ================== PREPROCESS ==========================
# ========================================================
'''PREPROCESS observed y_ variable'''
# transform desired variables from strings to floats
numeric_vars = ['mes','anio','sup_sembrada', 'sup_cosechada', 'sup_siniestrada',
                'produccion', 'rendimiento', 'mes_agricola', 'anio_agricola']
for var in numeric_vars:
    df[var] = df[var].map(lambda x: x.replace(',',''))

df[numeric_vars] = df[numeric_vars].apply(pd.to_numeric)


#df[(df['sup_sembrada'] == df['sup_cosechada'] + df['sup_siniestrada'])]
df['date'] = [datetime.datetime(x,y,1) for x,y in zip(df['anio'],df['mes'])] 
df['date_agricola'] = [datetime.datetime(x,y,1) for x,y in zip(df['anio_agricola'],df['mes_agricola'])] 


# plot the production and loss in the municipality
# dates were not added; they would be repeated 
plt.plot(df['sup_sembrada'], label = 'sembrada'), plt.plot(df['sup_cosechada'], label = 'cosechada'), plt.plot(df['sup_siniestrada'], label = 'siniestrada'), plt.legend(), plt.title('Superficie sembrada, cosechada y siniestrada en el municipio ' + cve_mpo)


# choose only the dates we are interested in (i.e., the ones for which we have the ndvi TS)
df = df.loc[(df['date'] >= init_date) & (df['date'] <= end_date)]

# sort in ascending date values
df = df.sort_values(by = 'date', ascending = True)

# generate observed value harvested_surface/planted_surface
df['prop_cosecha'] = df['sup_cosechada']/df['sup_sembrada']


# I WILL FIRST RUN DIFFERENT MODELS TO SEE WHICH IS BETTER:
# not every municiplaity reports all of the values for T or R

# divide y variable by agricultural cycle ("ciclo") and watering method ("modalidad hidrica")
y_TPV = df.loc[(df['moda_hidr'] == 'T') & (df['ciclo'] == 'PV')][['date','prop_cosecha']]
y_RPV = df.loc[(df['moda_hidr'] == 'R') & (df['ciclo'] == 'PV')][['date','prop_cosecha']]
y_TOI = df.loc[(df['moda_hidr'] == 'T') & (df['ciclo'] == 'OI')][['date','prop_cosecha']]
y_ROI = df.loc[(df['moda_hidr'] == 'R') & (df['ciclo'] == 'OI')][['date','prop_cosecha']]


'''VUSUALIZE Primavera-Verano Values:'''
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())# .MonthLocator())
plt.plot(np.asarray(y_RPV['date'], dtype='datetime64[ns]'),y_RPV['prop_cosecha'], label = 'riego prim-Ver')
plt.plot(np.asarray(y_TPV['date'], dtype='datetime64[ns]'), y_TPV['prop_cosecha'], label = 'temporal prim-Ver')
plt.title('Proporción de superficie cosechada respecto a la sembrada')
plt.legend()
xfmt = mdates.DateFormatter('%b %Y')
ax.xaxis_date()
plt.gca().xaxis.set_major_formatter(xfmt)
plt.gcf().autofmt_xdate() # tilts the x labels
plt.show()

'''VISUALIZE Otonho-Invierno Values'''
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())# .MonthLocator())
plt.plot(y_ROI['date'],y_RPV['prop_cosecha'])
plt.plot(y_TOI['date'], y_TPV['prop_cosecha'])
plt.gcf().autofmt_xdate()


'''I M P O R T A N T:
    ErrorValue in last plot due because for municipality 01007, y_TOI and y_TOI are empty!!!
'''


dates = utils_loc.perdelta(init_date, end_date, relativedelta(months=1))
df_temp = pd.DataFrame(dates, columns = ['date'], index = range(len(dates)))
df_temp = df_temp.loc[df_temp['date']>= min(x.min() for x in [y_TPV['date'],y_RPV['date']]) ]


df2 = pd.merge(df_temp, y_TPV, on = 'date', how = 'left')
df2 = pd.merge(df2, y_RPV, on = 'date', how = 'left')
df2 = pd.merge(df2, y_TOI, on = 'date', how = 'left')
df2 = pd.merge(df2, y_ROI, on = 'date', how = 'left')
df2.columns = ['date','tpv', 'rpv','toi','roi']

''''''
df2['toi'][0] = 0
df2['roi'][0] = 0
# fill the NaN values with known values
''' TO DO:
    make this next part more efficient!!!!!!'''
for i in range(len(df2)):
    if (df2['date'][i].month == 4):
        if pd.isnull(df2['tpv'][i]):
            df2['tpv'][i] = 0
        if pd.isnull(df2['rpv'][i]):
            df2['rpv'][i] = 0
        if pd.isnull(df2['toi'][i]):
            df2['toi'][i] = 0
        if pd.isnull(df2['roi'][i]):
            df2['roi'][i] = 0
    else:
        if pd.isnull(df2['tpv'][i]):
            df2['tpv'][i] = df2['tpv'][i-1]
        if pd.isnull(df2['rpv'][i]):
            df2['rpv'][i] = df2['rpv'][i-1]
        if pd.isnull(df2['toi'][i]):
            df2['toi'][i] = df2['toi'][i-1]
        if pd.isnull(df2['roi'][i]):
            df2['roi'][i] = df2['roi'][i-1]
            
#df2['tpv2'] = [ df2['tpv'][i-1]   for i in range(len(df2['tpv'])) if (pd.isnull(df2['tpv'][i]))  ]


# these ones accumulate T and R modalities:
df2['temporal'] = df2['tpv'] + df2['toi']
df2['riego'] = df2['rpv'] + df2['roi']





'''PREPROCESS X variable'''
# aggregate data of the pixels time series per cluster
meanTS, DBA, iterdDBA = utils_loc.gen_clusterTS(im, clust)
iterdDBA
TS_clusts = np.zeros([clust.max(),iterdDBA.shape[2] ])
'''TO DO: give back the Time Series of each element in the cluster '''
for c in range(clust.max()+1):
    '''I think I have done it in utils_loc '''
    pass

# visualize the aggregations
utils_loc.gridPlotsTS(meanTS)
utils_loc.gridPlotsTS(DBA)
utils_loc.gridPlotsTS(iterdDBA)


plt.figure()
plt.plot(dba, label = 'DBA-1')
plt.plot(media, '--', label = 'x-axis mean')
for j in  range(len(imageTS[cloc])):
    plt.plot(imageTS[cloc][j], '--', alpha=0.3, label='clusterMap' + str(c)+' j'+str(j))
plt.plot(dba_iterated, label='DBA-iterated')
plt.legend()























































#def read_NDVI(mun_num, clust_cut_val):
'''
    Read the NDVI pickles. The ones already generated by the "modis-luigi.py" pipeline
    INPUT:
        - mun_num: (int) municipality number 
            (TO_DO: mun_num does not include the whole number. La cve_ent tiene que ser incorporada, para buscar municipios dentro de estados.... ahora es solo de "jugete")
        - clust_cut_val: (int) desired clusterization cut-value, i.e., the cut-value parameter the clusters were created with in the "modis-luigi.py" pipeline
    
    OUTPUT:
        - cve_mun: Municipality number (including the cve_ent)
    '''
#    path = './arrays/*.p'
#    tss = glob.glob(path)
#    path2 = './clusters/*.p'
#    clusts = glob.glob(path2)
#     
#    i = mun_num
#    corte = clust_cut_val
#    im = pd.read_pickle(tss[i]) # read the municipality NDVI/EVI time series values
#    clust = pd.read_pickle(clusts[(i*5 + corte)]).astype('int16') # read the cluster "map"
#    cve_mpo = tss[i][(tss[i].find('\\')+1):tss[i].find('_')]
#    return(cve_mpo, im, clust)