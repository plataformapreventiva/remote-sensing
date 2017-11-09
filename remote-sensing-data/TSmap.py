# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


def TSmap(i):
        #=============================================================
        # funcion para visualizar la serie de tiempo de un municipio
        #
        # param i := numero entero que indica el  municipio en esta
        #            version (BETA) i \in {0,.., |{pickle_files}|-1 };
        #            |.| = card(.)
        #=============================================================


        # set path to the EVI arrays (Enhanced Vegetation Index) 
        path = './remote-sensing-data/arrays/*.p'
        pickles = glob.glob(path)


# -------------------------------------------------------------------
# ---------- puede servir en el futuro, para otra funcion: ----------
# -------------------------------------------------------------------
## vector of names
#names  = ['im' + str(i) for i in range(1,(len(pickles)+1))]
#
## read the pickles and save them in their respective name
#for i in range(0, (len(pickles))):
#	globals()[names[i]] = pd.read_pickle(pickles[i])
# -------------------------------------------------------------------
#esto es para hacer todos al mismo tiempo, pero se tiene que descomentar el bloque anterior:
#for each pickle make a grid with the time series of the "map"
#for im in names: #indent the following
#



        im = pd.read_pickle(pickles[i])
        a = len(pickles[i])- 30
        finit = pickles[i][(pickles[i].find('_', a)+1): (pickles[i].find('_', a)+11) ]
        ffin  = pickles[i][(pickles[i].find('-', a)+1): (pickles[i].find('-', a)+11) ]
        mpo = pickles[i][(a+1):a+6]; # la clave de municipio siempre tiene 5 caracteres

        for t in range(0,(im.shape[2])):
                image = im[:,:,t] 
                pl = plt.subplot(10,10,t+1)
                pl = plt.imshow(image, aspect = 'auto')
                pl.axes.get_xaxis().set_ticks([])
                pl.axes.get_yaxis().set_ticks([])
                if (t==5): {
                plt.title('Serie del índice EVI del '+finit+' al '+ffin+'\n para el municipio '+mpo)
                }
                #plt.title('Image'+str(t))

        plt.savefig(pickles[i][:-2]+'.png')        
        plt.show()

# make callable from the temrinal with a statement of the form:
#       python __path_to_file__\TSmap.py --i X ;where X is the integer argument i of TSmap.py
if __name__=='__main__':
        import argparse
        parser = argparse.ArgumentParser(description = 'visualizar la serie de tiempo del EVI de un municipio')
        parser.add_argument('--i', type = int, default = 10, help = 'param i := numero entero que indica el municipio \
                                                                                en esta version (BETA) i in {0,.., |{pickle_files}|-1 }; donde  |.| = card(.)' )
        args = parser.parse_args()
        municipio = args.i
        TSmap(municipio)
