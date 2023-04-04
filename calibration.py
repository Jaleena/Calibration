import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy import integrate
#from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib as mpl
import scipy.stats
#import statsmodels.api as sm
import pylab
import pandas as pd
import itertools
import areametric_easy as am
from matplotlib.ticker import ScalarFormatter
import random
import dataset as dm


log10 = np.log10
clin = -0.6
numbr = 500
AM_sigma_minimum = float('inf')
site_out = []
area = []
l10_boore = []
all_i = []
i_val = []
Y_out_log = []
y_inv_u = []
y_inv_l = []
Y_opti = []

#defining the ecdf as a step function and getting the stairs for plots
def ecdf(data):
        x1 = np.sort(data)
        x = x1.tolist()
        n = len(x)
        p = 1/n
        pvalues = list(np.linspace(p,1,n))
        return x, pvalues
def stairs(data):
        def stepdata(x,y): # x,y must be python lists
            
            xx,yy = x*2, y*2
            xx.sort()
            yy.sort()
            return xx, [0.]+yy[:-1]
        x, p = ecdf(data)
        x, y = stepdata(x,p)
        return x, y

#to get confidence band around empirical data using Dvoretzky–Kiefer–Wolfowitz inequality.
def conf(data,target):
    X,Y = ecdf(data)
    b_lo = []
    b_hi = []
    n = len(Y)
    alpha = 1 - target
    e = np.sqrt(((np.log(2/alpha))/(2*n)))

    for i in range(len(Y)):
        if Y[i] - e < 0:
            b_lo.append(max(0,Y[i] - e))
        else:
            b_lo.append(Y[i] - e)
        if Y[i] + e > 1:
            b_hi.append(min(1,Y[i] + e))
        else:
            b_hi.append(Y[i] + e)
        
    return b_lo,b_hi

#to plot the confidence band around empirical data using Dvoretzky–Kiefer–Wolfowitz inequality.
def conf_plot(data,target):
    X,Y = stairs(data)
    b_lo = []
    b_hi = []
    n = len(Y)
    alpha = 1 - target
    e = np.sqrt(((np.log(2/alpha))/(2*n)))

    for i in range(len(Y)):
        if Y[i] - e < 0:
            b_lo.append(max(0,Y[i] - e))
        else:
            b_lo.append(Y[i] - e)
        if Y[i] + e > 1:
            b_hi.append(min(1,Y[i] + e))
        else:
            b_hi.append(Y[i] + e)
        
    return b_lo,b_hi



#getting inverse confidence band for the corresponding confidence bands.
def inverse_confidence_band(d,u,alpha):
    n=len(d)
    h = np.sqrt(((np.log(2/alpha))/(2*n)))
    x,y=dm.ecdf(d)
    y_hi = y+h
    y_lo = y-h
    y_hi[y_hi>1]=1
    y_lo[y_lo<0]=0
    o = int(h//(1/n)+1)
    a = h+(1/n)
    b = h+(1/n)*(n-o)
    if u<a:
        index_le=0
    elif u>b:
        index_le=-1
    else:
        index_le = int( (u-a)//(1/n) + 1 )
    x_left = np.asarray([-2]+list(x[1:-o+1]))
    if (1-u)<a:
        index_ri=-1
    elif (1-u)>b:
        index_ri=0
    else:
        index_ri = -int( ((1-u)-a)//(1/n) + 2 )
    x_right = np.asarray(list(x[o-1:-1])+[10])
    return y_hi,y_lo, x, x_left[index_le],x_right[index_ri]




#reading the observed data from Italy.
df1 = pd.read_csv("/Users/jaleenasunny/code_notebook/AM_Calibration/data/italy_esm_final.csv",sep=',')
dfu = df1['v']
dfw = df1['w']
r = df1['Repi']
m = df1['Mw']
vs30 = df1['vs30']
station = df1['stn']

#reading the resampled csv file for the analysis
df_samp1 = pd.read_csv("/Users/jaleenasunny/code_notebook/AM_Calibration/data/italy_resampled.csv",sep=',')
Repi_resampled = df_samp1['Repi']
dfu_resample = df_samp1['u']
dfw_resample = df_samp1['w']
mw_resampled = df_samp1['Mw']

#reading the output from SMSIM for 493 simulations
df_out = pd.read_csv("/Users/jaleenasunny/code_notebook/AM_Calibration/data/resampled_out.csv",sep=',',header=None)

#reading the output from SMSIM using the initial parameters - from Bindi and Kotha (2020).
df_originalout = pd.read_csv("/Users/jaleenasunny/code_notebook/AM_Calibration/data/italy_originalout_resam.csv",sep=',',header=None)


#getting the geometric mean and log 10 of pga values of the horizontal components
#for the observed data
gm = [a * b for a,b in zip(dfu,dfw)]
obspgar = [math.sqrt(abs(i)) for i in gm]
obspga=[math.log10(i) for i in obspgar]
#for the resampled data
gm_resam = [a * b for a,b in zip(dfu_resample,dfw_resample)]
obspgar_resam = [math.sqrt(abs(i)) for i in gm_resam]
obspga_resam = [math.log10(i) for i in obspgar_resam]


if __name__=='__main__': 

    #getting the SMSIM initial output with site effects
    for i in range (len(df_originalout)):
        l10 = [math.log10(j) for j in df_originalout.iloc[:,0]]
        for k in range(len(l10)):
            if vs30[k] == 0:
                boore = 0
            else:
                boore =  clin * math.log(min(vs30[k],1500)/760)* np.log10(2.71828)
        site_out = [boore+j for j in l10]

    #getting all AM values along with the site effects
    for i in range (493):
        l10 = [math.log10(j) for j in df_out.iloc[:,i]]
        for k in range(len(l10)):
            if vs30[k] == 0:
                boore = 0
            else:
                boore =  clin *math.log(min(vs30[k],1500)/760) * np.log10(2.71828)
        l10_boore = [boore+j for j in l10]   
        a = am.areaMe(l10_boore,obspga_resam)
        area.append(a)
    
    #getting the minimum AM value index
    for i in range(len(area)):
        if area[i] < AM_sigma_minimum:
            AM_sigma_minimum = area[i]
            f = i
    #corresponding SMSIM optimum output along with site effects for plotting
    optimum = df_out[f] 
    optimum_log = [math.log10(i) for i in optimum]
    optimum_log_boore = []
    for k in range(len(optimum_log)):
        if vs30[k] == 0:
            boore = 0
        else:
             boore =  clin *math.log(min(vs30[k],1500)/760)* np.log10(2.71828)
        optimum_log_boore = [boore+j for j in optimum_log]
        

        
    #plot of the initial AM before calibration
    
    #am.plot(site_out,obspga_resam)
    #plt.show()
    

    #plot of am for the optimum parameter from the calculation
    #fig = plt.figure(figsize=(8,8))
    #am.plot(optimum_log_boore,obspga_resam)
    #plt.show()


    AM_sort = np.sort(area)
    i = np.argsort(area)   
    all_i.append(i[:numbr])
    i_val = all_i[0].tolist()
    Y_out = df_out.values.T.tolist()

    for i in range (len(Y_out)):
        gh = Y_out[i]
        Y_out_log.append([math.log10(j) for j in gh])

    #getting the inverse confidence band of the data
    for ui in np.linspace(0,1,len(obspga_resam)): #[0.1,0.4,0.5,0.6,0.7,0.8,0.999]:
        _, _, _, y_u,y_l = inverse_confidence_band(obspga_resam,ui,alpha=0.01)
        y_inv_u.append(y_u)
        y_inv_l.append(y_l)

    #plotting the ecdfs inside the confidence band
    for j in range(len(i_val)):
        i = i_val[j]
        m,n = stairs(Y_out_log[i])
        m1,n1 = ecdf(Y_out[i])
        
        y_lo,y_hi = conf_plot(obspga_resam,0.99)
        X,Y = stairs(obspga_resam)
        

        D1 = dm.dataset_parser(y_inv_u)
        D2 = dm.dataset_parser(Y_out_log[i])
        D3 = dm.dataset_parser(y_inv_l)
        u1_ = dm.ecdf(D1)
        u1 = u1_[1]
        u2_ = dm.ecdf(D2)
        u2 = u2_[1]
        u3_ = dm.ecdf(D3)
        u3 = u2_[1]
        num_toleft = dm.pseudoinverse_(D1,u2-1e-9) - D2.value()[D2.index()]>0
        how_many_d2_to_the_left = sum(num_toleft)
        percent_left = how_many_d2_to_the_left / len(obspga_resam)
        num_toright = dm.pseudoinverse_(D2,u3-1e-9) - D3.value()[D3.index()]>0
        how_many_d2_to_the_right = sum(num_toright)
        percent_right = how_many_d2_to_the_right / len(obspga_resam)
        
        
        if percent_left < 0.1 and percent_right < 0.1 :
            Y_opti.append(m[i])
            plt.rcParams["figure.figsize"] = (10,10)
            plt.plot(X,Y,color='black')
            plt.plot(m,n)
            plt.fill_between(X, y_lo,y_hi,color='lightgray')
            plt.ylabel('probability')
            plt.xlabel('PGA')

    am.plot(optimum_log_boore,obspga_resam)
    am.plot(site_out,obspga_resam)

    plt.show()
    print (len(Y_opti))


    








