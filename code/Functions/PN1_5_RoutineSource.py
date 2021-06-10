
# coding: utf-8
# PN1.5 by Edom Moges @ ESDL translated from the matlab version by Ruddell @ NAU
# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import glob
np.random.seed(50)
import sys
import time
import random

from joblib import Parallel, delayed

# np.ravel()
# np.flatten()


# #### GetUniformBinEdges

# In[2]:


def GetUniformBinEdges(sampleMat,nBinMat,pctlRange,NoDataCode):
    #nBinMat = nBinMat.astype(int)
    nSignals=int(np.shape(sampleMat)[1])
    binMax=max(nBinMat).astype(int)
    binEdges=np.ones([nSignals,int(binMax)])*np.nan
    minEdge=np.ones([nSignals,1])*np.nan
    maxEdge=np.ones([nSignals,1])*np.nan
    
    binEdges[1,5]=5
    #print(binEdges[1,5])
    #make nodata entries NaN, because min and max function ignores them...
    sampleMat[sampleMat == NoDataCode] = np.nan
    
    # compute the bin edges using fractions of the min and max
    
    for s in np.arange(nSignals):
        # Use uniform bin width
        minEdge[s]=np.percentile(sampleMat[:,s],pctlRange[0])
        maxEdge[s]=np.percentile(sampleMat[:,s],pctlRange[1]) 
        E = np.linspace(minEdge[s],maxEdge[s],int(nBinMat[s]+1))  # % all edges, including start
                #print(s,int(nBinMat[s]),E[1:])
                #print(binEdges[s,0:int(nBinMat[s])])
        #print(s)
        binEdges[s,0:int(nBinMat[s])]= np.ravel(E[1:]) # it indicates the max edges for each bin [, max1] , E[1:].reshape(int(nBinMat[s])) 
      
    return  binEdges,minEdge,maxEdge #, E       


# #### classifySignal

# In[3]:

def GetEvenBinEdgesGlobal(nBinVect,minEdgeLocal,maxEdgeLocal):
    nVars = np.shape(maxEdgeLocal)[0]
    minEdgeGlobal = np.nanmin(minEdgeLocal, axis = 1)
    maxEdgeGlobal = np.nanmax(maxEdgeLocal,axis = 1)
    binEdgesGlobal = np.ones([nVars,np.max(nBinVect)])*np.nan
    
    for i in np.arange(nVars):
        E = np.linspace(minEdgeGlobal[i],maxEdgeGlobal[i],int(nBinVect[i]+1))# all edges, including start
        binEdgesGlobal[i,0:int(nBinVect[i])] = E[1:].reshape(int(nBinVect[i]))
    
    return binEdgesGlobal,minEdgeGlobal,maxEdgeGlobal



def classifySignal(sampleMat,binEdges,nBinMat,NoDataCode): 
   
    # zero is not part of a class. Consistent to the matlab version.
    
    nData,nSignals = np.shape(sampleMat)
    classifiedMat = np.ones([nData,nSignals])*np.nan

    sampleMat[sampleMat == NoDataCode] = np.nan
    
    for s in np.arange(nSignals): # variables
        smat = copy.deepcopy(sampleMat[:,s])
        cmat = nBinMat[s]*np.ones([np.shape(smat)[0]])
        #print(smat)
        
        for e in np.arange(nBinMat[s]): # bins
            # Find data in the variable falling within this bin
            
            ii = np.argwhere(smat <= binEdges[s.astype(int),e.astype(int)])
            
            #print(s,e,binEdges[s,e],(np.isnan(smat)).any(),ii.shape[0])
            
            cmat[ii] = e+1  #  Assign classification. The same as matlab starting from 1.
            
            smat[ii] = np.nan # remove those that were classified from further consideration
            
            # If this is the last bin (greatest values), assign values greater
            # than this bin edge to this bin
            
            if e == nBinMat[s]:
                ii = np.argwhere(smat > binEdges[s.astype(int),e.astype(int)])
                cmat[ii] = e+1  # Assign classification
                smat[ii] = np.nan # remove those that were classified from further consideration
        
        classifiedMat[:,s] = cmat 
        
        
    classifiedMat[np.isnan(sampleMat)] = np.nan
    nCounts = sum(sum(~np.isnan(classifiedMat[:]))) # column sum will be followed by row sum.
    nCounts = nCounts/nSignals
            
    classifiedMat[np.isnan(classifiedMat)] = NoDataCode        
            
    return classifiedMat,nCounts    
        


# #### getCountMat

# In[4]:


# Takes a classified/binned data and generates a 3D count of the bin for 3D probability computation.


def getCountMat(tupleMat,nBinMat,sX,sY,NoDataCode): # Faster version
    
    dim1Edge = np.r_[0,np.arange(nBinMat[sX])+1.5]
    dim2Edge = np.r_[0,np.arange(nBinMat[sY])+1.5]
    dim3Edge = np.r_[0,np.arange(nBinMat[sY])+1.5]
    
    tupleMat = tupleMat.astype(float)
    tupleMat[tupleMat == NoDataCode] = np.nan
    tupleMat = np.delete(tupleMat, np.argwhere(np.isnan(np.sum(tupleMat,1))==True), axis=0)#delete nan
    
    #print(tupleMat)
    nCounts = np.shape(tupleMat)[0]
    
    C1,cedge = np.histogramdd(tupleMat,bins=(dim1Edge,dim2Edge,dim3Edge))
    C = np.moveaxis(C1, 2, 0)

   
    return C.astype(int), nCounts 



# #### GetShannonBits

# In[5]:


# Takes the 3D counts and convert it to marginal and joint probabilities to generate entropy, MI and T.
def GetShannonBits(C,nCounts):
    # Get Marginal and Joint Shannon Entropies in terms of bits (log 2), given probability matrices
    eps = sys.float_info.epsilon
    
    #print(C.shape,nCounts)
    
    pXtYwYf=(C+eps)/nCounts
    #print(pXtYwYf.shape)
    pp = np.sum(pXtYwYf,1)
    #print(pp.shape)
    
    #NB., numpy goes by depth, row, colum while matlab is row, col, depth.
    pXt = np.sum(np.sum(pXtYwYf,2),0) # Marginal PDFs. 
    pYw = np.sum(np.sum(pXtYwYf,1),0)
    pYf = np.sum(np.sum(pXtYwYf,1),1)
    
    pXtYw = np.sum(pXtYwYf,0) # Joint PDFs
    pXtYf = np.sum(pXtYwYf,2)
    pYwYf = np.sum(pXtYwYf,1)
    
    HXt = -np.sum(pXt*np.log2(pXt)) # Shannon Entropies
    HYw = -np.sum(pYw*np.log2(pYw)) 
    HYf = -np.sum(pYf*np.log2(pYf))
    
    
    HXtYw = -np.sum(np.sum(pXtYw*np.log2(pXtYw))) # Joint Shannon Entropies
    HXtYf = -np.sum(np.sum(pXtYf*np.log2(pXtYf)))
    HYwYf = -np.sum(np.sum(pYwYf*np.log2(pYwYf)))
    
    HXtYwYf = -np.sum(np.sum(np.sum(pXtYwYf*np.log2(pXtYwYf)))) #  Triple Shannon Entropy
    
    return HXt,HYw,HYf,HXtYw,HXtYf,HYwYf,HXtYwYf


# #### ShannonBitsWrapper

# In[6]:


def ShannonBitsWrapper(classifiedData, lag, nTuples, nBinMat, lagRange, nYw, NoDataCode):
    
    
    # lagVect - sequence of lags eg. 0:36. Include 0 at first.
    # lag - lagging length. Read from lagVect[i]
    # lagRange - min and max of lagVect
    # nYw = 1. fixed to self prediction optimality.
    # nTuples = nData (nrow) + lagRange(0) - max([lagRange(1), nYw])-1 Length of tuple data where TE is computed.
    
    nSignals = np.shape(classifiedData)[1]
    
    #np.place(classifiedData, classifiedData == NoDataCode, np.nan)
    

    classifiedData[classifiedData == NoDataCode] = np.nan

    HXt             = np.ones( [nSignals, nSignals] )*np.nan
    HYw             = np.ones( [nSignals, nSignals] )*np.nan
    HYf             = np.ones( [nSignals, nSignals] )*np.nan
    HXtYw           = np.ones( [nSignals, nSignals] )*np.nan
    HXtYf           = np.ones( [nSignals, nSignals] )*np.nan
    HYwYf           = np.ones( [nSignals, nSignals] )*np.nan
    HXtYwYf         = np.ones( [nSignals, nSignals] )*np.nan
    I               = np.ones( [nSignals, nSignals] )*np.nan
    T               = np.ones( [nSignals, nSignals] )*np.nan
    nCounts         = np.ones( [nSignals, nSignals] )*np.nan
    
    
    for sX in np.arange(nSignals):
        for sY in np.arange(nSignals):
            
            #print(sX,sY)
            # CONSTRUCT THREE-COLUMN MATRIX WITH COLUMNS TIME-SHIFTED
            
            XtSTART = np.max([lagRange[1], nYw]) - lag  # where to start x
            YwSTART = np.max([lagRange[1], nYw]) - nYw  # where to start conditioning y
            YfSTART = np.max([lagRange[1], nYw])        # where to start predictive y
            
            #print(XtSTART,YwSTART, YfSTART)
            
            tupleMat= np.nan*np.ones([nTuples,3])
            
            # wehre to end the data. 
            # Key is all column should have the same length but start at different point depending on lag.
            # note change -1 to -2. But, so far it is kept as -1. End of lagged data.
            tupleMat[:,0]=classifiedData[XtSTART:(XtSTART+nTuples),sX]        #Leading Node Xt (lag tau earlier than present)
            tupleMat[:,1]=classifiedData[YwSTART:(YwSTART+nTuples),sY]        #Led Node Yw (present time)
            tupleMat[:,2]=classifiedData[YfSTART:(YfSTART+nTuples),sY]        #Led Node Yf (one timestep in future)

            # CHECK TO ENSURE TUPLEMAT HAS AT LEAST ONE COMPLETE ROW OF DATA
            
            if np.sum(np.sum(np.isnan(tupleMat),1) > 0) == nTuples:
                print('Warning: no data in tupleMat, skipping sX = ', str(1), ', sY = ', str(2), ', lag = ', str(3))
                
            # CALCULATE ENTROPIES FROM TUPLEMAT
            C, nCounts[sX,sY] = getCountMat( tupleMat, nBinMat, sX, sY, np.nan)
            HXt[sX,sY],HYw[sX,sY],HYf[sX,sY],HXtYw[sX,sY],HXtYf[sX,sY],HYwYf[sX,sY],HXtYwYf[sX,sY] =                                                                                     GetShannonBits( C, nCounts[sX,sY] )
            I[sX,sY] = HXt[sX,sY] + HYf[sX,sY] - HXtYf[sX,sY]
            T[sX,sY] = HXtYw[sX,sY] + HYwYf[sX,sY] - HYw[sX,sY] - HXtYwYf[sX,sY]
            
            
    return HXt, HYw, HYf, HXtYw, HXtYf, HYwYf, HXtYwYf, I, T, nCounts


    
    


# #### entropyFunction

# In[7]:


def entropyFunction(classifiedData,lagVect,nBinMat,NoDataCode,parallelWorkers):
    nData,nSignals = np.shape(classifiedData)
    nLags = np.shape(lagVect)[0]    #numebr of lags [# lagVect the lag sequence with 0 at the first index]
    lagRange = [min(lagVect), max(lagVect)] # 0 must be included somewhere in lagVect
    nYw = 1    # the number of data points signifiying the previous history of Y. Hard-coded as 1 point previous for now, but code structured to make this an option in the future
    nTuples = nData + lagRange[0]- max(lagRange[1], nYw) - 1
    
    # INITIALIZE PARALLEL OUTPUTS OF THE SHANNON BIT FUNCTION
    HXt             = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HYw             = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HYf             = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HXtYw           = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HXtYf           = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HYwYf           = np.ones( [nLags, nSignals, nSignals] )*np.nan
    HXtYwYf         = np.ones( [nLags, nSignals, nSignals] )*np.nan
    I               = np.ones( [nLags, nSignals, nSignals] )*np.nan
    T               = np.ones( [nLags, nSignals, nSignals] )*np.nan
    nCounts         = np.ones( [nLags, nSignals, nSignals] )*np.nan
    
    p = Parallel(n_jobs=parallelWorkers,max_nbytes=50e6)(delayed(ShannonBitsWrapper)(classifiedData,lagVect[i], nTuples, nBinMat, lagRange, nYw, NoDataCode) for i in np.arange(nLags)) #'50M’
    
    #ArrayP = np.asarray(p)
    for i in range(nLags):
        HXt[i,:,:]             = p[i][0]
        HYw[i,:,:]             = p[i][1]
        HYf[i,:,:]             = p[i][2]
        HXtYw[i,:,:]           = p[i][3]
        HXtYf[i,:,:]           = p[i][4]
        HYwYf[i,:,:]           = p[i][5]
        HXtYwYf[i,:,:]         = p[i][6]
        I[i,:,:]               = p[i][7]
        T[i,:,:]               = p[i][8]
        nCounts[i,:,:]         = p[i][9]
    
    E = {'HXt': HXt, 'HYw': HYw, 'HYf' : HYf, 'HXtYw': HXtYw, 'HXtYf':HXtYf, 'HYwYf': HYwYf ,'HXtYwYf':HXtYwYf, 
         'I':I, 'T':T, 'nCounts': nCounts}
    return E
    


# #### createSurrogates

# In[8]:


def createSurrogates(opts,Data,nsur): # Fourier not implimented yet
    
    Surrogates = np.ones([nsur,np.shape(Data)[0],np.shape(Data)[1]])*np.nan
    
    
    if opts['SurrogateMethod'] == 2:
        # Randomly shuffle data (leaving NaNs where they are)
        
        for i in np.arange(np.shape(Data)[1]): # variables
            ni = ~np.isnan(Data[:,i])          # number of data
            for ti in np.arange(nsur):         # number of surrogates
                
                Surrogates[ti,ni,i] = np.ravel(random.sample(list(Data[ni,i]),np.sum(ni)))
    
    # Create Iterated Amplitude Adjusted Fourier Transform surrogates
    elif opts['SurrogateMethod'] == 3:
        print('Iterated Amplitude Adjusted Fourier Transform surrogates not yet implimented')

    return Surrogates
    
    


# #### removePeriodicMean

# In[9]:


def removePeriodicMean(signalMat,period,sliderWidth,NoDataCode):
    #signalMat is an n-column matrix where n variables are represented as a
    #timeseries- columns are variables, rows are timeseries records

    #period is the length of the repeating pattern in the data- if using
    #FluxNet data with 30 min resolution, the period is 48 or one day

    # sliderWidth is the number of periods to base the anomaly on - for a five
    # day moving anomaly, set sliderWidth to 5.

    # NoDataCode is a data value that will be ignored. 
    
    signalMat[signalMat == NoDataCode] = np.nan 
    
    #Initialize
    sz = np.shape(signalMat)
    aveMat = np.ones(sz)*np.nan

    # Compute moving-periodic average
    
    for i in range(sz[0]):
        
           
    
        # Get averaging indices
        ai = np.arange(i-period*np.floor(sliderWidth/2),i+period*np.floor(sliderWidth/2)+1,period)+1
        #print(i, ai)
        
        # Handle edges
        if np.min(ai) < 1:
            ai = ai + np.floor(1-np.min(ai/period))*period
        elif max(ai) > sz[0]:
            ai = ai-(np.ceil((np.max(ai)-sz[0])/period)*period)
        
        # Average
        ai = ai.astype(int) - 1 # -1 because of 0 than 1 in python
        #print(i,ai)
        aveMat[i,:] = np.nanmean(signalMat[ai,:],0)
        
    # Compute anomaly
    anomalyMat = signalMat-aveMat
    anomalyMat[np.isnan(anomalyMat)] = NoDataCode
    
    return anomalyMat
              


# #### preProcess

# In[10]:


def preProcess(opts,Data): # Wavelet not implimented yet
    # Turn NoDataCode into NaN
    Data[Data == opts['NoDataCode']] = np.nan

    # Trim rows where there is missing data
    
    if opts['trimTheData']:
        Data[np.sum(np.isnan(Data),1) > 0,:] = np.nan
    # Apply transformation
    
    if opts['transformation'] == 1:
        # Apply anomaly filter 
        Data = removePeriodicMean(Data,opts['anomalyPeriodInData'],opts['anomalyMovingAveragePeriodNumber'],np.nan);

    elif opts['transformation'] == 2:
        # Apply wavelet transform
        print('wavelet transform not yet implimented')
        
    return Data
             


# #### intializeOutput

# In[11]:


def intializeOutput(nDataFiles,nVars,opts):
    nBins = np.transpose(opts['nBins']) # Make column vector
    nLags = np.shape(opts['lagVect'])[0]
    
    
    # Determine size of 3rd dimension for statistical tests depending on 
    # whether we test each lag
    if opts['SurrogateMethod'] > 0:
        if opts['SurrogateTestEachLag']:
            nSLags = nLags
        else:
            nSLags = 1
    else:
        nSLags = 1
    
    # INITIALIZE PREPROCESSING QUANTITIES
    R = {'nRawData'  :         np.zeros([nDataFiles,1]),
        'nVars'      :         np.zeros([nDataFiles,1]),
        'varNames'   :         opts['varNames'],
        'varSymbols' :         opts['varSymbols'],
        'varUnits'   :         opts['varUnits']}
    
    #print(np.shape(nBins),nVars)
    
    if np.shape(nBins)[0] < nVars:
        R['nBinVect']  = (np.ones([nVars,1])*nBins).astype(int)
    else:
        R['nBinVect']  = (nBins).astype(int)
    
    R['nClassified']             = np.ones([nDataFiles,1])*np.nan
    R['binEdgesLocal']           = np.ones([nDataFiles, nVars,np.max(nBins)])*np.nan # 3D data switched dimension
    R['minEdgeLocal']            = np.ones([nVars,nDataFiles])*np.nan
    R['maxEdgeLocal']            = np.ones([nVars,nDataFiles])*np.nan
    R['minSurrEdgeLocal']        = np.ones([nVars,nDataFiles])*np.nan
    R['maxSurrEdgeLocal']        = np.ones([nVars,nDataFiles])*np.nan
    R['LocalVarAvg']             = np.ones([nVars,nDataFiles])*np.nan
    R['LocalVarCnt']             = np.ones([nVars,nDataFiles])*np.nan
    R['binEdgesGlobal']          = np.ones([nVars,np.max(nBins)])*np.nan
    R['minEdgeGlobal']           = np.ones([nVars,1])*np.nan
    R['maxEdgeGlobal']           = np.ones([nVars,1])*np.nan
    R['binSurrEdgesGlobal']      = np.ones([nVars,np.max(nBins)])*np.nan
    R['minSurrEdgeGlobal']       = np.ones([nVars,1])*np.nan
    R['maxSurrEdgeGlobal']       = np.ones([nVars,1])*np.nan
    R['GlobalVarAvg']            = np.ones([nVars,1])*np.nan
    
    # INITIALIZE OUTPUT QUANTITIES FROM THE ENTROPYFUNCTION
    R['lagVect']                = opts['lagVect']
    R['HXt']                    = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan # 4D
    R['HYw']                    = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['HYf']                    = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['HXtYw']                  = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['HXtYf']                  = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['HYwYf']                  = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['HXtYwYf']                = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['SigThreshT']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['SigThreshI']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['meanShuffT']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['sigmaShuffT']            = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['meanShuffI']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['sigmaShuffI']            = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['nCounts']                = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['I']                      = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['T']                      = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['IR']                     = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['TR']                      = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['SigThreshTR']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['SigThreshIR']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['meanShuffTR']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['sigmaShuffTR']            = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['meanShuffIR']             = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['sigmaShuffIR']            = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['Tplus']                   = np.ones([nDataFiles,nVars,nLags])*np.nan
    R['Tminus']                  = np.ones([nDataFiles,nVars,nLags])*np.nan
    R['Tnet']                    = np.ones([nDataFiles,nVars,nLags])*np.nan
    R['TnetBinary']              = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['InormByDist']             = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['TnormByDist']             = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['SigThreshInormByDist']    = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['SigThreshTnormByDist']    = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    
    R['Ic']                 = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['Tc']                     = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['TvsIzero']               = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['SigThreshTvsIzero']      = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['IvsIzero']               = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['SigThreshIvsIzero']      = np.ones([nDataFiles,nSLags,nVars,nVars])*np.nan
    R['Abinary']                = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['Awtd']                   = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['AwtdCut']                = np.ones([nDataFiles,nLags,nVars,nVars])*np.nan
    R['charLagFirstPeak']       = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['TcharLagFirstPeak']      = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['charLagMaxPeak']         = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['TcharLagMaxPeak']        = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['TvsIzerocharLagMaxPeak'] = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['nSigLags']               = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['FirstSigLag']            = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['LastSigLag']             = np.ones([nDataFiles, nVars,nVars])*np.nan
    R['HXtNormByDist']          = np.ones([nDataFiles, nLags,nVars,nVars])*np.nan
    
    return R

