# %%

# coding: utf-8
# PN1.5 by Edom Moges @ ESDL translated from the matlab version by Ruddell @ NAU
# %%


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
from datetime import datetime
import os
import math
import pickle
from matplotlib import cm

from joblib import Parallel, delayed


# %%


# Local function 1
from PN1_5_RoutineSource import *


# %%


import xarray as xr
import numcodecs
import zarr


# %%


def optsCheck(opts,nVars):
    print('optsCheck Not yet implimented')
    return opts


# %%


def logwrite(msg,vout):
    if vout:
        print(msg)


# %%


def DoProduction(T):
    
    nFiles, nTaus, nVars,xx = np.shape(T)
    
    Tplus = np.zeros([nFiles,nVars,nTaus])
    Tminus = np.zeros([nFiles,nVars,nTaus])
    TnetBinary = np.ones([nFiles,nTaus,nVars,nVars])*np.nan
    
    for f in np.arange(nFiles):
        for i in np.arange(nVars):
            for j in np.arange(nVars):
                for t in np.arange(nTaus):
                    if ~np.isnan(T[f,t,i,j]):
                        Tplus[f,i,t]=Tplus[f,i,t] + T[f,t,i,j]
                        Tminus[f,j,t]=Tminus[f,j,t] + T[f,t,i,j]
                        
                        
    
    Tnet=Tplus-Tminus
    for f in np.arange(nFiles):
        for t in np.arange(nTaus):
            SQRmat = T[f,t,:,:]
            TnetBinary[f,t,:,:] = SQRmat- np.transpose(SQRmat)
    
   
    
    return Tplus,Tminus,Tnet,TnetBinary


# %%


def NormTheStats(nBinVect,I,T,SigThreshI,SigThreshT,meanShuffI,sigmaShuffI,meanShuffT,sigmaShuffT,HXt,HYf,lagVect):
    
    # Get size
    nFiles,nLags, nSignals,xx  = np.shape(I)
    nSLags = np.shape(SigThreshT)[1]
    
    
    # Initialize
    InormByDist = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    TnormByDist = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    SigThreshInormByDist = np.ones([nFiles,nSLags,nSignals,nSignals])*np.nan
    SigThreshTnormByDist = np.ones([nFiles,nSLags,nSignals,nSignals])*np.nan
    Ic = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    Tc = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    TvsIzero = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    SigThreshTvsIzero = np.ones([nFiles,nSLags,nSignals,nSignals])*np.nan
    RelI = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    RelT = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    HXtNormByDist = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    IvsIzero = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    SigThreshIvsIzero = np.ones([nFiles,nSLags,nSignals,nSignals])*np.nan
    
    # index of lag = 0
    l0i = np.argwhere(lagVect == 0)[0]
    
    for f in np.arange(nFiles):
        for i in np.arange(nSignals):
            for j in np.arange(nSignals):
                for t in np.arange(nLags):
                    
                    # Statistical sig threshold available for 1 or all lags?
                    if nSLags == 1:
                        tS = 0
                    else:
                        tS = t
                        
                    n = np.min(np.r_[nBinVect[i],nBinVect[j]])
                    InormByDist[f,t,i,j] = I[f,t,i,j]/np.log2(n)
                    TnormByDist[f,t,i,j] = T[f,t,i,j]/np.log2(n)
                    
                    Ic[f,t,i,j] = 0.5*(1+math.erf((I[f,t,i,j] - meanShuffI[f,tS,i,j])/ (np.sqrt(2)*sigmaShuffI[f,tS,i,j]) ))
                    Tc[f,t,i,j] = 0.5*(1+math.erf((T[f,t,i,j]-meanShuffT[f,tS,i,j])/(np.sqrt(2)*sigmaShuffT[f,tS,i,j])))

                    SigThreshInormByDist[f,tS,i,j] = SigThreshI[f,tS,i,j]/np.log2(n)
                    SigThreshTnormByDist[f,tS,i,j] = SigThreshT[f,tS,i,j]/np.log2(n)
                    TvsIzero[f,t,i,j] = T[f,t,i,j]/I[f,l0i,i,j]
                    SigThreshTvsIzero[f,tS,i,j] = SigThreshT[f,tS,i,j]/I[f,l0i,i,j]
                    

                    RelI[f,t,i,j] = I[f,t,i,j]/HYf[f,t,i,j]
                    RelT[f,t,i,j]=T[f,t,i,j]/HYf[f,t,i,j]
                    HXtNormByDist[f,t,i,j]=HXt[f,t,i,j]/np.log2(n)
                    IvsIzero[f,t,i,j]=I[f,t,i,j]/I[f,l0i,i,j]
                    
                    SigThreshIvsIzero[f,tS,i,j]=SigThreshI[f,tS,i,j]/I[f,l0i,i,j]

    
    
    return InormByDist,TnormByDist,SigThreshInormByDist,SigThreshTnormByDist,Ic,Tc,TvsIzero,SigThreshTvsIzero,RelI,RelT,HXtNormByDist,IvsIzero,SigThreshIvsIzero


# %%


def AdjMatrices(T,SigThreshT,TvsIzero,lagVect):
    # COMPUTES AN ADJACENCY MATRIX A
    # COMPUTES THE CHARACTERISTIC LAG WHICH IS THE FIRST SIGNIFICANT LAG
    # TAKES THE TRANSFER INFORMATION MATRIX AND THE SIGNIFICANCE THRESHOLDS
    # NOTE - This code considers ONLY POSITIVE LAGS in the computation of these matrices
    
    nFiles,nLags, nSignals,xx = np.shape(T)
    nSLags = np.shape(SigThreshT)[1]
    
    
    Abinary = np.zeros([nFiles,nLags,nSignals,nSignals])
    Awtd = np.ones([nFiles,nLags,nSignals,nSignals])*np.nan
    AwtdCut = np.zeros([nFiles,nLags,nSignals,nSignals])
    charLagFirstPeak = np.zeros([nFiles,nSignals,nSignals])
    TcharLagFirstPeak = np.zeros([nFiles,nSignals,nSignals])
    charLagFirst = np.zeros([nFiles,nSignals,nSignals])
    TcharLagFirst = np.zeros([nFiles,nSignals,nSignals])
    charLagMaxPeak = np.zeros([nFiles,nSignals,nSignals])
    TcharLagMaxPeak = np.zeros([nFiles,nSignals,nSignals])
    TvsIzerocharLagMaxPeak = np.zeros([nFiles,nSignals,nSignals])
    nSigLags = np.zeros([nFiles,nSignals,nSignals])
    FirstSigLag = np.ones([nFiles,nSignals,nSignals])*np.nan
    LastSigLag = np.ones([nFiles,nSignals,nSignals])*np.nan
    
    # index of lag = 0
    l0i = np.argwhere(lagVect == 0)[0]
    
    for f in np.arange(nFiles):
        for sX in np.arange(nSignals):
            for sY in np.arange(nSignals):
                
                FirstPeakFlag=0
                FirstSigFlag=0

                Awtd=T
                
                # check 0 lag
                lag=l0i
                
                # Statistical sig threshold available for 1 or all lags?
                if nSLags == 1:
                    lagS = 0
                else:
                    lagS = lag
                
                if T[f,lag,sX,sY] > SigThreshT[f,lagS,sX,sY]:
                    Abinary[f,lag,sX,sY] = 1
                    AwtdCut[f,lag,sX,sY] = T[f,lag,sX,sY]
                    LastSigLag[f,sX,sY] = lag
                    nSigLags[f,sX,sY] = nSigLags[f,sX,sY]+1
                    charLagMaxPeak[f,sX,sY] = lag
                    TcharLagMaxPeak[f,sX,sY] = T[f,lag,sX,sY]
                    TvsIzerocharLagMaxPeak[f,sX,sY] = TvsIzero[f,lag,sX,sY]
                    FirstSigFlag = 1
                    
                    if nLags > 1:
                        if T[f,lag,sX,sY] > T[f,lag+1,sX,sY]:
                            charLagFirstPeak[f,sX,sY] = lag
                            TcharLagFirstPeak[f,sX,sY] = T[f,lag,sX,sY]
                            FirstPeakFlag = 1

                    else:
                        charLagFirstPeak[f,sX,sY] = lag
                        TcharLagFirstPeak[f,sX,sY] = T[f,lag,sX,sY]
                        FirstPeakFlag = 1

                # check the other lag
                if nLags > 1:
                    for lag in np.arange(l0i+1,nLags-1):
                      # Statistical sig threshold available for 1 or all lags?
                        if nSLags == 1:
                            lagS = 0
                        else:
                            lagS = lag
                            
                        if T[f,lag,sX,sY] > SigThreshT[f,lagS,sX,sY]:
                            Abinary[f,lag,sX,sY] = 1
                            AwtdCut[f,lag,sX,sY] = T[f,lag,sX,sY]
                            LastSigLag[f,sX,sY] = lag
                            nSigLags[f,sX,sY] = nSigLags[f,sX,sY]+1
                            
                            if FirstSigFlag == 0:
                                FirstSigLag[f,sX,sY] = lag
                                FirstSigFlag = 1
                                
                            #print( lag,(FirstPeakFlag == 0) & (T[f,lag,sX,sY] > T[f,lag-1, sX,sY]) & (T[f,lag,sX,sY] > T[f,lag+1,sX,sY]))
                            if (FirstPeakFlag == 0) & (T[f,lag,sX,sY] > T[f,lag-1, sX,sY]) & (T[f,lag,sX,sY] > T[f,lag+1,sX,sY]):
                                charLagFirstPeak[f,sX,sY] = lag
                                TcharLagFirstPeak[f,sX,sY] = T[f,lag,sX,sY]
                                FirstPeakFlag = 1
                                
                            if T[f,lag,sX,sY] > TcharLagMaxPeak[f,sX,sY]:
                                charLagMaxPeak[f,sX,sY] = lag
                                TcharLagMaxPeak[f,sX,sY] = T[f,lag,sX,sY]
                                TvsIzerocharLagMaxPeak[f,sX,sY] = TvsIzero[f,lag,sX,sY]    

                            
                    #check the last lag
                    lag=nLags-1
                    # Statistical sig threshold available for 1 or all lags?
                    if nSLags == 1:
                        lagS = 0
                    else:
                        lagS = lag
                        
                    
                    
                    if T[f,lag,sX,sY] > SigThreshT[f,lagS,sX,sY]:
                        Abinary[f,lag,sX,sY] = 1
                        AwtdCut[f,lag,sX,sY] = T[f,lag,sX,sY]
                        LastSigLag[f,sX,sY] = lag 
                        nSigLags[f,sX,sY] = nSigLags[f,sX,sY]+1
                        if FirstSigFlag == 0:
                            FirstSigLag[f,sX,sY]=lag
                            FirstSigFlag = 1
                        if (FirstPeakFlag == 0) & (T[f,lag,sX,sY] > T[f,lag-1,sX,sY]):
                            charLagFirst[f,sX,sY] = lag
                            TcharLagFirst[f,sX,sY] = T[f,lag,sX,sY]
                            FirstPeakFlag = 1
                            
                        if T[f,lag,sX,sY] > TcharLagMaxPeak[f,sX,sY]:
                            charLagMaxPeak[f,sX,sY] = lag
                            TcharLagMaxPeak[f,sX,sY]=T[f,lag,sX,sY]
                            TvsIzerocharLagMaxPeak[f,sX,sY] = TvsIzero[f,lag,sX,sY]                       
                        
                        
                        
                        
    
    
    return Abinary,Awtd,AwtdCut,charLagFirstPeak,TcharLagFirstPeak,charLagMaxPeak,TcharLagMaxPeak,TvsIzerocharLagMaxPeak,nSigLags,FirstSigLag,LastSigLag
    
    


# %%



def ProcessNetwork(opts):
    
    global processLog
    processLog = []
    
    now = datetime.now() # current date and time
    clk = now.strftime("%m/%d/%Y, %H:%M:%S")
    plogName = 'processLog_' + clk
    
    logwrite(['Processing session for: ', clk],0)
    logwrite('Checking options & parameters...',1)
    
    opts = optsCheck(opts,[])

    # Intitialize
    nDataFiles = np.shape(opts['files'])[0] #
    badfile = 0 # skipped files
    nLags = np.shape(opts['lagVect'])[0]
    
    ## Main program
    logwrite('*** Beginning processing ***',1)
    
    for fi in np.arange(nDataFiles):
        
         # Load file (can be matlab format or ascii
        logwrite(['--Processing file # ', fi ,':' , opts['files'][fi], '...'],1)
        
        try :
            name,ext = os.path.splitext(opts['files'][fi])
            Data = np.loadtxt(opts['files'][fi]) #,delimiter=',')   
        except :
            logwrite('Error: Problem loading file. Skipping...',1)
            badfile = badfile+1
            
        nData,nVars = np.shape(Data)
        
        # Retain data as loaded
        rawData = copy.deepcopy(Data)

        # Check or create variable names, symbols, and units
        opts = optsCheck(opts,nVars)

        if np.shape(opts['varNames'])[0] != nVars:
                logwrite(['Unable to process file # ', fi,  ': ', opts['files'][fi], '. # of data columns inconsistent with # of varNames.'],1)
                badfile = badfile+1


        #------------------------
        # step 0
        # Run the preprocessing options, including data trimming, anomaly
        # filter or wavelet transform

        logwrite('Preprocessing data.',1)

        if opts['trimTheData']:
            logwrite('Trimming rows with missing data',1)



        if opts['transformation'] == 1:

            logwrite(['Applying anomaly filter over ', opts['anomalyMovingAveragePeriodNumber'], ' periods of ', opts['anomalyPeriodInData'], ' time steps per period.'],1);

        elif opts['transformation'] == 2:
            if opts['waveDorS'] == 1:
                DorS = 'detail'
            else:
                DorS = 'approximation'

            logwrite(['Applying MODWT wavelet filter at ', DorS, ' scale(s) [', opts.waveN, '] using ', opts['waveName'], ' mother wavelet.'],1)

        #--------------------------
        # step 1 preProcess
        
        Data = preProcess(opts,Data)
        #print(Data[0:5,:])


        #--------------------------
        # step 2 intializeOutput
        # Intialize if first file
        if fi-badfile == 0:
            R = intializeOutput(nDataFiles,nVars,opts)

        # Collect summary stats on variables for archiving and later
        #classification
        logwrite('Computing local statistics',1)
        R['nRawData'][fi] = nData
        R['nVars'][fi] = nVars
        
        #print(Data[0:5,])
        #--------------------------
        # step 3 GetUniformBinEdges
        GBE = GetUniformBinEdges(Data,R['nBinVect'],opts['binPctlRange'],np.nan)
        R['binEdgesLocal'][fi,:,:] = GBE[0]
        R['minEdgeLocal'][:,fi] = GBE[1].reshape(nVars)
        R['maxEdgeLocal'][:,fi] = GBE[2].reshape(nVars)
        R['LocalVarAvg'][:,fi] = np.transpose(np.nanmean(Data,0))
        R['LocalVarCnt'][:,fi] = np.transpose(np.sum(~np.isnan(Data)))
        
        
        # If we are doing local binning or data is already binned, we can 
        # go straight into classification and/or entropy calculations. 
        # Otherwise, we are just saving stats for now
        
        # print(R['binEdgesLocal'])
        if (np.argwhere(np.array([0, 1])==opts['binType']).size != 0): # No binning required if binType = 0
            
            #--------------------------
            # step 4 classifySignal
            # Classify the data with local binning
            if opts['binType'] == 1:
                logwrite(['Classifying with [', opts['nBins'], '] local bins over [', opts['binPctlRange'], '] percentile range.'],1)
                Data,R['nClassified'][fi]=classifySignal(Data,R['binEdgesLocal'][fi,:,:],R['nBinVect'],np.nan)
            
            #print(Data[0:5,:])
            
           # Save preprocessed Data if we're done processing it
            if opts['savePreProcessed'] & (np.argwhere(np.array([0, 1])==opts['binType']).size != 0): 
                logwrite('Saving preprocessed data',1)
                eval("plogName + ' ' + str(processLog)")
                #np.savez(opts['outDirectory'] + name +  opts['preProcessedSuffix'],Data)
                
                # -----------------------
                # Writing to a zarr file
                nameZarr = opts['outDirectory'] + name + 'fi_' + str(fi) + opts['preProcessedSuffix']
                rrw = np.shape(Data)[0]
                clm = np.shape(Data)[1]
                #print(rrw,clm)
                ds = xr.Dataset({'data': ( ('depth', 'nrow', 'nvar'), Data.reshape([1,rrw,clm]) )},  # data
                                 coords={'depth': np.arange(1,2), # dimesion 1 
                                         'nrow': np.arange(rrw),
                                         'nvar': np.arange(clm)})
                ds.to_zarr(str(nameZarr) + '.zarr','w')                          
                
                

            # Run entropy function
            if opts['doEntropy']:
                logwrite('Running entropy function.',1)

                # Check that data have been classified
                if np.nansum(np.remainder(np.ravel(Data),1)) != 0:
                    logwrite('ERROR: Some or all input data are not classified. Check options and/or data. Skipping file...',1);


                # ---------------------------
                # step 4 entropyFunction
                #print(R['HYwYf'])
                
                E = entropyFunction(Data,R['lagVect'],R['nBinVect'],np.nan,opts['parallelWorkers'])
                
                #print(E.keys())
                # Assign outputs
                R['HXt'][fi,:,:,:]=E['HXt']
                R['HYw'][fi,:,:,:]=E['HYw']
                R['HYf'][fi,:,:,:]=E['HYf']
                R['HXtYw'][fi,:,:,:]=E['HXtYw']
                R['HXtYf'][fi,:,:,:]=E['HXtYf']
                R['HYwYf'][fi,:,:,:]=E['HYwYf']
                R['HXtYwYf'][fi,:,:,:]=E['HXtYwYf']
                R['nCounts'][fi,:,:,:]=E['nCounts']
                R['I'][fi,:,:,:]=E['I']
                R['T'][fi,:,:,:]=E['T']
        
        
        # ==============================
        # Surrogate computation
        #===============================
        # If we already have surrogates, check we have them
        
        if opts['SurrogateMethod'] == 1:
            print('Surrogates supplied needs loading')
            #---> Load surrugate from zarr
            SavedSurrogates = copy.deepcopy(Surrogates) # read the zarr file as a surrogate at the beigning

        elif opts['savePreProcessed'] == 1 & opts['SurrogateMethod'] > 1:
            # Initialize if we are saving them and don't already have them
            SavedSurrogates = np.ones([opts['nTests'],nData, nVars])*np.nan
            
        # Create and/or process surrogates for statistical significance 
        # testing
        if opts['SurrogateMethod'] > 0 :
            if opts['SurrogateMethod'] == 1:
                logwrite(['Running the same operations on ', opts['nTests'], ' surrogates contained in input file...'],1)
            elif opts['SurrogateMethod'] == 2 :
                logwrite(['Creating and running the same operations on ', opts['nTests'], ' surrogates using random shuffle method.'],1);
            elif opts['SurrogateMethod'] == 3 :
                logwrite(['Creating and running the same operations on ', opts['nTests'],  ' surrogates using IAAFT method (this may take a while).'],1);

            # Initalize surrogate matrix
            if opts['SurrogateTestEachLag'] == 1:
                # Test every lag in surrogates
                SlagVect = R['lagVect']
                nSLags = nLags
                if (np.argwhere(np.array([0, 1])==opts['binType']).size != 0) & opts['doEntropy'] == 1:
                    logwrite('Testing surrogates at each lag (this may take a while).',1);    

            else:
                # Test only the first and last lags (to restrict data range
                # to same as actual data). We will only retain stats for the
                # last lag.
                SlagVect = np.transpose([R['lagVect'][0], R['lagVect'][-1]])
                nSLags = 1
                if (np.argwhere(np.array([0, 1])==opts['binType']).size != 0) & opts['doEntropy'] == 1:
                    logwrite('Testing surrogates at final lag only.',1)


        
            shuffT = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
            shuffI = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
            shuffHYf = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
            
            #-------------------------------
            # step 5 createSurrogates
            for si in np.arange(opts['nTests']):
                #print(si)
                if opts['SurrogateMethod'] == 1:
                    Surrogates = SavedSurrogates[si,:,:]
                elif np.argwhere(np.array([2, 3])==opts['SurrogateMethod']).size != 0:
                    Surrogates = (createSurrogates(opts,rawData,1)).reshape(rawData.shape[0],rawData.shape[1])
                            

                # ---------------------------
                # step 5.1 preProcess
                # Preprocess surrogates same as Data
                
                #print(Surrogates)
                Surrogates = preProcess(opts,Surrogates)
                #print(Surrogates)
                
                # Collect stats on Surrogates
                #-----------------------------
                # step 5.2 GetUniformBinEdges
                SbinEdgesLocal,minEdgeLocal,maxEdgeLocal = GetUniformBinEdges(Surrogates,R['nBinVect'],opts['binPctlRange'],np.nan)
                #print(np.nanmin(np.r_[np.ravel(minEdgeLocal), R['minSurrEdgeLocal'][:,fi]],axis = 0))
                R['minSurrEdgeLocal'][:,fi] = np.nanmin(np.r_[np.ravel(minEdgeLocal), R['minSurrEdgeLocal'][:,fi]],axis = 0)
                R['maxSurrEdgeLocal'][:,fi] = np.nanmax(np.r_[np.ravel(maxEdgeLocal), R['maxSurrEdgeLocal'][:,fi]],axis = 0)
                
                
                # ---------------------------
                # step 5.3 classifySignal
                
                
                if (np.argwhere(np.array([0, 1])==opts['binType']).size != 0):
                    # Classify the data with local binning
                    if opts['binType'] == 1:
                        Surrogates,xx = classifySignal(Surrogates,SbinEdgesLocal,R['nBinVect'],np.nan)
                        
                    # Are we saving the surrogates?
                    if opts['savePreProcessed']:
                        SavedSurrogates[si,:,:] = Surrogates
                                        
                    # -------------------------
                    # step 5.4 entropyFunction on surrogates
                    if opts['doEntropy']:
                        if np.nansum(np.remainder(np.ravel(Surrogates),1)) != 0:
                            logwrite('ERROR: Surrogate data are not classified. Check options and/or input data. Aborting surrogate testing...',1)
                            
                        E = entropyFunction(Surrogates,SlagVect,R['nBinVect'],np.nan,opts['parallelWorkers'])   
                            
                        # Assign outputs specific to surrogate data
                        if opts['SurrogateTestEachLag'] == 1:
                            # All lags tested
                            shuffT[si,:,:,:] = E['T']
                            shuffI[si,:,:,:] = E['I']
                            shuffHYf[si,:,:,:] = E['HYf'] 
                        else:
                            # Just last lag
                            shuffT[si,0,:,:] = E['T'][-1,:,:]
                            shuffI[si,0,:,:] = E['I'][-1,:,:]
                            shuffHYf[si,0,:,:] = E['HYf'][-1,:,:]

                
            # Here  
            # Calculate stats for statistical significance
            R['meanShuffT'][fi,:,:,:] = np.mean(shuffT,axis=0)
            R['sigmaShuffT'][fi,:,:,:] = np.std(shuffT,0)
            R['meanShuffI'][fi,:,:,:] = np.mean(shuffI,0)
            R['sigmaShuffI'][fi,:,:,:] = np.std(shuffI,0)
            R['meanShuffTR'][fi,:,:,:] = np.mean(shuffT/shuffHYf,0)
            R['sigmaShuffTR'][fi,:,:,:] = np.std(shuffT/shuffHYf,0)
            R['meanShuffIR'][fi,:,:,:] = np.mean(shuffI/shuffHYf,0)
            R['sigmaShuffIR'][fi,:,:,:] = np.std(shuffI/shuffHYf,0)
            
            
            # Save preprocessed Surrogates if we're done processing
            if opts['savePreProcessed'] & (np.argwhere(np.array([0, 1])==opts['binType']).size != 0):
                
                logwrite('Saving preprocessed surrogates',1)
                Surrogates = SavedSurrogates
                eval("plogName + ' ' + str(processLog)")
                                
                # -----------------------
                # Writing to a zarr file
                nameZarr = opts['outDirectory'] + name + 'fi_' + str(fi) + opts['preProcessedSuffix']
                dpth = np.shape(Surrogates)[0]
                rrw = np.shape(Surrogates)[1]
                clm = np.shape(Surrogates)[2]
                #print(rrw,clm)
                ds1 = xr.Dataset({'Surrogates': ( ('depth', 'nrow', 'nvar'), Surrogates.reshape([dpth,rrw,clm]) )},  # data
                                 coords={'depth': np.arange(dpth), # dimesion 1 
                                         'nrow': np.arange(rrw),
                                         'nvar': np.arange(clm)})
                ds1.to_zarr(str(nameZarr) + '.zarr', append_dim='depth') # a classified surrogate data is saved
                
    # ==================================================
    # E computation on raw data and surrogates completed 
    # Make sure we processed at least 1 file
    
    if badfile == nDataFiles:
        logwrite('No files were processed. Check processLog',1)
        R = []
    
     # ####################
     # step 6 global statistics

    # Establish global statistics and bins
    logwrite('Computing global statistics',1)
    R['GlobalVarAvg'] = np.nansum(R['LocalVarAvg']*R['LocalVarCnt'],axis=1)/np.nansum(R['LocalVarCnt'],axis=1)
    R['binEdgesGlobal'],R['minEdgeGlobal'],R['maxEdgeGlobal'] = GetEvenBinEdgesGlobal(R['nBinVect'],R['minEdgeLocal'],R['maxEdgeLocal'])# data
    #print(R['nBinVect'],R['minSurrEdgeLocal'],R['maxSurrEdgeLocal'])
    R['binSurrEdgesGlobal'],R['minSurrEdgeGlobal'],R['maxSurrEdgeGlobal']=GetEvenBinEdgesGlobal(R['nBinVect'],R['minSurrEdgeLocal'],R['maxSurrEdgeLocal'])# surrogates

    
    # If we chose the global binning option, we need to run through the data
    # again
    
    if opts['binType'] == 2:
        
        logwrite('*** Processing files again, this time using global binning ***',1)
        
        # Reset Surrogate stats
        R['minSurrEdgeLocal'] = np.ones([nVars,nDataFiles])*np.nan
        R['maxSurrEdgeLocal'] = np.ones([nVars,nDataFiles])*np.nan
        
        for fi in np.arange(nDataFiles):
            logwrite(['--Processing file # ' + str(fi)+ ': ' + opts['files'][fi] + '...'],1)
            
            # Clear previously generated variables
            del rawData, Data, Surrogates
            if 'SavedSurrogates' in locals(): # check if it exists in the list of local variables global()
                print('True')
                del SavedSurrogates
            
            # Load file (can be zarr format or ascii)
            
            try : # if zarr file, edit reading with ext
                name,ext = os.path.splitext(opts['files'][fi])
                Data = np.loadtxt(opts['files'][fi]) #,delimiter=',')   
            except :
                logwrite(['Unable to process file # ' + fi +': '+ opts['files'][fi] +'. Problem loading file.'],1)
                badfile = badfile+1
            
            nData,nVars = np.shape(Data)
            # Retain data as loaded
            rawData = copy.deepcopy(Data)
            
            # Run the preprocessing options, including data trimming, anomaly
            # filter or wavelet transform
            logwrite('Preprocessing data.',1)
            
            if opts['trimTheData']:
                logwrite('Trimming rows with missing data',1)
                
            if opts['transformation'] == 1:
                logwrite(['Applying anomaly filter over ' + str(opts['anomalyMovingAveragePeriodNumber']) + ' periods of ' + str(opts['anomalyPeriodInData']) + ' time steps per period.'],1)
            elif opts['transformation'] == 2:
                if opts['waveDorS'] == 1:
                    DorS = 'detail'
                else:
                    DorS = 'approximation'
                logwrite(['Applying MODWT wavelet filter at ' + str(DorS) +' scale(s) [' + str(opts['waveN']) +'] using ' + opts['waveName'] + ' mother wavelet.'],1)    
                    


            
            #----------------------------
            # step 6.1 preprocess
            Data = preProcess(opts,Data)
            # Classify the data with global binning
            logwrite(['Classifying with [' + str(opts['nBins']) + '] global bins over [' + str(opts['binPctlRange']) + '] percentile range.'],1)

            # ---------------------------
            # step 6.2 classifySignal        
            Data,R['nClassified'][fi]=classifySignal(Data,R['binEdgesGlobal'],R['nBinVect'],np.nan)

            #print(Data[0:5,:])
            
            # Save preprocessed Data
            if opts['savePreProcessed']:
                logwrite('Saving preprocessed data',1)
                              
                eval("plogName + ' ' + str(processLog)")
                                
                # -----------------------
                # Writing to a zarr file
                nameZarr = opts['outDirectory'] + name + 'fi_' + str(fi) + opts['preProcessedSuffix']
                rrw = np.shape(Data)[0]
                clm = np.shape(Data)[1]
                #print(rrw,clm)
                ds = xr.Dataset({'data': ( ('depth', 'nrow', 'nvar'), Data.reshape([1,rrw,clm]) )},  # data
                                 coords={'depth': np.arange(1,2), # dimesion 1 
                                         'nrow': np.arange(rrw),
                                         'nvar': np.arange(clm)})
                ds.to_zarr(str(nameZarr) + '.zarr','w')          
                
                
             #---------------------------
             # step 6.3 entropyFunction 
             # Run entropy function
            if opts['doEntropy']:
                logwrite('Running entropy function.',1);
                E = entropyFunction(Data,R['lagVect'],R['nBinVect'],np.nan,opts['parallelWorkers'])
                #print('EHXt',E['HXt'])

                # Assign outputs
    
                R['HXt'][fi,:,:,:]=E['HXt']
                R['HYw'][fi,:,:,:]=E['HYw']
                R['HYf'][fi,:,:,:]=E['HYf']
                R['HXtYw'][fi,:,:,:]=E['HXtYw']
                R['HXtYf'][fi,:,:,:]=E['HXtYf']
                R['HYwYf'][fi,:,:,:]=E['HYwYf']
                R['HXtYwYf'][fi,:,:,:]=E['HXtYwYf']
                R['nCounts'][fi,:,:,:]=E['nCounts']
                R['I'][fi,:,:,:]=E['I']
                R['T'][fi,:,:,:]=E['T']
                
            # Are we saving the Surrogates? If so, load or initialize matrix
            if opts['SurrogateMethod'] == 1:
                #---> Load surrugate from zarr
                SavedSurrogates = copy.deepcopy(Surrogates)
            elif opts['savePreProcessed'] == 1 & opts['SurrogateMethod'] > 1:
                SavedSurrogates = np.ones([nData, nVars, opts['nTests']])
            
            # Create and/or process surrogates for statistical significance 
            # testing
            if opts['SurrogateMethod'] > 0:
                
                if opts['SurrogateMethod'] == 1:
                    logwrite(['Running the same operations on ', opts['nTests'], ' surrogates contained in input file...'],1)
                elif opts['SurrogateMethod'] == 2 :
                    logwrite(['Creating and running the same operations on ', opts['nTests'], ' surrogates using random shuffle method.'],1)
                elif opts['SurrogateMethod'] == 3 :
                    logwrite(['Creating and running the same operations on ', opts['nTests'],  ' surrogates using IAAFT method (this may take a while).'],1)
                    
                # Initalize surrogate matrix
                if opts['SurrogateTestEachLag'] == 1 & opts['doEntropy'] == 1:
                    # Test every lag in surrogates
                    SlagVect = R['lagVect']
                    nSLags = nLags
                    logwrite('Testing surrogates at each lag (this may take a while).',1)     
                elif opts['doEntropy'] == 1:
                    # Test only the first and last lags (to restrict data range
                    # to same as actual data). We will only retain stats for the
                    # last lag.
                    SlagVect = SlagVect = np.transpose([R['lagVect'][0], R['lagVect'][-1]])
                    nSLags = 1
                    logwrite('Testing surrogates at final lag only.',1)
                    
                shuffT = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
                shuffI = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
                shuffHYf = np.ones([opts['nTests'], nSLags, nVars, nVars])*np.nan
                
                #---------------------
                #step 6.4 createSurrogates            
            
                for si in np.arange(opts['nTests']):
                    
                    if opts['SurrogateMethod'] == 1:
                        
                        Surrogates = SavedSurrogates[si,:,:]
                        
                    elif np.argwhere(np.array([2, 3])==opts['SurrogateMethod']).size != 0:
                        # Create surrogates using method specified
                        Surrogates = createSurrogates(opts,rawData,1).reshape(rawData.shape[0],rawData.shape[1])
                        
                    # -------------------------
                    # step 6.5.1 preProcess
                    # Preprocess surrogates same as Data
                    Surrogates = preProcess(opts,Surrogates)
                    
                    # Collect stats on Surrogates - we're going to rewrite
                    # these to ensure that the global stats we calculated from
                    # different surrogates match the new ones we create here
                    xxx,minEdgeLocal,maxEdgeLocal = GetUniformBinEdges(Surrogates,R['nBinVect'],opts['binPctlRange'],np.nan)
                    R['minSurrEdgeLocal'][:,fi] = np.nanmin(np.r_[np.ravel(minEdgeLocal), R['minSurrEdgeLocal'][:,fi]],axis=0)
                    R['maxSurrEdgeLocal'][:,fi] = np.nanmax(np.r_[np.ravel(maxEdgeLocal), R['maxSurrEdgeLocal'][:,fi]],axis =0)   
      
                    # --------------------------
                    # step 6.5.2 classifySignal
                    # Classify surrogates
                    Surrogates,xx = classifySignal(Surrogates,R['binSurrEdgesGlobal'],R['nBinVect'],np.nan)
                    
                    
                    # Are we saving the surrogates? If so, archive.
                    if opts['savePreProcessed']:
                        SavedSurrogates[si,:,:] = Surrogates
                    
                    # Run entropy function
                    if opts['doEntropy']:
                        # -----------------------
                        # step 6.5.2 entropyFunction       
                        
                        E = entropyFunction(Surrogates,SlagVect,R['nBinVect'],np.nan,opts['parallelWorkers'])
                        # Assign outputs specific to surrogate data
                        if opts['SurrogateTestEachLag'] == 1:
                            # All lags tested
                            shuffT[si,:,:,:] = E['T']
                            shuffI[si,:,:,:] = E['I']
                            shuffHYf[si,:,:,:] = E['HYf']
                        else:
                            # Just last lag                           
                            shuffT[si,0,:,:] = E['T'][-1,:,:]
                            shuffI[si,0,:,:] = E['I'][-1,:,:]
                            shuffHYf[si,0,:,:] = E['HYf'][-1,:,:]
                            
                # Calculate stats for statistical significance
                       
                R['meanShuffT'][fi,:,:,:] = np.mean(shuffT,axis=0)
                R['sigmaShuffT'][fi,:,:,:] = np.std(shuffT,0)
                R['meanShuffI'][fi,:,:,:] = np.mean(shuffI,0)
                R['sigmaShuffI'][fi,:,:,:] = np.std(shuffI,0)
                R['meanShuffTR'][fi,:,:,:] = np.mean(shuffT/shuffHYf,0)
                R['sigmaShuffTR'][fi,:,:,:] = np.std(shuffT/shuffHYf,0)
                R['meanShuffIR'][fi,:,:,:] = np.mean(shuffI/shuffHYf,0)
                R['sigmaShuffIR'][fi,:,:,:] = np.std(shuffI/shuffHYf,0)    
                             
                # Save preprocessed Surrogates
                if opts['savePreProcessed']:
                    print('done')
                    logwrite('Saving preprocessed surrogates',1);
                    Surrogates = SavedSurrogates
                    
                    eval("plogName + ' ' + str(processLog)")
                    # Writing to a zarr file
                    nameZarr = opts['outDirectory'] + name + 'fi_' + str(fi) + opts['preProcessedSuffix']
                    dpth = np.shape(Surrogates)[0]
                    rrw = np.shape(Surrogates)[1]
                    clm = np.shape(Surrogates)[2]
                    #print(rrw,clm)
                    ds1 = xr.Dataset({'Surrogates': ( ('depth', 'nrow', 'nvar'), Surrogates.reshape([dpth,rrw,clm]) )},  # data
                                     coords={'depth': np.arange(dpth), # dimesion 1 
                                             'nrow': np.arange(rrw),
                                             'nvar': np.arange(clm)})
                    ds1.to_zarr(str(nameZarr) + '.zarr', append_dim='depth') # a classified surrogate data is saved                                              
                    
                    
    # ##############
    # Step 7 post processing   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # POSTPROCESS DERIVED INFORMATION THEORY AND PHYSICAL QUANTITIES
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Statistical significance thresholds
    if opts['SurrogateMethod'] > 0:
        R['SigThreshT'] = R['meanShuffT'] + opts['oneTailZ'] * R['sigmaShuffT']
        R['SigThreshI'] = R['meanShuffI'] + opts['oneTailZ'] * R['sigmaShuffI']
        R['SigThreshTR'] = R['meanShuffTR'] + opts['oneTailZ'] * R['sigmaShuffTR']
        R['SigThreshIR'] = R['meanShuffIR'] + opts['oneTailZ'] * R['sigmaShuffIR']
                                                              
    # Derived Quantities
    if opts['doEntropy']:
        logwrite('Computing final entropy quantities.',1)
        R['Tplus'],R['Tminus'],R['Tnet'],R['TnetBinary']=DoProduction(R['T'])
        
        R['InormByDist'],R['TnormByDist'],R['SigThreshInormByDist'],R['SigThreshTnormByDist'],R['Ic'],R['Tc'],R['TvsIzero'],        R['SigThreshTvsIzero'],R['IR'],R['TR'],R['HXtNormByDist'],R['IvsIzero'],R['SigThreshIvsIzero'] =        NormTheStats(R['nBinVect'],R['I'],R['T'],R['SigThreshI'],R['SigThreshT'],R['meanShuffI'],R['sigmaShuffI'],R['meanShuffT'],R['sigmaShuffT'],R['HXt'],R['HYf'],R['lagVect'])
        
        R['Abinary'],R['Awtd'],R['AwtdCut'],R['charLagFirstPeak'],R['TcharLagFirstPeak'],R['charLagMaxPeak'],R['TcharLagMaxPeak'],        R['TvsIzerocharLagMaxPeak'],R['nSigLags'],R['FirstSigLag'],R['LastSigLag'] = AdjMatrices(R['TnormByDist'],R['SigThreshTnormByDist'],R['TvsIzero'],R['lagVect'])
        
        R['Hm'] = np.sum(np.squeeze(R['HXt'][:,1,:,1]/ np.squeeze(np.tile(np.log2(R['nBinVect']),[nDataFiles,1, 1,1])) ))/nVars
        
        R['TSTm'] = np.squeeze(np.sum(np.sum(R['T']/np.tile(np.log2(R['nBinVect']),[nDataFiles,nLags,1,nVars]),axis=0),axis=1))/(nVars**2)
        
        
    # Save output
    if opts['saveProcessNetwork'] == 1:
        logwrite('Saving results.',1)
        outfile = opts['outFileProcessNetwork']
            
        # Rename the processLog so it is unique
                
        eval("plogName + ' ' + str(processLog)")
     
                
        # -----------------------
        # Writing result to a pickle file
        nameZarr_out = opts['outDirectory'] + outfile
        f = open(nameZarr_out+"_R.pkl","wb")
        pickle.dump(R,f)
        f.close()
        
        # write opts on a pickle file
        f_o = open(nameZarr_out+"_opts.pkl","wb")
        pickle.dump(opts,f_o)
        f_o.close()
        
        
        
    return R, opts
    ## p = zarr.load(str(nameZarr)+'.zarr')
    ## R = pd.read_pickle(r'R.pkl') # R is the name
    # #pd.read_pickle(r'./result/Resulttest1_R.pkl')

        
    
    
    


# %%


def couplingLagPlot(R,popts):
    
    # Pull the test statistic
    # TE values (relative)
    X = eval('R' + "['" + str(popts['testStatistic'])+"']")
    nFiles,nul1, nVars,nul2 = np.shape(X)

    # TE critical values
    XSigThresh = eval('R' + "['" + str(popts['SigThresh']) + "']")
    
    
    ri = np.argwhere((np.asarray(R['varNames']) == np.asarray(popts['vars'][0])))[0]
    ci = np.argwhere((np.asarray(R['varNames']) == np.asarray(popts['vars'][1])))[0]
    lagi = np.arange(np.min(R['lagVect']), np.max(R['lagVect']+1))
    Xp = X[popts['fi'],lagi, ri,ci]
    #print(Xp)
#     print(np.max(Xp))
        
    # Surrogate TE values
   
    
    if np.shape(XSigThresh)[1] == 1:
        XsTp = XSigThresh[popts['fi'],0, ri,ci]*np.ones([len(lagi)])
    else:
        XsTp = XSigThresh(popts['fi'],lagi, ri,ci)
    
    
    plt.plot(lagi,Xp,'g',label='Rel TE from ' + popts['vars'][0] + '_to_' +popts['vars'][1])
    plt.plot(lagi,XsTp,'k-.', label='Critical TE value')
    plt.ylabel(popts['testStatistic'])
    plt.xlabel('Lag')
    plt.legend(bbox_to_anchor=(.50,1), loc="lower center") #(0.5,-0.4)
    plt.grid(linestyle='-.')
    
    
    
    return


# %%


def multiCouplingSynchronyPlot(R,popts):
    
    # Pull the test statistic
    # TE values (relative)
    
    
    
    X = eval('R' + "['" + str(popts['testStatistic'])+"']")
    nFiles,nul1, nVars,nul2 = np.shape(X)

    # TE critical values
    XSigThresh = eval('R' + "['" + str(popts['SigThresh']) + "']")
    
    ci = np.argwhere((np.asarray(R['varNames']) == np.asarray(popts['ToVar'][0])))[0]
    ri = np.setxor1d(np.arange(nVars),ci)
    lagi = np.arange(np.min(R['lagVect']), np.max(R['lagVect']+1))
    
    lagVect = R['lagVect']
    
    #print(X[popts['fi'],lagi[:,None], 0:5,ci])
    x2 = X[popts['fi'],lagi[:,None], ri,ci]
   
    X = x2.transpose()
    l0i = np.argwhere(lagVect == 0)[0]
    
    

   
    X0 = X[:,l0i]             # zero-lag statistic
    XM = np.max(X, axis=1)    #  max statistic at any lag
    maxi = np.argmax(X, axis=1) 
    lagMax = lagVect[maxi]    #  lag of max statistic
    
    if np.shape(XSigThresh)[1] == 1: 
        XsT = XSigThresh[popts['fi'], 0, ri,ci]
    else:
        XsT = XSigThresh[popts['fi'], lagi, ri,ci]
        
    XsT = np.asarray(XsT) 
    
#     print(lagMax)   
#     print(XM)
#     print(X0)
        
    colors = cm.jet(lagMax / float(max(lagMax)))
    plot = plt.scatter(lagMax, lagMax, c = lagMax, cmap = 'jet')
    plt.clf() # clear the scatter
    #plt.colorbar(plot)
   
    lablC = 'Lag (' + r'$\tau$' + ') of Max' + popts['testStatistic']
    cb = plt.colorbar(plot)
    cb.set_label(label=str(lablC)) #,weight='bold')

    
#     plt.barh(range(len(XM)), XM, color = colors)
    
    for i in np.arange(len(ri)):
        if i == len(ri) -1:
            
            plt.barh(i+1,XM[i], color = colors[i],label='TE max across lags')
            plt.barh(i+1,X0[i], color ='k', label='TE at lag 0')
            plt.plot(XsT[i]*np.ones([2]),np.array([i+1-0.5, i+1+0.5]),'--',color='gray',label='SigThresh')
        else:
            plt.barh(i+1,XM[i], color = colors[i])
            plt.barh(i+1,X0[i], color ='k')
            plt.plot(XsT[i]*np.ones([2]),np.array([i+1-0.5, i+1+0.5]),'--',color='gray')
        
    plt.legend(bbox_to_anchor=(1.3,1), loc="upper left")
    plt.yticks(np.arange(1,len(ri)+1), np.asarray(R['varNames'])[ri])
    st = 'File ' + str(popts['fi'])+ ' TE from sources to ' + str(np.asarray(R['varNames'])[ci])
    plt.xlabel(popts['testStatistic'])
    plt.title(st)
    
    return
    
    

