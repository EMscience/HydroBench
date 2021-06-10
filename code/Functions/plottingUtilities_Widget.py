import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, output_file
hv.extension("bokeh", "matplotlib")

# local 
from PN1_5_RoutineSource import *
from ProcessNetwork1_5MainRoutine_sourceCode import *

# For a single case

def CouplingAndInfoFlowPlot(optsHJ,popts):
    

    # Execute InfoFlow code
    R, optsHJ = ProcessNetwork(optsHJ)
    
    # Pull the test statistic
    # TE values (relative)
    
    R2 = copy.deepcopy(R)
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

    plt.figure()
    plt.subplot(1,2,2)
    
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
    plt.grid(linestyle='-.')
    plt.title(st)
    
    plt.figure(figsize=[12,5])
    plt.subplot(1,2,1)
    couplingLagPlot(R,popts)
    
    return R, optsHJ

# %%time
# RCalib, optsHJCal = CouplingAndInfoFlowPlot(optsHJ,popts)

def extractTestStatistics(x,toVar,testStat, SigThr):
    fi = 0
    popts = {}
    popts['testStatistic'] = testStat # Relative transfer intropy T/Hy
    popts['SigThresh'] = SigThr # significance test critical value # 
    popts['fi'] = [fi] # which file
    popts['ToVar'] = [toVar]
    
    X = eval('x' + "['" + str(popts['testStatistic'])+"']") # IR, TR, HX
    nFiles,nul1, nVars,nul2 = np.shape(X)

    # TE critical values
    XSigThresh = eval('x' + "['" + str(popts['SigThresh']) + "']")


    ci = np.argwhere((np.asarray(x['varNames']) == np.asarray(popts['ToVar'][0])))[0] # to var index
    ri = np.setxor1d(np.arange(nVars),ci) # from var index
    lagi = np.arange(np.min(x['lagVect']), np.max(x['lagVect']+1)) # lags

    x2 = X[popts['fi'],lagi[:,None], ri,ci] # TE file, lag, from to var
    X = x2.transpose()

    XsT = XSigThresh[popts['fi'], 0, ri,ci] # critic
    Source = np.asarray(x['varNames'])[ri]
    Sink = np.asarray(x['varNames'])[ci]
    
    # Extract when T > Tcrit
        # Set test stat below ctitic to nan
    for j in np.arange(X.shape[0]):
            X[j,:][X[j,:] < XsT[j]] = np.nan
    
    return X, XsT, Source, Sink


def generateResultStore(modelVersion,R): # Compiles the Results into a dictionary
    
    XTE, XsTE, Source, Sink = extractTestStatistics(R,'model_Q','TR','SigThreshTR')
    XTE_obs, XsTE_obs, Source_obs, Sink_obs = extractTestStatistics(R,'observed_Q','TR','SigThreshTR')
    XI, XsI, Source, Sink = extractTestStatistics(R,'model_Q','IR','SigThreshIR')
    
    Store = {}
    Store[modelVersion+'_I'] = pd.DataFrame(data = np.transpose(XI), columns=Source)
    Store[modelVersion+'_TE'] = pd.DataFrame(data = np.transpose(XTE), columns=Source)
    Store[modelVersion+'_TE_obs'] = pd.DataFrame(data = np.transpose(XTE_obs), columns=Source_obs)
    
    # StoreCalibrated = generateResultStore('Calibrated',RCalib)
    # StoreUncalibrated['Uncalibrated_I']
    
    return Store


def plotPerformanceTradeoff(lag, RCalib, modelVersion):
    
    Store = generateResultStore(modelVersion,RCalib)
    
    if modelVersion == 'Calibrated':
        
        PerfCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershade', 'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
        PerfCal.iloc[0,:] = 'HJ Andrews', Store['Calibrated_TE'].basin_ppt[lag] - Store['Calibrated_TE_obs'].basin_ppt[lag], 1 - Store['Calibrated_I'].observed_Q[lag]

        plt.figure(figsize=[7,5])
        plt.scatter(PerfCal.iloc[:,1],PerfCal.iloc[:,2],color='blue', s = 50, marker = 'o', facecolors='none', edgecolors='r', label='Calibrated')
        plt.axvline(x=0,color = 'b',ls=':', lw = 3)
        plt.xlabel(PerfCal.columns[1], size=12)
        plt.ylabel(PerfCal.columns[2], size = 12)
        plt.grid(linestyle='-.')
        plt.title('Performance tradeof at lag = ' + str(lag), size =14) 
        plt.legend()
        
    if modelVersion == 'Uncalibrated':
                
        PerfUnCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershade', 'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
        PerfUnCal.iloc[0,:] = 'HJ Andrews', Store['Uncalibrated_TE'].basin_ppt[lag] - Store['Uncalibrated_TE_obs'].basin_ppt[lag], 1 - Store['Uncalibrated_I'].observed_Q[lag]

        plt.figure(figsize=[7,5])
        plt.scatter(PerfUnCal.iloc[:,1],PerfUnCal.iloc[:,2],color='black', s = 50, marker = 'o', facecolors='none', edgecolors='k', label='UnCalibrated')
        plt.axvline(x=0,color = 'b',ls=':', lw = 3)
        plt.xlabel(PerfCal.columns[1], size=12)
        plt.ylabel(PerfCal.columns[2], size = 12)
        plt.grid(linestyle='-.')
        plt.title('Performance tradeof at lag = ' + str(lag), size =14) 
    plt.legend()
    
    
    
def NSE(o,s, k): # computes both NSE and logNSE
    NSE = 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
    
    logS = o + k
    logO = s + k
    logNSE =  1 - sum((logS - logO)**2)/sum((logO-np.mean(logO))**2)
    
    return NSE, logNSE    
    
    
def generateChordPlots(R,optLag,optsHJ,modelVersion):
    
    CalTR = R['TR'][0,optLag,:,:] # Calibrated, picked zero as Lag since it is max across variables
    CalibChord = pd.DataFrame(data=CalTR, columns=optsHJ['varNames'], index=optsHJ['varNames'])
    dfCal = CalibChord.drop(columns=['observed_Q','basin_potet'],index=['observed_Q','basin_potet'], axis=[0,1])
    # Set the diagonals to zero for the chord plot
    np.fill_diagonal(dfCal.values, 0)
    
    # Declare a gridded HoloViews dataset and call dframe to flatten it
    dataCal = hv.Dataset((list(dfCal.columns), list(dfCal.index), (dfCal.T*1000).astype(int)),
                  ['source', 'target'], 'value').dframe()
    
    dfCal.columns = ['Precipitation','Min Temperature','Max Temperature','Streamflow','Soil Moisture','Snow melt','Actual ET']
    dfCal.index = ['Precipitation','Min Temperature','Max Temperature','Streamflow','Soil Moisture','Snow melt','Actual ET']
    dfCal.loc[:,'Min Temperature']=int(0)
    dfCal.loc[:,'Max Temperature']=int(0)
    dfCal.loc[:,'Precipitation']=int(0)
    dfCal.loc[('Streamflow'),:]=int(0)
    dfCal.loc[('Actual ET'),('Streamflow','Snow melt')]=int(0) # set all to 0?????
    dfCal.loc['Soil Moisture','Snow melt']=int(0)
    dfCal.loc['Precipitation','Actual ET']=int(0)
    dfCal.loc['Snow melt',('Actual ET','Soil Moisture')]=int(0)
    
        # Calibrated chord diagram
    dataCal = hv.Dataset((list(dfCal.columns), list(dfCal.index), (dfCal.T*1000).astype(int)),
                  ['source', 'target'], 'value').dframe()
    
     # Now create your Chord diagram from the flattened data
    #plt.title('Uncalibrated')
    chord_Cal = hv.Chord(dataCal)
    chord_Cal.opts(title=modelVersion,
        node_color='index', edge_color='source', label_index='index', 
        cmap='Category10', edge_cmap='Category10', width=500, height=500)
    
    
    
    return chord_Cal, dfCal*100
    # Plot process networks
# optLag = 0 # lag time -- lag 0 is chosen as TE is max at lag 0.
# PN_Plot, PN_Table = generateChordPlots(RCalib,optLag,optsHJ,'Calibrated') # lag=0

def generateChordPlots2(R,optLag,optsHJ,modelVersion):
    
    CalTR = R['TR'][0,optLag,:,:] # Calibrated, picked zero as Lag since it is max across variables
    CalibChord = pd.DataFrame(data=CalTR, columns=optsHJ['varNames'], index=optsHJ['varNames'])
    dfCal = CalibChord.drop(columns=['observed_Q','basin_potet'],index=['observed_Q','basin_potet'], axis=[0,1])
    # Set the diagonals to zero for the chord plot
    np.fill_diagonal(dfCal.values, 0)
    
    
    dfCal.columns = ['Precipitation','Min Temperature','Max Temperature','Streamflow','Soil Moisture','Snow melt','Actual ET']
    dfCal.index = ['Precipitation','Min Temperature','Max Temperature','Streamflow','Soil Moisture','Snow melt','Actual ET']
    dfCal.loc[:,'Min Temperature']=int(0)
    dfCal.loc[:,'Max Temperature']=int(0)
    dfCal.loc[:,'Precipitation']=int(0)
    dfCal.loc[('Streamflow'),:]=int(0)
    dfCal.loc[('Actual ET'),('Streamflow','Snow melt')]=int(0) # set all to 0?????
    dfCal.loc['Soil Moisture','Snow melt']=int(0)
    dfCal.loc['Precipitation','Actual ET']=int(0)
    dfCal.loc['Snow melt',('Actual ET','Soil Moisture')]=int(0)
    
        # Calibrated chord diagram
    dataCal = hv.Dataset((list(dfCal.columns), list(dfCal.index), (dfCal.T*1000).astype(int)),
                  ['source', 'target'], 'value').dframe()
    
     # Now create your Chord diagram from the flattened data
    #plt.title('Uncalibrated')
    chord_Cal = hv.Chord(dataCal)
    chord_Cal.opts(title=modelVersion,
        node_color='index', edge_color='source', label_index='index', 
        cmap='Category10', edge_cmap='Category10', width=500, height=500)
    
    show(hv.render(chord_Cal) )
    