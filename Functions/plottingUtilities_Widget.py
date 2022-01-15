import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.metrics import r2_score

import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, output_file
from scipy.stats import pearsonr
hv.extension("bokeh", "matplotlib")

# local 
from PN1_5_RoutineSource import *
from ProcessNetwork1_5MainRoutine_sourceCode import *

# For a single case

pathData = (r"./Data/")
pathResult = (r"./Result/")

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
    popts['testStatistic'] = testStat # Relative transfer entropy T/Hy
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

def plotPerformanceTradeoff(lag, RCalib, modelVersion, WatershedName, SourceVar, SinkVar):
    
    Store = generateResultStore(modelVersion,RCalib)
    
    PerfCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershed', 
            'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
    
    PerfCal.iloc[0,:] = WatershedName, Store[modelVersion+'_TE'][SourceVar][lag]- \
    Store[modelVersion+'_TE_obs'][SourceVar][lag], 1 - Store[modelVersion+'_I'][SinkVar][lag]
    
    plt.figure(figsize=[7,5])
    plt.scatter(PerfCal.iloc[:,1],PerfCal.iloc[:,2],color='blue', 
            s = 50, marker = 'o', facecolors='none', edgecolors='r', label=modelVersion)
    plt.axvline(x=0,color = 'b',ls=':', lw = 3)
    plt.xlabel(PerfCal.columns[1], size=12)
    plt.ylabel(PerfCal.columns[2], size = 12)
    plt.grid(linestyle='-.')
    plt.title('Performance tradeof at lag = ' + str(lag), size =14) 
    plt.legend()
    return PerfCal
    
    
def plotPerformanceTradeoff_Two(lag, RCalib, modelVersion):
    
    Store = generateResultStore(modelVersion,RCalib)
    
    if modelVersion == 'Calibrated':
        
        PerfCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershed', 'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
        
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
                
        PerfUnCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershed', 'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
        PerfUnCal.iloc[0,:] = 'HJ Andrews', Store['Uncalibrated_TE'].basin_ppt[lag] - Store['Uncalibrated_TE_obs'].basin_ppt[lag], 1 - Store['Uncalibrated_I'].observed_Q[lag]

        plt.figure(figsize=[7,5])
        plt.scatter(PerfUnCal.iloc[:,1],PerfUnCal.iloc[:,2],color='black', s = 50, marker = 'o', facecolors='none', edgecolors='k', label='UnCalibrated')
        plt.axvline(x=0,color = 'b',ls=':', lw = 3)
        plt.xlabel(PerfCal.columns[1], size=12)
        plt.ylabel(PerfCal.columns[2], size = 12)
        plt.grid(linestyle='-.')
        plt.title('Performance tradeof at lag = ' + str(lag), size =14) 
    plt.legend()
    
    
    
def NSE2(o,s, k): # computes both NSE and logNSE
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

def generateChordPlots2_(R,optLag,optsHJ,modelVersion):
    
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
    
def kge(observed_Q, model_Q):
    
    cc = np.corrcoef(observed_Q, model_Q)[0, 1]
    alpha = np.std(model_Q) / np.std(observed_Q)
    beta = np.sum(model_Q) / np.sum(observed_Q)
    return 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    
def PredictivePerformance(ModelVersion, PerformanceMetrics, MetricTransformation, nameFCalib, nameFunCalib, obsQCol, modQCol): # computes both NSE and logNSE
    Log_factor = 0.1

    
    if ModelVersion == 'Calibrated':
        CalibMat = np.loadtxt(pathData + nameFCalib, delimiter='\t') # cross validate with matlab
        observed_Q = CalibMat[:,obsQCol]
        model_Q = CalibMat[:,modQCol]
        
               
        fig, ax1 = plt.subplots(figsize=[15,5])

        ax2 = ax1.twinx()
        ax1.plot(observed_Q, 'k', label= 'Observed stream flow')
        ax1.plot(model_Q, 'r', label = 'Model stream flow')
        
        ax2.plot(CalibMat[:,1], 'b-',label = 'Basin precipitation estimate')
        ax2.invert_yaxis()

        ax1.set_xlabel('Days Since January 1980')
        ax1.set_ylabel('Stream flow (cfs)',color='k')
        ax2.set_ylabel('Precipitation (in)', color='b')
        ax1.legend(loc='center', bbox_to_anchor=(0.85, 1.2))
        ax2.legend(loc='center', bbox_to_anchor=(0.5, 1.2))
        ax1.grid(linestyle='-.')
        
    if ModelVersion == 'Uncalibrated':
        UnCalibMat = np.loadtxt(pathData + nameFunCalib,delimiter='\t') # cross validate with matlab
        observed_Q = UnCalibMat[:,obsQCol]
        model_Q = UnCalibMat[:,modQCol]
        
        fig, ax1 = plt.subplots(figsize=[15,5])

        ax2 = ax1.twinx()
        ax1.plot(observed_Q, 'k', label= 'Observed Stream flow')
        ax1.plot(model_Q, 'r', label = 'Model Stream flow')
        
        ax2.plot(UnCalibMat[:,1], 'b-', label = 'Basin precipitation estimate')
        ax2.invert_yaxis()

        ax1.set_xlabel('Days Since January 1980')
        ax1.set_ylabel('Stream flow (cfs)',color='k')
        ax2.set_ylabel('Precipitation (in)', color='b')
        ax1.legend(loc='center', bbox_to_anchor=(0.85, 1.2))
        ax2.legend(loc='center', bbox_to_anchor=(0.5, 1.2))
        ax1.grid(linestyle='-.')

      
    if MetricTransformation == 'Untransformed':
        if PerformanceMetrics == 'NSE':
            Pmt = 1 - sum((model_Q - observed_Q)**2)/sum((observed_Q - np.mean(observed_Q))**2)
        if PerformanceMetrics == 'KGE':
            Pmt = kge(observed_Q, model_Q)
        if PerformanceMetrics == 'PBIAS':
            Pmt = (100 * np.sum(observed_Q - model_Q, axis=0)/ np.sum(observed_Q))
        if PerformanceMetrics == 'r2':
            Pmt = pearsonr(observed_Q, model_Q)[0] # Correlation coefficient
            
        return print(colored(text = [PerformanceMetrics +' of the', ModelVersion, 'model is = ', np.round(Pmt,3)], 
                             color='green', attrs=['reverse', 'blink']) )
    
    if MetricTransformation == 'Logarithmic':
        logO = np.log(observed_Q + Log_factor)
        logS = np.log(model_Q + Log_factor)
        
        if PerformanceMetrics == 'NSE':
            Pmt =  1 - sum((logS - logO)**2)/sum((logO-np.mean(logO))**2)
        if PerformanceMetrics == 'KGE':
            Pmt =  kge(logO, logS)
        if PerformanceMetrics == 'PBIAS':
            Pmt = None # (100 * np.sum(logO - logS, axis=0)/ np.sum(logO))
        if PerformanceMetrics == 'r2':
            Pmt = pearsonr(logO, logS)[0]
            
            
        return print(colored(text = ['log' + PerformanceMetrics + ' of the', ModelVersion, 'model is = ', np.round(Pmt,3)], 
                             color='green', attrs=['reverse', 'blink']) )
    # plot hydrograph
    
def PredictivePerformanceSummary(observed_Q, model_Q,Log_factor=0.1):
    
    PMT = pd.DataFrame(data = np.nan, columns = ['NSE', 'KGE', 'PBIAS', 'r2'],
                       index = ['Untransformed Flow', 'logTransformed Flow'])
    
    PMT.loc['Untransformed Flow', 'NSE'] = 1 - sum((model_Q - observed_Q)**2)/sum((observed_Q - np.mean(observed_Q))**2)
    PMT.loc['Untransformed Flow', 'KGE'] = kge(observed_Q, model_Q)
    PMT.loc['Untransformed Flow', 'PBIAS'] = (100 * np.sum(observed_Q - model_Q, axis=0)/ np.sum(observed_Q))
    PMT.loc['Untransformed Flow', 'r2'] = pearsonr(observed_Q, model_Q)[0]
    
    logO = np.log(observed_Q + Log_factor)
    logS = np.log(model_Q + Log_factor)
    
    PMT.loc['logTransformed Flow', 'NSE'] = 1 - sum((logS - logO)**2)/sum((logO-np.mean(logO))**2)
    PMT.loc['logTransformed Flow', 'KGE'] = kge(logO, logS)
    PMT.loc['logTransformed Flow', 'PBIAS'] = None
    PMT.loc['logTransformed Flow', 'r2'] = pearsonr(logO, logS)[0]
    
    return PMT

def plot_FDC(Q, title, colrShape, unit):
    n = len(Q)
    sorted_array = np.sort(Q)
    
    # Reverse the sorted array
    reverse_array = sorted_array[::-1]
       
    plt.plot(np.arange(1,n+1)/(n),reverse_array,colrShape,label= title)
    plt.xlabel('Exceedence probabilty',fontsize=20)
    plt.ylabel('Streamflow' + unit,fontsize=20)
    plt.yscale('log')
    plt.grid(linestyle = '-.')
    plt.legend()
    
def plotRecession(ppt, Q, dateTime, title,labelP,labeltxt, season, alpha):
    
    # Lag in days after neglible rainfall before analysis starts
    rainfall_lag = 1.0
    mean_fraction = 0.001

    lag = 1 
    dt = 1.0
    ppt = pd.Series(data=ppt,index=dateTime)
    Q = pd.Series(data=Q,index=dateTime)
    
    # Lists to store q and its derivative
    dqs = []
    qs = []
    years = list(set(dateTime.year))
    years.sort()
    meanQ = np.mean(Q)
    for year in years:
        # Do not loop beyond the 2016 water year, which starts in 2015
        if year==years[-1]:
            continue
        if season == 'Winter':
            
            # Winter month recessions
            startdate = '11-' + str(year)
            enddate = '3-' + str(year+1)
            rain = np.array(ppt.loc[startdate:enddate])
            runoff = np.array(Q.loc[startdate:enddate])
        elif season == 'Summer':
            # Summer month recessions
            startdate = '4-' + str(year)
            enddate = '10-' + str(year)
            rain = np.array(ppt.loc[startdate:enddate])
            runoff = np.array(Q.loc[startdate:enddate])
        else:
            startdate = '1-' + str(year)
            enddate = '12-' + str(year)
            rain = np.array(ppt.loc[startdate:enddate])
            runoff = np.array(Q.loc[startdate:enddate])
            
        i = lag

        #print(dt)
        while i<len(rain):
            # Too much rain
            #print(rain[i-lag:i+1])
            if np.sum(dt*rain[i-lag:i+1]) > .002:
                i+=1
                continue

            # period of negligible rainfall 
            # Find index of next day of rainfall 
            idx_next_rain = np.where(rain[i+1:]>0)[0]
            if len(idx_next_rain)>0:
                idx_next_rain = idx_next_rain[0] + (i+1)
            else: 
                # no more rain for this particular year
                break

            # too short of a rainless period for analysis 
            if idx_next_rain==i+1: 
                i += 2
                continue

            # get dq/dt going forward, not including the next day of rainfall
            for j in range(i, idx_next_rain):
                q_diffs = runoff[j] - runoff[j+1:idx_next_rain]
                # print(idx_end)
                idx_end = np.where(q_diffs>mean_fraction*meanQ)[0]
                if len(idx_end)>0:
                    idx_end = idx_end[0]
                    qs.append((runoff[j] + runoff[j+idx_end+1])/2)
                    dqs.append((runoff[j+idx_end+1]-runoff[j])/(dt*(idx_end+1)))
                else:
                    i = idx_next_rain + lag + 1
                    break 

            i = idx_next_rain + lag + 1

    qs = np.array(qs)
    dqs = np.array(dqs)
    #print(dqs)
    plt.plot(np.log(qs),np.log(-1*dqs),labelP,label=labeltxt,alpha=alpha)
    #plt.scatter(np.log(qs),np.log(-1*dqs), label=labeltxt,alpha=alpha)
    
    plt.xlabel('log(Q)',fontsize=12)
    plt.title(season + ' ' + title ,fontsize=12)#12
    plt.ylabel(r'$\log \left( -\mathrm{\frac{dQ}{dt}}\right)$', color='k',fontsize=12)
    plt.grid(linestyle='-.')
    plt.legend()
    
    
def AnnualRunoffCoefficient(table,StrtHydroYear,EndHydroYear,PrecipName,RunoffName):
    yearInt = min(table.index.year)
    yearMax = max(table.index.year)

    years = np.arange(yearInt,yearMax, 1)
    lenYear = np.count_nonzero(years)

    RCoeff = np.nan*np.ones([lenYear,2])
    

    for year in years: # years
        if StrtHydroYear == None:
            srt = str(year) + '-10-01'
        else:
            srt = str(year) + str(StrtHydroYear)
            
        if EndHydroYear ==None:
            end = str(year+1) + '-09-30'
        else:
            end = str(year+1) + EndHydroYear
            
        NewT = table.loc[pd.to_datetime(srt):pd.to_datetime(end),:]
        Pyr = np.sum(NewT[str(PrecipName)])
        Qyr = np.sum(NewT[str(RunoffName)])

        RC = Qyr/Pyr #AET = Pyr - Q
        RCoeff[year-yearInt,0] = year
        RCoeff[year-yearInt,1] = RC
        

    StationRCoeff = np.nanmean(RCoeff[:,1])
    
    return  RCoeff, StationRCoeff

# Time Linked FDC

def histedges_equalA(x, nbin): # Equal area binning
    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    return np.interp(np.linspace(0, tmp.max(), nbin + 1),tmp,np.sort(x))

def histedges_equalN(x, nbin): # Equal depth binning - avoids empty bins - same frequency
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),np.arange(npt),np.sort(x))

def histedges_equalW(x,nbin): # equal width binning
    mn = min(x)
    mx = max(x)
    
    bns = np.arange(mn,mx,(mx-mn)/nbin)
    bns = np.r_[bns,max(x)]
    if min(x) > 0:
        bns[0] = min(x)/2
    else:
        bns[0] = bns[0]-0.1
    return bns

def FDCdiagnostics(obs,model,binSize,Flag):# Sum off-diagonals and minimize them
    # time tag the data series and bin it to a histogram/FDC
    
    # comb = np.r_[obs,model] # combined binning
    # comb = np.r_[obs, np.min(model),np.max(model)] # binning based on observed data plus model min/max
    
    # Folding model lowest and highest to observed min/max
    model[model <= np.min(obs)] = np.min(obs)
    model[model >= np.max(obs)] = np.max(obs)
    comb = obs
    
    if Flag ==1: # Equal Width
        bns = histedges_equalW(comb,binSize)
    if Flag == 2: # Equal area
        bns = histedges_equalA(comb,binSize)
    if Flag == 3: # Equal frequency
        bns = histedges_equalN(comb,binSize)

    #print(bns)
    clsObs = np.nan*np.ones(len(obs))
    clsMod = np.nan*np.ones(len(obs))
    
    clsMod = np.digitize(model, bns)
    clsObs = np.digitize(obs, bns)
    

    return clsObs, clsMod, bns

def squareConfusionMatix(df_confusion,binSize):
    # Make the matrix square 
    SquMat = np.nan*np.ones([binSize,binSize])
    for i in np.arange(1,binSize+1):
        for j in np.arange(1,binSize+1):
            #print(i,j)
            if i in df_confusion.index:
                if j in df_confusion.columns:
                    SquMat[i-1,j-1] = df_confusion.loc[i,j]
    SquMat = pd.DataFrame(data=SquMat,columns=np.arange(1,binSize+1),index=np.arange(1,binSize+1))
    SquMat.index.name = df_confusion.index.name
    SquMat.columns.name = df_confusion.columns.name
    return SquMat

#squareConfusionMatix(df_confusion,binSize)

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

def plot_confusion_matrix(df_confusion, bns, ticks):
    
    cmap=plt.cm.jet #plt.cm.jet, gray_r
    extent = (0, df_confusion.shape[1], df_confusion.shape[0], 0)
    df_confusion2 = np.ma.masked_where(df_confusion == 0.0, df_confusion)
    cmap.set_bad(color='white')
    im=plt.imshow(df_confusion2, cmap=cmap,extent=extent) # imshow
      
    
    plt.plot(range(len(bns)),'k-.')
    #plt.title(title)
    tick_marksx = np.arange(len(df_confusion.columns))
    tick_marksy = np.arange(len(df_confusion.index))
        
#     plt.xticks(tick_marksx, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marksy, df_confusion.index)

    plt.xticks(tick_marksx, np.round(bns,2), rotation=90)
    plt.yticks(tick_marksy, np.round(bns,2))
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.grid()
    
    ax = plt.gca()
    
    aspect = 20
    pad_fraction = 0.7
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb=plt.colorbar(im, cax=cax,ticks=ticks)
    cb.set_label(label='Fraction of Observed',fontsize= 12)
    plt.show()
    
def plotHist(SquMat,bns,sz, title=['Observed Q', 'Model Q']):
    fig = plt.figure(figsize=sz)
    ax = fig.add_subplot(111)
    ind = SquMat.index
    width = 0.2
    #bnsPD = pd.Series(bns).rolling(window=2).mean()[1:]
    bnsPD = bns
    Observed = ax.bar(ind, SquMat.sum(axis=1), width,color='black',)
    Model = ax.bar(ind+width, SquMat.sum(axis=0), width,color='red')
    xTickMarks = np.round(bnsPD,2)
    ax.set_xticks(ind+width/2)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=90, fontsize=10)
    ax.yaxis.grid(linestyle='-.')
    ax.set_xlabel('Median value')
    ax.set_ylabel('Frequency')
    ## add a legend
    ax.legend( (Observed[0], Model[0]), title )
    plt.show()
    
def timeLinkedFDC(obs, mod, binSize, Flag, FigSize1, FigSize2, NameObserved, NameModel,ticks):
    
    # Step 1 - generate the time tagged FDC (histogram)
    clsObs, clsMod,bns = FDCdiagnostics(obs,mod,binSize,Flag)
    
    # Step 2 - generate the confusion matrix 
    Obs_Series = pd.Series(clsObs,name=str(NameObserved))
    mod_Series = pd.Series(clsMod,name=str(NameModel))
    df_confusion = pd.crosstab(Obs_Series, mod_Series, rownames=[str(NameObserved)], 
                               colnames=[str(NameModel)]) # without margins
    
    # Step 3 - convert the confusion matrix to a square matrix
    SquMat = squareConfusionMatix(df_confusion,binSize)
    
    # Step -4 Normalize the matrix and plot
    df_conf_norm = SquMat.loc[:,SquMat.columns].div(SquMat.sum(axis=1), axis=0)
    plt.figure(figsize=FigSize1)
    plot_confusion_matrix(df_conf_norm,bns,ticks)
    
    plotHist(SquMat,bns,FigSize2,[str(NameObserved),str(NameModel)])

    
