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
from scipy import interpolate

# local 
from PN1_5_RoutineSource import *
from ProcessNetwork1_5MainRoutine_sourceCode import *

# For a single case

pathData = (r"./Data/")
pathResult = (r"./Result/")

def CouplingAndInfoFlowPlot(optsHJ,popts):
    
    """Computes the information flow metrics and plots source to sink information flow test statistics e.g., TE along with its statistical significance level at each lag. 
    
    .. ref::
   Ruddell et al., 2019 WRR

    Parameters
    ----------
    optsHJ : a dictionary that defines the plotting options such as file names, number of bins. Please refer to the main Notebook for the complete list of options.
    
    popts : an output dictionary containing the information theoretic metrics.
        model simulated time series.
        
    Log_factor : float, optional
        offset for log transformed computation of the metrics. default is 0.1.

    Returns
    -------
    1. a pickle file saved in the results folder containing information theoretic metrics including:
    1.1 H - entopy of each varible, and joint entropy of the combination of each variables
    1.2 MI - mutual information between each of the two variables
    1.3 TE - transfer entropy from to each variable
    1.4 their statistical significance
    2. a plot of source to sink information flow values. 
    2.1 bar plot of each source to any sink
    2.2 line plot of lag vs the test statistics for the indicated source and sink variables. 
    3. the options defined by the input optsHJ.

    """
    

    # Execute InfoFlow code -- main function
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

def extractTestStatistics(x,toVar,testStat, SigThr,fi = 0):
    
    """Returns any metric among the calculated information theoretic metrics. Name of the test statistics is defined by the parameter testStat.
    
    .. ref::
    

    Parameters
    ----------
    x  : a dictionary containing the information theoretic metrics.
    toVar : the name of the sink variable. 
    testStat : the name of the test statistics to be extracted
    SigThr : the statistical significance level
        
    Returns
    ---------
    the information theoretic metric defined by testStat.

    """
    
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
    
    """Returns a dictionary of the information theoretic metrics.
    
    .. ref::
    

    Parameters
    ----------
    modelVersion  : user defined model name e.g., calibrated or uncalibrated.
    R : a pickle file with the information theoretic metrics. 
        
    Returns
    ---------
    Store : a dictionary containing the information theoretic metrics.

    """
    
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
    
    """Returns the plot of the tradeoff between functional (transfer entropy) and predictive (mutual entropy) performances.
    
    .. ref:: Ruddell et al., 2019 WRR
    

    Parameters
    ----------
    lag  : lag time for computing transfer entropy and mutual enformation.
    RCalib : the information theoretic metrics repository
    modelVersion : model version e.g. calibrated, uncalibrated
    WatershedName : watershed name
    SourceVar : the name of the source variable 
    SinkVariable : the name of the sink variable 
        
    Returns
    ---------
    plots tranfer entropy vs mutual information.

    """
    
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
    


def plotPerformanceTradeoffNoFigure(lag, RCalib, modelVersion, WatershedName, SourceVar, SinkVar):
    
    """Returns a dataframe of the plot of the tradeoff between functional (transfer entropy) and predictive (mutual entropy) performances.
    
    .. ref:: Ruddell et al., 2019 WRR
    

    Parameters
    ----------
    lag  : lag time for computing transfer entropy and mutual enformation.
    RCalib : the information theoretic metrics repository
    modelVersion : model version e.g. calibrated, uncalibrated
    WatershedName : watershed name
    SourceVar : the name of the source variable 
    SinkVariable : the name of the sink variable 
        
    Returns
    ---------
    plots tranfer entropy vs mutual information.

    """ 
    
    Store = generateResultStore(modelVersion,RCalib)
    
    PerfCal = pd.DataFrame(np.ones([1,3])*np.nan,columns= ['Watershed', 
            'Functional Performance (TEmod - TEobs)', 'Predictive Performance (1-MI)'])
    
    PerfCal.iloc[0,:] = WatershedName, Store[modelVersion+'_TE'][SourceVar][lag]- \
    Store[modelVersion+'_TE_obs'][SourceVar][lag], 1 - Store[modelVersion+'_I'][SinkVar][lag]
    return PerfCal
    

    
    
def kge(observed_Q, model_Q):
    
    """Returns Kling-Gupta efficiency measure.
    
    .. ref::
    

    Parameters
    ----------
    observed_Q  : observed timeseries.
    model_Q : model simulated timeseries. 
        
    Returns
    ---------
    kge : Kling-Gupta efficiency measure .

    """
    
    cc = pearsonr(observed_Q, model_Q)[0]
    alpha = np.std(model_Q) / np.std(observed_Q)
    beta = np.sum(model_Q) / np.sum(observed_Q)
    return 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    
def PredictivePerformance(ModelVersion, PerformanceMetrics, MetricTransformation, nameFCalib, nameFunCalib, obsQCol, modQCol): # computes both NSE and logNSE
    
    """Returns an interactive plot of untransformed and logarithm transformed predictive performance measures including:
     1. Nash-Sutcliffe coefficent (NSE),
     2. Kling-gupta efficiency (KGE)
     3. Percent Bias (PBIAS)
     4. Pearson correleation coefficient (r)
     5. Hydrograph plot ( a time series plot of streamflow and precipitation.

    .. ref::
    Klign and Gupta 2009, title' Hydrological Sciences Journal.

    Parameters
    ----------
    ModelVersion : a sting indicating the name of the model version. Options are calibrated an Uncalibrated.
    PerformanceMetrics : flag indicating which predictive performance metrics to print out. 
    MetricTransformation : flag indicating a logarithmic transform or untransformed. Options are  'Untransformed', 'Logarithmic'
    nameFCalib : name of the calibrated model file.
    nameFunCalib : name of the uncalibrated model file.
    obsQCol : column number of observed streamflow. 
    modQCol : column number of model simulated streamflow.
    
    Returns
    -------
    An interactive plot of untransformed and transformed predictive performance metrics along with a hydrograph.

    """
    
    
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
        if PerformanceMetrics == 'r':
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
        if PerformanceMetrics == 'r':
            Pmt = pearsonr(logO, logS)[0]
            
            
        return print(colored(text = ['log' + PerformanceMetrics + ' of the', ModelVersion, 'model is = ', np.round(Pmt,3)], 
                             color='green', attrs=['reverse', 'blink']) )
    # plot hydrograph
    
def PredictivePerformanceSummary(observed_Q, model_Q,Log_factor=0.1):
    
    """Returns untransformed and logarithm transformed predictive performance measures including:
     1. Nash-Sutcliffe coefficent (NSE),
     2. Kling-gupta efficiency (KGE)
     3. Percent Bias (PBIAS)
     4. Pearson correleation coefficient (r)

    .. ref::
    Klign and Gupta 2009, title' Hydrological Sciences Journal.

    Parameters
    ----------
    observed_Q : numpy array 
        observed time series
    model_Q : numpy array 
        model simulated time series
    Log_factor : float, optional
        offset for log transformed computation of the metrics. default is 0.1.

    Returns
    -------
    Pandas data frame containig both logarithm transformed and untransformed metrics.

    """
    
    PMT = pd.DataFrame(data = np.nan, columns = ['NSE', 'KGE', 'PBIAS', 'r'],
                       index = ['Untransformed Flow', 'logTransformed Flow'])
    
    PMT.loc['Untransformed Flow', 'NSE'] = 1 - sum((model_Q - observed_Q)**2)/sum((observed_Q - np.mean(observed_Q))**2)
    PMT.loc['Untransformed Flow', 'KGE'] = kge(observed_Q, model_Q)
    PMT.loc['Untransformed Flow', 'PBIAS'] = (100 * np.sum(observed_Q - model_Q, axis=0)/ np.sum(observed_Q))
    PMT.loc['Untransformed Flow', 'r'] = pearsonr(observed_Q, model_Q)[0]
    
    logO = np.log(observed_Q + Log_factor)
    logS = np.log(model_Q + Log_factor)
    
    PMT.loc['logTransformed Flow', 'NSE'] = 1 - sum((logS - logO)**2)/sum((logO-np.mean(logO))**2)
    PMT.loc['logTransformed Flow', 'KGE'] = kge(logO, logS)
    PMT.loc['logTransformed Flow', 'PBIAS'] = None
    PMT.loc['logTransformed Flow', 'r'] = pearsonr(logO, logS)[0]
    
    return PMT

def plot_FDC(Q, SlopeInterval, title, colrShape, unit):
    
    """Returns a plot of Flow Duration Curve (FDC) that relates streamflow and its exceedance probability.
    
    .. ref::
    

    Parameters
    ----------
    Q : numpy array 
        Streamflow time series
    SlopeInterval : percentage exceedance levels to compute FDC slopes e.g., [25, 45]
    title : label associated with the streamflow data. 
        
    colrShape : string indicating plot markers
    unit : unit associated with the streamflow data e.g., cfs

    Returns
    -------
    A plot of flow duration curve.
    Slope of the flow duration for a given exceedance interval.

    """
        
    n = len(Q)
    sorted_array = np.sort(Q)
    
    # Reverse the sorted array
    reverse_array = sorted_array[::-1]
    excedanceP = np.arange(1,n+1)/(n+1)
    median_Q = np.nanmedian(Q)
    f = interpolate.interp1d(excedanceP, reverse_array)
    Q_atGivenSlope = [f(SlopeInterval[0]/100.0), f(SlopeInterval[1]/100.0)]
    slope_FDC = -1.0*(Q_atGivenSlope[1] - Q_atGivenSlope[0]) / median_Q*(SlopeInterval[1]-SlopeInterval[0])
    
    
       
    plt.plot(excedanceP,reverse_array,colrShape,label= title)
    plt.xlabel('Exceedence probabilty',fontsize=20)
    plt.ylabel('Streamflow' + unit,fontsize=20)
    plt.yscale('log')
    plt.grid(linestyle = '-.')
    plt.legend()
    
    return slope_FDC
    
def plotRecession(ppt, Q, dateTime, title,labelP,labeltxt, season, alpha):
    
    
    """Returns a plot of Recession curve that relates mean streamflow and its time derivative in the absence of precipitation.
    
    .. ref::
    

    Parameters
    ----------
    ppt : numpy array containing precipitation time series.
        
    Q : a numpy array containg streamflow data with corresponding to the precipitation data. 
        
    dateTime : a date time timestamp for both precipitation streamflow. 
    title : title for the plotted recession curve.
    labelP : label of the streamflow timeseries.
    labeltxt: plotting marker color and shape.
    season: which season recession curve to plot. The options are:
    1. Summer - covering the months of April to September.
    2. Winter - covering the months of October to March.
    3. All - covers the entire year
    alpha : transparecy for the plotted marker. 
    
    Returns
    -------
    A plot of recession curve.
    z : the slope and intercept of the recession curve in the log-space

    """
    
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
            startdate = '10-' + str(year)
            enddate = '3-' + str(year+1)
            rain = np.array(ppt.loc[startdate:enddate])
            runoff = np.array(Q.loc[startdate:enddate])
        elif season == 'Summer':
            # Summer month recessions
            startdate = '4-' + str(year)
            enddate = '9-' + str(year)
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
    
    plt.xlabel('log(Q)',fontsize=14)
    plt.title(season + ' ' + title ,fontsize=14)#12
    plt.ylabel(r'$\log \left( -\mathrm{\frac{dQ}{dt}}\right)$', color='k',fontsize=14)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=14)
    
    # Fitting a linear line to recession
    x = np.log(qs)
    y = np.log(-1*dqs)
    z = np.polyfit(x, y, 1)
    
    return z
    
    
    
def AnnualRunoffCoefficient(table,StrtHydroYear,EndHydroYear,PrecipName,RunoffName):
    
    """Returns annual runoff coefficient.
    
    .. ref::
    

    Parameters
    ----------
    table : numpy array containg precipitation, observed streamflow and model simulated streamflow.  
        
    StrtHydroYear : Starting month of the hydrological year. Default is October.
    EndHydroYear : Ending month of the hydrological year. Default is September.
    PrecipName : name of the precipitation data.
    RunoffName : name of the streamflow data.
    
    Returns
    -------
    Annual Runoff Coefficient for both observed and model data.

    """
    
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

def histedges_equalA(x, nbin):
    
    """Returns bin edges based on equal area binnin.
    
    .. ref::
    

    Parameters
    ----------
    x : numpy array 
    nbin : number of bins. 
        
    Returns
    ---------
    a numpy array of the bin edges .

    """
    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    return np.interp(np.linspace(0, tmp.max(), nbin + 1),tmp,np.sort(x))

def histedges_equalN(x, nbin): 
    
    """Returns bin edges based on equal depth binning.
    
    .. ref::
    

    Parameters
    ----------
    x : numpy array 
    nbin : number of bins. 
        
    Returns
    ---------
    a numpy array of the bin edges .
    
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),np.arange(npt),np.sort(x))

def histedges_equalW(x,nbin): 
    
    """Returns bin edges based on equal width binning.
    
    .. ref::
    

    Parameters
    ----------
    x : numpy array 
    nbin : number of bins. 
        
    Returns
    ---------
    bns : a numpy array of the bin edges .
    
    """
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
    
    """Returns bin id of a given time series.
    
    .. ref::
    

    Parameters
    ----------
    obs : observed timeseries data
    model : simulated timeseries data
    binSize : number of bins for Time linked FDC
    Flag : Flag indicating binning methods
        Flag = 1 : Equal Width
        Flag = 2 : Equal Area
        Flag = 3 : Equal depth (i.e. Frequency)
    
    Returns
    ---------
    clsObs : bin id of the observed timeseries
    clsMod : bin id of the model timeseries based on the observed bin classes/id
    bns : bin edges
    
    """
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
    """Returns a square matrix of a time series of bin ids.
    
    .. ref::
    

    Parameters
    ----------
    df_confusion : a time series of bin ids 
    binSize : number of bins. 
        
    Returns
    ---------
    SquaMat: a square matrix with counts of model bin id that in the same bin class as observed data.
    
    """
    
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
    
    """Returns a heatmap plot of the square matrix that relates the counts of simulated and observed binnin based on the function squareConfusionMatix().
    
    .. ref::
    

    Parameters
    ----------
    df_confusion : a square matrix with counts of model bin id that in the same bin class as observed data 
    bns : bin edges.  
    ticks : markers for the heatmap plot
        
    Returns
    ---------
    a heatmap plot of time linked flow duration curve .
    """
    
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

    plt.xticks(tick_marksx, np.round(bns,2), rotation=90,fontsize=10)
    plt.yticks(tick_marksy, np.round(bns,2),fontsize=10)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name,fontsize=14)
    plt.xlabel(df_confusion.columns.name,fontsize=14)
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
    
    """Returns a histogram plot of the square matrix that relates the counts of simulated and observed binnin based on the function squareConfusionMatix().
    
    .. ref::
    

    Parameters
    ----------
    df_confusion : a square matrix with counts of model bin id that in the same bin class as observed data 
    bns : bin edges.  
    sz : figure size
    title : figure title
        
        
    Returns
    ---------
    a histogram plot of the time linked flow duration curve.
    """
    
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
    plt.yticks(fontsize=10)
    ax.yaxis.grid(linestyle='-.')
    ax.set_xlabel('Median value')
    ax.set_ylabel('Frequency')
    ## add a legend
    ax.legend( (Observed[0], Model[0]), title )
    plt.show()
    
def timeLinkedFDC(obs, diag, mod, binSize, Flag, FigSize1, FigSize2, NameObserved, NameModel,ticks):
    
    """Returns a plot of the time linked flow duration curve.
    and the fraction of modeled data in the diagonal
    
    .. ref:: 
    

    Parameters
    ----------
    obs : numpy array containg observed streamflow.  
    mod : numpy array containg model simulated streamflow
    binSize : number of bins to construct a histogram. 
    Flag : Which binning method to use the ptions are:
    a. Flag = 1 :  Equal width based binning
    b. Flag = 2 : Equal area based binning
    c. Flag = 3 : Equal depth/frequency based binning
    FigSize1 : figure dimension for the heatmap plot
    FigSize2 : figure dimension for the histogram plot.
    NameObserved : Label of the observed streamflow
    NameModel : label for the model simulated streamflow
    ticks : 
    
    Returns
    -------
    Returns a plot of the time linked flow duration curve.
    Fraction of data in the diagonal.

    """
    
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
    
    if diag == 0:
        DiagElement = np.diagonal(df_conf_norm,diag) # Only the main diagonal
        RatioDiag = np.nansum(DiagElement)/np.nansum(df_conf_norm)
    else:
        dimn = np.arange(-1*diag,diag+1)
        Diagsum = np.ones(np.shape(dimn))*np.nan
        
        for i in dimn:
            Diagsum[i] = np.nansum(np.diagonal(df_conf_norm,i))# sum of diagonal elements
        RatioDiag = np.nansum(Diagsum)/np.nansum(df_conf_norm)

    return RatioDiag
