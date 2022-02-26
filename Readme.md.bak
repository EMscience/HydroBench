[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/NHM_PRMS_Bechmarking/HEAD)

Please click the above binder link to launch the notebook on cloud.
 
## Hydrological Model Benchmarking and Diagnostics

Hydrological model performance is commonly evaluated based on different statistical metrics e.g., 
the Nash Sutcliffe coefficient ([NSE](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient)). 
However, these metrics do not reveal model functional performances, such as how different flux and store variables interact 
within the model. As such, they are poor in model diagnostics and fail to indicate whether the model is right for the right 
reason. In contrast, hydrological signatures and information theoretic metrics are capable of revealing model internal functions 
and their tradeoffs with predictive performance. 

In this notebook, we present HydroBench an interactive and reproducible hydrological benchmarking and diagnostics tool. HydroBench 
computes not only traditional predictive performance metrics but also both hydrological signatures and information flow metrics. 
From hydrological signatures, HydroBench includes flow duration curves,
hydrograph recession analysis and water-balance metrics while from information theoretic analysis it computes
[Transfer Entropy (TE)](https://en.wikipedia.org/wiki/Transfer_entropy) and 
[Mutual Information(MI)](https://en.wikipedia.org/wiki/Mutual_information), in diagnosing model functional performances.

We demonstrate the application of these metrics on the the National Hydrologic Model product using the PRMS model ([NHM-PRMs]()). 
NHM-PRMS has two model products covering the CONUS - the calibrated and uncalibrated model products. 
Out of the CONUS wide NHM-PRMS products, this notebook focused on the NHM-PRMS product at the 
Cedar River, WA. 

### File Description

#### Scripts
	- Widgets_NHM_PRMS_Benchmarking.ipynb -- the main notebook demonstrating the implimentation of information flow 
	(info-flow) based model benchmarking.

	The notebook demonstrates:
	1. Traditional model performance efficiency measure using NSE, KGE, PBIAS and logNSE
	2. Hydrological signature measures including flow duration and recession curves
	3.1 Tradeoffs between predictive and functional performance
	3.2 Info-flow based model internal functions using process networks (PN)


#### Data
- Contains two example outpus of the [NHM-PRMS](https://pubs.er.usgs.gov/publication/tm6B9) model results at the Cedar River, WA
	1. 12115000_Calibrated.statVar -- Calibrated NHM-PRMS model output data
	2. 12115000_unCalibrated.statVar -- Uncalibrated NHM-PRMS model output data


#### Functions 

- Contains local routines 

	- plottingUtilities_Widget.py -- contains the interactive widget plot utilities for the traditional and hydrological signatures
	- PN1_5_RoutineSource.py -- contains utilities of the main info-flow routines
	- ProcessNetwork1_5MainRoutine_sourceCode.py -- contains the main info-flow routines
>>
                               =============[********]============== 
\
*Edom Moges* \
*edom.moges@berkeley.edu* \
*[Environmental Systems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)*\
*University of California, Berkeley* 
