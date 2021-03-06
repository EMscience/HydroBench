[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/HydroBench/HEAD)
[![DOI](https://zenodo.org/badge/375593287.svg)](https://zenodo.org/badge/latestdoi/375593287)

## Hydrological Model Benchmarking and Diagnostics

Hydrological model performances are commonly evaluated based on different statistical metrics e.g., the Nash Sutcliffe 
coefficient ([NSE](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient)). However, these metrics 
do not reveal neither the hydrological consistency of the model nor the model's functional performances, such as how different 
flux and store variables interact within the model. As such, they are poor in model diagnostics and fail to indicate whether 
the model is right for the right reason. In contrast, hydrological process signatures and information theoretic metrics are 
capable of revealing the hydrological consistency of the model prediction and model internal functions respectively. In this 
notebook, we demonstrate the use of interactive and reproducible comprehensive model benchmarking and diagnostic using:

1) a set of statistical predictive performance metrics: Nash Sutcliffe coefficient, Kling and Gupta coefficient, percent bias and pearson correlation coefficient

2) a set of hydrological process based signatures: flow duration curve, recession curve, time-linked flow duration curve, and runoff coefficient, and

3) information theoretic based metrics, particularly [Transfer Entropy (TE)](https://en.wikipedia.org/wiki/Transfer_entropy) and [Mutual Information(MI)](https://en.wikipedia.org/wiki/Mutual_information). 

We demonstrate the application of these metrics on the the National Hydrologic Model product using the PRMS model ([NHM-PRMs](https://www.sciencebase.gov/catalog/item/58af4f93e4b01ccd54f9f3da)). 
NHM-PRMS has two model products covering the CONUS - the calibrated and uncalibrated model products. 
Out of the CONUS wide NHM-PRMS products, this notebook focused on the NHM-PRMS product at the 
[Cedar River, WA](https://waterdata.usgs.gov/nwis/nwismap/?site_no=12115000&agency_cd=USGS). 

For a full online documentation, please refer to this [link](https://emscience.github.io/HydroBenchJBook/HydroBenchIntroduction.html)

### File Description

#### Scripts
	- Widgets_NHM_PRMS_Benchmarking.ipynb -- the main notebook demonstrating the implimentation of information flow 
	(info-flow) based model benchmarking.

	The notebook demonstrates:
	1. Traditional model performance efficiency measure using NSE, KGE, PBIAS and r along with their log transformations.
	2. Hydrological signature measures including runoff coefficient, flow duration, time linked flow duration and recession curves
	3. Information-theoretic based performance metrics
	3.1 Tradeoffs between predictive and functional performances
	3.2 Quantification of model internal functions using process networks (PN)


#### Data
- Contains two example outpus of the [NHM-PRMS](https://pubs.er.usgs.gov/publication/tm6B9) model results at the [Cedar River, WA](https://waterdata.usgs.gov/nwis/nwismap/?site_no=12115000&agency_cd=USGS):
	1. 12115000_Calibrated.statVar -- Calibrated NHM-PRMS model output data
	2. 12115000_unCalibrated.statVar -- Uncalibrated NHM-PRMS model output data


#### Functions 

- Contains local routines 

	- plottingUtilities_Widget.py -- contains the interactive widget plot utilities for the traditional and hydrological signatures
	- ProcessNetwork1_5MainRoutine_sourceCode.py -- contains the main information-theoretic routines
	- PN1_5_RoutineSource.py -- contains utilities supporting the main information-theoretic routines
>>
                               =============[********]============== 
\
*Edom Moges* \
*edom.moges@berkeley.edu* \
*[Environmental Systems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)*\
*University of California, Berkeley* 
