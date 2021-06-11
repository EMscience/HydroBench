[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/NHM_PRMS_Bechmarking/HEAD)

## Information flow based Model Benchmarking

Hydrological model performance is commonly evaluated based on different statistical metrics e.g., 
the Nash Sutcliffe coefficient ([NSE](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient)). 
However,these metrics do not reveal model functional performances, such as how different flux and store variables interact 
within the model. As such, they are poor in model diagnostics and fail to indicate whether the model is right for the right 
reason. In contrast, information theoretic metrics are capable of revealing model internal functions and their tradeoffs with
predictive performance. In this, notebook we demonstrate the use of interactive and reproducible computation of information 
flow metrics, particularly [Transfer Entropy (TE)](https://en.wikipedia.org/wiki/Transfer_entropy) and 
[Mutual Information(MI)](https://en.wikipedia.org/wiki/Mutual_information), in diagnosing model performance.

The model in focus is the the National Hydrologic Model using the PRMS model ([NHM-PRMs]()). 
NHM-PRMS has two model products covering the CONUS - the calibrated and uncalibrated model products. 
Out of the CONUS wide NHM-PRMS products, this notebook focused on the NHM-PRMS product at the 
[HJ Andrews watershed, OR](https://andrewsforest.oregonstate.edu/). 

### File Description

#### Scripts
	- Widgets_NHM_PRMS_Benchmarking_at_HJAndrews.ipynb -- the main notebook demonstrating the implimentation of information flow 
	(info-flow) based model benchmarking.

	The notebook demonstrates:
	1. Traditional model performance efficiency measure using NSE and logNSE
	2. Tradeoffs between predictive and functional performance
	3. Info-flow based model internal functions using process networks (PN)
	4. Sensitivity of Info-flow metrics computation

#### Data
- contains two outpus of the [NHM-PRMS](https://pubs.er.usgs.gov/publication/tm6B9) model results at the HJ Andrews Watershed, OR
	1. CalibratedHJAndrew.txt -- Calibrated NHM-PRMS model output data
	2. UnCalibratedHJAndrew.txt -- Uncalibrated NHM-PRMS model output data



#### Functions 

- contains local routines 

	- plottingUtilities_Widget.py -- contains the interactive widget plot utilities
	- PN1_5_RoutineSource.py -- contains utilities of the main info-flow routines
	- ProcessNetwork1_5MainRoutine_sourceCode.py -- contains the main info-flow routines
	