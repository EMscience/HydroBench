[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/NHM_PRMS_Bechmarking/HEAD)

## Information flow based Model Benchmarking

### File Description

#### Data
- contains two outpus of the [NHM-PRMS](https://pubs.er.usgs.gov/publication/tm6B9) model results at the HJ Andrews Watershed, OR
	1. CalibratedHJAndrew.txt -- Calibrated model output
	2. UnCalibratedHJAndrew.txt -- Uncalibrated model output

#### Scripts
	* Widgets_NHM_PRMS_Benchmarking_at_HJAndrews.ipynb -- the main notebook demonstrating the implimentation of info-flow based model benchmarking.

	The notebook demonstrates:
	1. Traditional model performance measure using NSE
	2. Tradeoffs between predictive and functional performance
	3. Info-flow based model internal functions using process networks (PN)
	4. Sensitivity of Info-flow metrics computation

#### Folders
Functions -- contains local routines 

	* plottingUtilities_Widget.py -- contains the interactive widget plot utilities
	* PN1_5_RoutineSource.py -- contains utilities of the main info-flow routines
	* ProcessNetwork1_5MainRoutine_sourceCode.py -- contains the main info-flow routines
	