# Machine Learning Techniques for Fault Detection on Unmanned Underwater Vehicles

Codes for my Master's project: Machine Learning Techniques for Fault Detection on Unmanned Underwater Vehicles

Rather than an integrated package, this repository is a collection of several pieces of code used throughout the development of my Master's project. Data files are available upon request. More information can be found on **Dissertação-Felipe_Adachi.pdf**

Scripts are ordered by sequence of use:
1. Generate_Files -> 2.Correlation_Analysis -> 3. Feature_Selection -> 4.Grid_NARX -> 5.Train_Final_Models -> 6. Inject_and_eval


## Abstract

The need for reliability and safety on the operation of unmanned vehicles increases in conjunction with the growing criticality of operations undertaken by these vehicles. In underwater environments, distinguished by its hostile environment, the importance of fault-tolerant systems design is straightforward. This paper proposes a strategy for fault detection by obtaining the dynamic model of the vehicle’s normal behavior and comparing the predicted output to the actual information provided by the vehicle. The predictive model is obtained by training a Nonlinear Autoregressive with Exogenous Inputs(NARX) neural network. The proposed strategy is applied to a real case study, by gathering data of the OpenROV, a low-cost remotely operated vehicle (ROV). In a preprocessing phase, dynamic relationships between available features are assessed through auto and cross-correlation analysis. Thereafter, different feature selection procedures are applied- ReliefF, Correlation-Based Feature Selection, Stepwise Regression and no feature selection. Performance results for each resulting model are compared and assessed, regarding mean squared error of regression models. Detection performance results are also compared through precision, recall and latency measures, when the strategy is applied to several different intermittent sensor faults, from incipient to abrupt nature. Results have shown that, in overall, the ReliefF method for feature selection yielded better performance. The proposed strategy proved to be able to detect several types of faults, albeit being insensitive to certain faults of low magnitude and short duration. In addition, the proposed strategy was succesfully integrated into a framework based on Internet of Things and Big Data. Latency measurements denoted the feasibility of the implemented scheme, allowing a prompt response to the ROV pilot, contributing to the ﬂexibility and scalability of the proposed strategy.


# 1.Generate_Files

The ROV is operated at several operating sessions, and data is collected as JSON format. 
Files are present at folders by date of collection. Inside each folder, each file is named in the following format: "X.sx.vx.flags.time.json", in which:
*X* - Number of chronological order of data collection

*sx* - maneuver sequence of respective session:

    s0 - Forward/Backward
    s1 - Right/Left Turn
    s2 - Eight-shape
    s3 - Zig-zag
    s4 - Forward/half-turn/Forward
    s5 - Circular movement
    
*vx* - Motor Speed. v2 or v3 (of a scale from 1 to 5).
flags - Session details. Examples:

    controle - ROV piloted through gamepad, as opposed to the keyboard
    dpfreq - Depth gathered at 5Hz frequency. Note: A part of the data was collected with depth sampled at 1Hz.
    choque - Presence of light clashes with the border pool. Time of contact not present.

    Remaining flags can be ignored - They are outdated and are not informative
    
*hora* - Ending time of the operating session

- **preprocess.py** -> Reads the raw files from the ROV, indents, decumulates, parses mtarg argument, and stores at a new path. Each session may contain operating data from the prior sessions. Decumulation refers to isolating the data of each operating session.
- **read-db** -> Joins all separate files of each folder into a single pickle file (a json variable)
generate-set -> Generates training and validation data, after some preprocessing steps.




                                                    Division  (DB-Train e DB-Test) done manually.
                                                        
                                                            |                               out->json
                                                            |                               generate_set.py (remove LACCX.max,min e cpuusage!=1)
                                          preprocess.py     |                                |
                                                            |                                v 
                                             |              v                                                          -----
                                             |                                               |--Kfolds_cpu/Fold_1/Train_1  |
                                             |              -----DB-Train-----DB-Train-JSON--|--Kfolds_cpu/Fold_2/Train_2  |
                                             |              |             ^                  |--Kfolds_cpu/Fold_3/Train_3  |-->"Whole_Train.json"
                                             |              |             |                  |--Kfolds_cpu/Fold_4/Train_4  |
                                             v              |             |                  |--Kfolds_cpu/Fold_5/Train_5  |                                   
                                        RAW------->DB-------|             |                                            ----- 
                                                            |             |
                                                            |             |                                             ------
                                                            |             |                  |--Kfolds_cpu/Fold_1/Validate_1 |
                                                            -----DB-Test-----DB-Test-JSON----|--Kfolds_cpu/Fold_2/Validate_2 |
                                                                          ^                  |--Kfolds_cpu/Fold_3/Validate_3 |-->"Whole_Test.json"
                                                                          |                  |--Kfolds_cpu/Fold_4/Validate_4 |
                                                                          |                  |--Kfolds_cpu/Fold_5/Validate_5 |
                                                                          |                                             ------
                                                                      read_db.py             ^
                                                                      (out->  pickle file )  |
                                                                                             |
                                                                                            generate_set.py (remove LACCX.max, min e cpuusage!=1)
                                                                                            out-> json
                                                                                            

# 2.Correlation_Analysis

Performs mutual information analysis for GYROX,GYROY and GYROZ (**mutinfo_GYROX,Y,X.py**) and partial autocorrelation (**pacf.py**)


# 3.Feature_Selection
Performs Feature Selection Analysis, after correlation analysis:

- RreliefF (**rrelieff_ALL.py**). Also includes method to determine the cutoff threshold (**Relief/cutoff_threshold.py**).
- CFS Note: CFS method was performed on WEKA, with ARFF files generated through **Final_models/toWeka_all.py**, from train kfold data.
- Stepwise - **stepwise_ALL.py**


# 4. Grid_NARX

Hyperparameter tuning through Grid Search, with 5-fold cv, for each feature selection method.

**GridResults.py** Plots gridsearch results (mean and standard deviation). Also stores the best hyperaparameter for each target feature and feature sleection.
**RankingGrid.py** Displays results obtained at *gridresults* as ranks, among the 4 feature selecion methods 

# 5. Train_Final_Models

Train final models, for hyperparameter combinations that yielded best performance, for each feature selection methos. Saves files of the ANN structure (structure - .json and weights - .h5), as well as normalization parameters (MinMAX) applied to the training set for subsequent use in the test set, for results evaluations.

In order to train the final models, mutual information and partial autocorrelation analysis and feature selection methods need to be re-run on the complete training set.


# 6. Inject_and_eval

Uses fault_inserter module to inject faults based on the config files (scenario_stuckat,scenario_drift,etc) at the test set. After fault injection, each model saved during the last stage is used for target feature prediction and residual generation. Different moving average are applied on the residuals, and different thresholds are applied for fault classification.

Evaluation metrics (precision, recall, f-score, etc.) are generated according to the fault injection parameters (e.g fault duration, magnitude, fault type, number of faults), the detector (moving average, threshold), feature selection (cfs,sw,all,rf) and target feature (GYROX,GYROY,GYROZ).
