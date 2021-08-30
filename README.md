# Comment_Probe-Source - Classification of Comments

Part 1 - Classification of comments 
Step 1 - Data Generation
1.a - Training Features generation (X feature values)

Working folder - GENERATE_FEATURES/CODES FINAL
File to run - run_script_feb_2020.py (Run this file for each project)
NOTE - Please make sure you have a folder named CSV in ‘CODES FINAL’ before running this. Also, make sure the StanfordParser folder is present in CODES FINAL.
Parameters to update in file before running - 
FILES_TO_RUN = [ ] (a list of all code files for which we have to run)
BASE_DIR = Path to base directory (relative to ‘CODES FINAL’) such that joining BASE_DIR + file from FILES_TO_RUN gives the path to code file
PROBLEM_DOMAIN_FILE = Path to problem domain file
Please note that the code logic in for loop might have to be changed depending on the way xml files and code files have been organised. The current code is based on the most recent way we arrange those files.
Output files will be in the CSV folder.

Command to run - sudo python2 run_script_feb_2020.py


1.b - Ground truth labels generation (Y values) and creating a merged training file sheet
Working folder - ML/
Prerequisite - Maintain this directory structure inside ML/
ML
	ANALYSIS
DATA
ANNOTATED
GENERATED
	TRAIN
Steps to run - 
1. Generate Y labels based on rules for all annotations sheets.
File to run - GetLabelsFromAnnotatedClasses.ipynb (Run this file for each Annotations excel sheet)
Input Parameters to update before running - 
FILE_PATH - path to annotations excel file
OUTPUT_FILE_PATH - path to output file containing Generated Y labels.
ANNOTATION_CLASS_START = Index at which Annotation class start in data
ANNOTATION_CLASS_NUM - Number of annotation classes (31 always for our use)
Please note that in the 6th cell of the jupyter notebook, I print all the index, value pairs present in our annotations file. You can look at that to determine what value to set in the field ANNOTATION_CLASS_START.

2. Combine entries from CSVs obtained in step 1.a with output generated from step 1.b.1 to generate X, Y and Z (comment file, comment text) files for each annotated file. (Run this file for each Annotations excel sheet)
First put the CSV folder obtained in step 1.a inside ML/DATA/
File to run - PrepareTrainingData.ipynb
Input Parameters to update before running -
FEATURES_DIR - path to file where CSVs are present corresponding to the project in its annotations sheet (output of step 1.a)
ANNOTATIONS_FILE - Output file from previous step (1.b.1)
PROJECT_NAME - Name of this project as it appears in CSV files.
AUTHOR_NAME - Who annotated it. Please make sure different annotations sheets from the same project should have different author names or otherwise they would be overwritten.
OUTPUT_DIR - Output directory for X, Y and Z files. 
OUTPUT_FILE_NAME - (NO NEED TO MODIFY, generated automatically based on above parameters)
INTUITIVE_INDEX - Index in data at which Intuitive score is present.
CALCULATED_INDEX - Index in data at which Calculated score is present.
COMMENT_TYPE_INDEX - Index in data at which Comment type labels are present.
Please note that in the 4th cell of the jupyter notebook, I print all the index, value pairs present in our annotations file. You can look at that to determine what value to set for INTUITIVE_INDEX, CALCULATED_INDEX and COMMENT_TYPE_INDEX.
In the 7th cell, we map the filenames present in annotation_file with the filename of CSV files to perform matching. This logic is mostly heuristic based and because of differences in the way annotations file and generated files could have their names, you might need to change this logic. For the ease of user, after matching is done, I finally print two sets of values named NOT FOUND and FOUND telling which files have been found and which not found. Use that information for finding and rectifying cases where a match is not found.

3. Merge X, Y and Z files from various annotations file into 1 single training file.
File to run - merge_files.ipynb
Input Parameters to update before running -
DATA_DIR - Same as OUTPUT_DIR from previous step (1.b.2)
DATA_FILES - List of OUTPUT_FILE_NAMES from various runs of previous step for which we have to merge.
OUTPUT_FILE - Names of Output file that we want.
Output Generated - 2 files. First - OUTPUT_FILE+”.csv” which contains the X and Y values. Second - “Z_”+OUTPUT_FILE+”.csv” which contains the corresponding Z values.


Step 2 - Model Training 

Working Folder - ML_EXPERIMENTS
Contains the list of all experiments. The experiment that has the final model is exp5. 
Command to run - python LSTM_endtoend_singleLabel.py
Some important command line arguments - 
TEST - Just for testing purposes. Runs the code on 20 comments to check if everything is working fine.
METRICS - To just report the metrics (skip training).
FEATS - Just extract the features sheet (skip training).
Input files needed - 
Features sheet - 'ML_DATASHEETS/LATEST_FEATURES_cal.csv' (output of Step 1.b.3). Parameter in code - FEATS (line 99).
Z sheet - 'ML_DATASHEETS/Z_LATEST_FEATURES_cal.csv' (output of step 1.b.3). Parameter in code - Z (line 96).
Path to embeddings model.
Output files generated - 
All_features sheet - 'ML_DATASHEETS/EXTRACTED/20kx220.csv'
Manual features + ground truth label + Predicted Label - 'ML_DATASHEETS/EXTRACTED/all_results.csv'
Output Model - ‘MODELS_NEW/model_all_fold_4.h5’


Part 2 - Analysis
Analysis 1 - Find Mismatches between Intuitive and Calculated Labels
Code File - ML/find mismatches.ipynb
Input Parameters to update before running - 
ANNOTATIONS_FILE - Name of Annotations sheet for which to generate analysis
INTUITIVE_INDEX - Index where Intuitive label is present in annotations sheet.
CALCULATED_INDEX - Index where Calculated label is present in annotations sheet.
Output Generated - ‘ANALYSIS/DIFF/<ANNOTATIONS_FILE name>’

Analysis 2 - Bias experiment
Step 1 - 
Code File - ML/find mismatches-bias.ipynb
Input Parameters to update before running - 
ANNOTATIONS_FILE1, ANNOTATIONS_FILE2 - Pairs of annotation sheets for which to run bias experiment
INTUITIVE_INDEX1, INTUITIVE_INDEX2 - Index where Intuitive label is present in respective annotations sheet.
CALCULATED_INDEX1, CALCULATED_INDEX2 - Index where Calculated label is present in respective annotations sheet.
CATEGORIES_START1, CATEGORIES_START2 - Index where Annotation categories start in respective annotations sheet.
Output File - ‘ANALYSIS/BIAS/<ANNOTATION_FILE1>__<ANNOTATION_FILE2>’

Step 2 - Segregate based on all permutations of annotations.
Code file - ML/Segregate Bias Entries.ipynb
Just run the notebook.
Output - ANALYSIS/BIAS/SEGREGATED
