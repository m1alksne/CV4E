Current repository for using opensoundscape package to train CNN to classify blue whale A, B, and D calls.
labeled data folder contains "train", "validation" and "test" csv's for inputting to model
to make these files, run modify_annotations.py and then make_Dataset.py
run train_oss_backup.py to train network. And then test and predict scripts for running model on new data.
eval_plots.ipynb makes predictions on test data and then plots true and false postives. and makes histogram of scores. 
