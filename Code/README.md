1) All the files in this folder must be downloaded into the same directory with consistent naming for the code to run.
2) The Code file contains data preprocessing steps, two algorithms and the predicted test/validation sale prices for the two algorithms.
3) CNN gives the best r2 of 0.875 on the local test set.
4) The r2 on the actual unseen validation/test set once the model is trained with the whole available training sample/full dataset must be around 88-89.
5) The salepricedata.csv dataset is the same as the train dataset available in original_files folder. This is our master dataset as the original test set does not have the target variable in it.
6) The test_val.csv is same as the test.csv availble in original_files. This is not used for model tuning. It is used only for final prediction.
