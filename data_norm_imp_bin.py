"""
RESEARCH PROJECT:
Prediction of Global Navigation Satellite System Positioning Errors with Guarantees
Main feature selector and random forest program
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random

"Change ABS_DIR to local address with synthetic_data.csv"
ABS_DIR = '/main_folder/folder/subfolder/'

#DESCRIPTION: Creates Normalization method based on training dataset and string input. Normalization can be minmax or zscore
#INPUT:
# df: input Dataframe
#OUTPUT:
# new_df: output Dataframe with normalized features
# normalization: Dictionary containing parameters for test data normalization
def create_normalization(df, normalizationtype):
	
	df = df.copy()
	normalization={}
	
	for id in df:
		
		if 'constellation' in id or 'amb_type' in id or 'tracking' in id:
			
			df[id].astype('category')
			continue
		
		if normalizationtype == 'minmax':     
			  
			min_ = df[id].astype(float).min()
			max_ = df[id].astype(float).max()
			normalization[id] = ('minmax', min_, max_)
			
		elif normalizationtype == 'zscore':
			
			mean = df[id].mean()
			std = df[id].std()
			normalization[id] = ('zscore', mean, std)
	            
	new_df = apply_normalization(df, normalization)
	return new_df, normalization

#DESCRIPTION: Applies Normalization based on normalization information input
#INPUT:
# test_data: Dataframe with the corresponding testing instances
# normalization: Dictionary containing parameters for test data normalization
#OUTPUT:
# test_data_norm: output Dataframe with normalized features for the test dataset
"Applies Normalization based on normalization information input"
def apply_normalization(test_data,normalization='minmax'):
	
	"Defines a copy of the input data frame"
	test_data_norm = test_data.copy()

	"Cycle through all columns"
	for i in test_data_norm.columns:
		
		if i == 'expected_total_error':
			
			continue
		
		else:
		
			"If the column is different from Constellation, amb_type, tracking"
			if 'constellation' in i or 'amb_type' in i or 'tracking' in i:
				
				continue
			
			"Verify if the normalization is zscore or minmax"
			if normalization[i][0] == 'minmax':
		
				"Picks the minimum and maximum value and normalizes the test instance"
				min_value = normalization[i][1]
				max_value = normalization[i][2]
				test_data_norm[i] = [min(max((x - min_value)/(max_value-min_value),0),1) if max_value != min_value else 0 for x in test_data_norm[i]]
			
			else:
				
				"Picks the mean and standard deviation value and normalizes the test instance"
				mean_value = normalization[i][1]
				std_dev_value = normalization[i][2]
				test_data_norm[i] = test_data_norm[i].apply(lambda x: (x - mean_value)/std_dev_value)	
		
	return test_data_norm


#DESCRIPTION: Creates Imputation method based on training dataset
#INPUT:
# train_data: Dataframe with the corresponding training instances
#OUTPUT:
# train_data_imp: output Dataframe with imputed features
# imputation: Dictionary containing parameters for test data imputation
def create_imputation(train_data):
	
	"Defines a copy of the input data frame"
	train_data_imp = train_data.copy()
	
	"Defines an empty dictionary"
	imputation = dict()

	"Cycle through all columns"
	for i in train_data_imp.columns:
		
		train_data_imp[i] = train_data_imp[i].astype('category')
		
		"Checks the data type of the column in the dataset (if int or float)"
		if str(train_data_imp[i].dtype) == 'int64' or str(train_data_imp[i].dtype) == 'float64':
			
			"Calculates the average of the columns and replaces the values in the data frame (if int or float)"
			mean = train_data_imp[i].mean()
			
			if np.isnan(mean):
				
				mean = 0				

			else:
				
				mean = mean			

			train_data_imp[i].fillna(mean,inplace=True)
			imputation[str(i)] = mean
		
		else:
			
			"Calculates the mode of the columns and replaces the values in the data frame (if object)"
			mode = train_data_imp[i].mode()[0]
			
			"If mode is empty (all missing values)"
			if (mode == []):
				
				mode = train_data_imp[i].cat.categories[0]

			train_data_imp[i].fillna(mode,inplace=True)		
			imputation[str(i)] = mode				
			
	return train_data_imp, imputation
	

#DESCRIPTION: Applies Imputation to dataset input and imputation entered as input
#INPUT:
# test_data: Dataframe with the corresponding testing instances
# imputation: Dictionary containing parameters for test data normalization
#OUTPUT:
# test_data_imp: output Dataframe with imputed features for the test dataset
def apply_imputation(test_data,imputation):
	
	"Defines a copy of the input data frame"
	test_data_imp = test_data.copy()	
	
	"Cycle through all columns"
	for i in test_data_imp.columns:
		
		"If the column is different from ID or Class"
		if i != "expected_total_error":	
		
			"Picks the imputation value and fills missing values"
			imp_value = imputation[i]
			test_data_imp[i].fillna(imp_value,inplace=True)
	
	return test_data_imp


#DESCRIPTION: Creates bins for the numerical variables
#INPUT:
# df: Dataframe with the corresponding training instances
# nobins: Number of bins to bin every feature. Default value = 10 (not relevant for regression algorithms)
# bintype: Select the binning amount of samples. Default value = 'equal-width'
#OUTPUT:
# df: output Dataframe with binned features
# binning: Dictionary containing parameters for test data binning
"Creates bins for the numerical variables"
def create_bins (df, nobins=10, bintype = 'equal-width'):
	
    df = df.copy()
    binning = {}
	
    for id in df:
		
        if id in ['expected_total_error']:
			
            continue
		
        if str(df[id].dtype) == 'float64' or str(df[id].dtype) == 'int32':
			
            if bintype == "equal-width":
				
                res, bins = pd.cut(df[id], nobins, labels = False, retbins = True, duplicates = 'drop')
                df[id] = df[id].astype('category')
                df[id] = res            
            
            elif bintype == "equal-size":
				
                res, bins = pd.qcut(df[id], nobins, labels = False, retbins = True, duplicates = 'drop')
                df[id] = df[id].astype('category')
                df[id] = res 
            
            bins[0] = -np.inf
            bins[len(bins)-1] = np.inf
            
            binning[id] =bins   
    
    return df, binning


#DESCRIPTION: Applies the binning to the numerical features
#INPUT:
# df: Dataframe with the corresponding testing instances
# binning: Dictionary containing parameters for test data binning
#OUTPUT:
# df: output Dataframe with binned features for the test dataset
def apply_bins (df, binning):
	
    for id in df:
		
        if id in ['expected_total_error']:
			
            continue
		
        if str(df[id].dtype) == 'float64' or str(df[id].dtype) == 'int32':
			
            df[id] = df[id].astype('category')
            binned = pd.cut(df[id],binning[id],labels=range(len(binning[id])-1))
            cats = range(len(binning[id])-1)
            df[id] = binned
			
    return df


#DESCRIPTION: Creates a one hot encoding for an input data frame
#INPUT:
# data: Dataframe with the corresponding training instances
#OUTPUT:
# onehot_data: output Dataframe with one-hot encoded features
# onehot: Dictionary containing parameters for test data one-hot encoding
def create_one_hot(data):
		
	"Defines a copy of the input data frame"
	onehot_data = data.copy()	
	
	"Defines an empty dictionary"
	onehot = dict()	
	
	"Cycle through all columns to apply one-hot encoding"	
	for i in onehot_data.columns:		
	
		"If the column is 'expected_total_error' do not apply one-hot encoding"
		if i == 'expected_total_error':
			
			continue
		
		else:
		
			"Checks the data type of the column in the dataset (if object)"
			if str(onehot_data[i].dtype) == 'category':
				
				"Gets all possible unique values from the selected column and stores them in dict"
				column_unique_values = onehot_data[i].unique()
				onehot[i] = column_unique_values
				
				"Uses Pandas get_dummies() method to create a one-hot encoding of the desired column"
				column_one_hot = pd.get_dummies(onehot_data[i], prefix=i)
				
				"Change all columns of column_one_hot to float"
				for j in column_one_hot.columns:
					
					column_one_hot[j] = column_one_hot[j].astype("float") 		
				
				"Place the one_hot_column in the one hot data set"
				onehot_data = pd.concat([onehot_data,column_one_hot], axis=1)
				
				"Eliminates the previous column selected"
				onehot_data.drop([i], axis=1, inplace=True)
					
	return onehot_data, onehot


#DESCRIPTION: Applies the one hot encoding for an input data frame
#INPUT:
# test_data: Dataframe with the corresponding testing instances
# onehot: Dictionary containing parameters for test data one-hot encoding
#OUTPUT:
# one_hot_test: output Dataframe with one-hot encoded features for the test dataset
def apply_one_hot(test_data,onehot):
			
	"Defines a copy of the input data frame"
	one_hot_test = test_data.copy()	
	
	"Cycle through all columns to apply one-hot encoding"	
	for i in one_hot_test.columns:		
		
		if i.startswith('constellation') or i.startswith('amb') or i.startswith('tracking'):
			
			one_hot_test[i] = one_hot_test[i].astype('object')
		
		"Checks the data type of the column in the dataset (if object)"
		if str(one_hot_test[i].dtype) == 'object':
			
			"Get the possible values corresponding to the dictionary"
			for j in onehot[i]:
				
				"Compares the dict object to the column objects and stores in auxiliary variable as float"
				aux_column = one_hot_test[i].apply(lambda x: float(x == j))
				
				"Place the aux_column in the one_hot_test data set"
				new_column_name = str(i)+'_'+str(j)
				one_hot_test[new_column_name] = aux_column
	
			"Eliminates the previous column selected"
			one_hot_test.drop([i], axis=1, inplace=True)
		
	return one_hot_test


#DESCRIPTION: Calls data in pandas DataFrame structure from
#INPUT:
# file_key: File name to be read from database
#OUTPUT:
# df: Structured Dataframe from file
def call_dataframe(file_key):

	if file_key == 'synthetic':

		df = pd.read_csv(ABS_DIR+'synthetic_data.csv')

	else:

		raise Exception('the file_key should be: synthetic')
	
	return df


#DESCRIPTION: Removes label from dataset (expected total error)
#INPUT:
# df: file: File name to be read from database
#OUTPUT:
# df: Dataframe without the label
def remove_features_label(df):

	"Eliminates expected_total_error"
	columns_selected = [i for i in df.columns if not i.startswith('expected_total_error')]
	df = df[columns_selected]

	return df


#DESCRIPTION: Creates a split in the data frame to apply normalization, imputation and binning mapping from the training data
#INPUT:
# df: Dataframe obtained from a given file
#OUTPUT:
# test_list: List of Dataframes (folds) with the corresponding testing instances
# train_list: List of Dataframes (folds) with the corresponding training instances
def create_random_split(df,nofolds):
	
	df1 = df.sample(frac=1)
	df = df.sample(frac=1).reset_index(drop=True)
	test_list = list()
	train_list = list()
	indexes = range(len(df))
	
	"Calculates the number of instances per fold"
	number_test_indexes = round(len(indexes)/nofolds)
	
	"Cycle to grab the data from the dataframe"
	for i in range(nofolds):
		
		"If on the last fold, take all remaining data in the dataframe"
		if i == nofolds - 1:
			
			test_indexes_sub = indexes[number_test_indexes*i:]
			
		else:
			
			test_indexes_sub = indexes[number_test_indexes*i:number_test_indexes*(i+1)]

		train_indexes_sub = [x for x in indexes if x not in test_indexes_sub]

		test_list.append(df1.iloc[test_indexes_sub])
		train_list.append(df1.iloc[train_indexes_sub])
						
	return test_list, train_list


#DESCRIPTION: Filter out features which have a single value (no difference per target value or information added)
#INPUT:
# data: Dataframe containing the training instances
#OUTPUT:
# data: Dataframe without features not providing any relevant information
# eliminated_features: List of features eliminated.
def filter_useless(data):
	
	eliminated_features = list()
	
	for i in data.columns:
	
		column = data[i]
		unique = column.unique()
	
		if len(unique) == 1:
			
			data.drop([i], axis=1, inplace=True)
			eliminated_features.append(i)
	
	return data,eliminated_features


#DESCRIPTION: Filter out features which have a single value (no difference per target value or information added) on the test dataset
#INPUT:
# data: Dataframe containing the testing instances
#OUTPUT:
# data: Dataframe without features not providing any relevant information
"Filter out features which have a single value (no difference per target value or information added) in the test set"
def filter_useless_test(data,eliminate):
	
	data.drop(eliminate, axis=1, inplace=True)
	
	return data


#DESCRIPTION: Calls all previous methods to normalize, impute, one-hot encode and filter features not relevant.
#INPUT:
# file_name: File name to be read from database
#OUTPUT:
# one_hot_norm_imp_train_df: Dataframe containing split training data into folds, normalized, imputed and one-hot encoded
# one_hot_norm_imp_test_df: Dataframe containing split training data into folds, normalized, imputed and one-hot encoded
# eliminated: Filtered features
def imp_norm_bin(file_name,nofolds):
	
	print("Starting reading and preparing "+str(file_name))
	print("Setup step 1. Obtaining data from "+str(file_name)+" as a structured data frame...")
	df = call_dataframe(file_name)
	
	print("3. Creates a random split instead for K-Fold")
	test_df,train_df = create_random_split(df,nofolds)
	
	print("Setup step 4. Preparing the folds in imputation, normalization, one-hot encoding and filtering features...")
	one_hot_norm_imp_train_df = list(range(nofolds))
	one_hot_norm_imp_test_df = list(range(nofolds))
	eliminated = list(range(nofolds))
	
	for i in range(nofolds):
		
		print("1. Picking the training dataset and applying imputation...")
		imp_train_df,imp_map = create_imputation(train_df[i])
		
		print("2. Picking the training dataset and applying normalization...")
		norm_imp_train_df,norm_map = create_normalization(imp_train_df,normalizationtype = 'minmax')

		print("3. Creating one hot encoding...")
		one_hot_norm_imp_train_df[i], one_hot = create_one_hot(norm_imp_train_df)

		print("4. Applying imputation, normalization, and binning to test data...")
		imp_test_df = apply_imputation(test_df[i],imp_map)
		norm_imp_test_df = apply_normalization(imp_test_df,norm_map)
		one_hot_norm_imp_test_df[i] = apply_one_hot(norm_imp_test_df,one_hot)
		
		print("5. Filtering useless features...")
		one_hot_norm_imp_train_df[i],eliminated[i] = filter_useless(one_hot_norm_imp_train_df[i])
		one_hot_norm_imp_test_df[i] = filter_useless_test(one_hot_norm_imp_test_df[i],eliminated[i])

	return one_hot_norm_imp_train_df,one_hot_norm_imp_test_df,eliminated