"""
RESEARCH PROJECT:
Prediction of Global Navigation Satellite System Positioning Errors with Guarantees
Main feature selector and random forest program
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data_norm_imp_bin as df_pre
import scipy as sci
import statsmodels.api as sm
import itertools as it
import collections as col
import time as time
import xlsxwriter
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

"------------------------- LINEAR REGRESSION FEATURE ANALYSIS -------------------------"


#DESCRIPTION: Calls the imputation, normalization and binning method from data_norm_imp_bin.py
#INPUT:
# file: File name to be read from database
# nofolds: Number of folds to use for cross-validation
#OUTPUT:
# pre_train_df: Dataframe containing nofolds Dataframes, each with the corresponding training instances
# pre_test_df: Dataframe containing nofolds Dataframes, each with the corresponding testing instances
# eliminated_f: list of filtered features in preprocessing
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
# target_test: Dataframe containing nofolds Dataframes, each with the corresponding testing instances target
# features: list of strings with the feature names to be considered for linear regression
def read_data_df(file,nofolds):

	pre_train_df,pre_test_df,eliminated_f = df_pre.imp_norm_bin(file,nofolds)

	features = list(range(nofolds))
	target = list(range(nofolds))
	target_test = list(range(nofolds))

	for i in range(nofolds):

		target[i] = pre_train_df[i]['expected_total_error']
		target_test[i] = pre_test_df[i]['expected_total_error']
		pre_train_df[i].drop(['expected_total_error'], axis=1, inplace=True)
		pre_test_df[i].drop(['expected_total_error'], axis=1, inplace=True)
		features[i] = pre_train_df[i].columns.tolist()

	return pre_train_df,pre_test_df,eliminated_f,target,target_test,features


#DESCRIPTION: Estimates baseline linear regression
#INPUT:
# features: list of strings with the feature names to be considered for linear regression
# pre_train_df: Dataframe containing nofolds Dataframes, each with the corresponding training instances
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
#OUTPUT:
# p_value_dict: Dictionary containing the p-values for all features
# r_square_dict: Dictionary containing the coefficient of determination R² for all features
# params_dict: Dictionary containing the linear coefficients for all features
def baseline_lin(features,pre_train_df,target):

	x = pre_train_df[features]
	x = sm.add_constant(x)
	y = target
	model = sm.OLS(y,x).fit()
	r_square_dict[str(features)] = model.rsquared
	params_dict = model.params
	p_value_dict = model.pvalues

	return p_value_dict,r_square_dict,params_dict


#DESCRIPTION: Estimates expert linear regression
#INPUT:
# pre_train_df: Dataframe containing nofolds Dataframes, each with the corresponding training instances
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
#OUTPUT:
# r_square_dict: Dictionary containing the coefficient of determination R² for all features
# params_dict: Dictionary containing the linear coefficients for all features
"Method that gathers only the important features according to experts"
def expertSelection(pre_train_df,target):

	exp_train_df = pd.DataFrame(index=pre_train_df.index)
	temp_train_df = pd.DataFrame()

	relevant_features = ['cycle_slip','multipath','amb_type','cno','pdop','correction_covariance','innovation','used','lsq_residuals','elevation','tracking_type','prediction_covariance','nr_used_measurements','difference']

	for i in relevant_features:

		columns_selected = [x for x in pre_train_df.columns if x.startswith(i)]
		temp_train_df = pre_train_df[columns_selected]
		exp_train_df = pd.concat([exp_train_df,temp_train_df.reindex(exp_train_df.index)],axis=1)

	x = exp_train_df
	x = sm.add_constant(x)
	y = target
	model = sm.OLS(y,x).fit()
	r_square_dict[str(relevant_features)] = model.rsquared
	params_dict = model.params

	return r_square_dict,params_dict


#DESCRIPTION: Estimates backward linear regression recursively
#INPUT:
# features: list of strings with the feature names to be considered for linear regression
# pre_train_df: Dataframe containing nofolds Dataframes, each with the corresponding training instances
# r_square_dict: Dictionary containing the coefficient of determination R² for all features
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
# deleted_features: list of features considered irrelevant
#OUTPUT:
# p_value_dict: Dictionary containing the p-values for all features
# r_square_dict: Dictionary containing the coefficient of determination R² for all features
# params_dict: Dictionary containing the linear coefficients for all features
# deleted_features: list of features considered irrelevant
"Applies backward selection method"
def backward_selection(features,pre_train_df,r_square_dict,target,deleted_features):

	x = pre_train_df[features]
	x = sm.add_constant(x)
	y = target
	model = sm.OLS(y,x).fit()
	r_square_dict[str(features)] = model.rsquared
	params_dict = model.params
	p_value_dict = model.pvalues
	ordered_p_value = np.sort(p_value_dict)
	ordered_p_value = ordered_p_value[::-1]
	p_value_feature_max = p_value_dict[p_value_dict == ordered_p_value[0]].index[0]

	if p_value_feature_max == 'const':

		p_value_feature_max = p_value_dict[p_value_dict == ordered_p_value[1]].index[0]

	if sum(p_value_dict >= 0.05) == 0:

		return p_value_dict,r_square_dict,params_dict,features,deleted_features

	features.remove(p_value_feature_max)
	deleted_features.append(p_value_feature_max)

	return backward_selection(features,pre_train_df,r_square_dict,target,deleted_features)


#DESCRIPTION: Estimates stepwise linear regression recursively
#INPUT:
# features: list of strings with the feature names to be considered for linear regression
# pre_train_df: Dataframe containing nofolds Dataframes, each with the corresponding training instances
# p_value_dict_added: Dictionary containing the p-values for all features added as relevant
# r_square_dict_added: Dictionary containing the coefficient of determination R² for all features added as relevant
# params_dict_added: Dictionary containing the linear coefficients for all features added as relevant
# added_features: list of features added as relevant
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
#OUTPUT:
# p_value_dict_added: Dictionary containing the p-values for all features added as relevant
# r_square_dict_added: Dictionary containing the coefficient of determination R² for all features added as relevant
# params_dict_added: Dictionary containing the linear coefficients for all features added as relevant
# added_features: list of features added as relevant
"Applies stepwise selection method"
def stepwise_selection(features,pre_train_df,p_value_dict_added,r_square_dict_added,params_dict_added,added_features,target):

	p_value_min = 1
	p_value_feature_min_sel = 'const'
	p_value_dict_sel = []
	ordered_p_value_sel = []

	for i in features:

		x = pre_train_df[i]
		x = sm.add_constant(x)
		y = target
		model = sm.OLS(y,x).fit()
		p_value_dict = model.pvalues
		ordered_p_value = np.sort(p_value_dict)
		p_value_feature_min = p_value_dict[p_value_dict == ordered_p_value[0]].index[0]

		if p_value_feature_min == 'const':

			p_value_feature_min = [x for x in p_value_dict.keys() if x != 'const'][0]

		if p_value_dict[p_value_feature_min] < p_value_min:

			p_value_min = p_value_dict[p_value_feature_min]
			p_value_feature_min_sel = p_value_feature_min
			p_value_dict_sel = p_value_dict
			ordered_p_value_sel = np.sort(p_value_dict_sel)

	if p_value_feature_min_sel == 'const' and features != []:

		p_value_feature_min_sel = p_value_dict_sel[p_value_dict_sel == ordered_p_value_sel[1]].index[0]

	if len(p_value_dict_sel) != 0:

		if p_value_dict_sel[p_value_feature_min_sel] < 0.05:

			features.remove(p_value_feature_min_sel)
			added_features.append(p_value_feature_min_sel)

			x_added = pre_train_df[added_features]
			x_added = sm.add_constant(x_added)
			y = target
			model_added = sm.OLS(y,x_added).fit()
			r_square_dict_added[str(added_features)] = model_added.rsquared
			params_dict_added = model_added.params
			p_value_dict_added = model_added.pvalues

			if sum(p_value_dict_added >= 0.05) >= 1:

				added_features.remove(p_value_feature_min_sel)
				x_added = pre_train_df[added_features]
				x_added = sm.add_constant(x_added)
				y = target
				model_added = sm.OLS(y, x_added).fit()
				r_square_dict_added[str(added_features)] = model_added.rsquared
				params_dict_added = model_added.params
				p_value_dict_added = model_added.pvalues

			return stepwise_selection(features,pre_train_df,p_value_dict_added,r_square_dict_added,params_dict_added,added_features,target)

	return p_value_dict_added,r_square_dict_added,params_dict_added,added_features


#DESCRIPTION: Estimates stepwise linear regression recursively
#INPUT:
# features: list of strings with the feature names to be considered for linear regression
# pre_train_df: Dataframe with the corresponding training instances
# p_value_dict_added: Dictionary containing the p-values for all features added as relevant
# r_square_dict_added: Dictionary containing the coefficient of determination R² for all features added as relevant
# params_dict_added: Dictionary containing the linear coefficients for all features added as relevant
# target: Dataframe with the corresponding training instances target
# added_features: list of features added as relevant
#OUTPUT:
# p_value_dict_added: Dictionary containing the p-values for all features added as relevant
# r_square_dict_added: Dictionary containing the coefficient of determination R² for all features added as relevant
# params_dict_added: Dictionary containing the linear coefficients for all features added as relevant
# features: list of strings with the feature names to be considered for linear regression
# added_features: list of features added as relevant
"Applies forward selection method"
def forward_selection(features,pre_train_df,p_value_dict_added,r_square_dict,params_dict_added,target,added_features):

	p_value_min = 1
	p_value_feature_min_sel = 'const'
	p_value_dict_sel = []
	ordered_p_value_sel = []

	for i in features:

		x = pre_train_df[i]
		x = sm.add_constant(x)
		y = target
		model = sm.OLS(y,x).fit()
		p_value_dict = model.pvalues
		ordered_p_value = np.sort(p_value_dict)
		p_value_feature_min = p_value_dict[p_value_dict == ordered_p_value[0]].index[0]

		if p_value_feature_min == 'const':

			p_value_feature_min = [x for x in p_value_dict.keys() if x != 'const'][0]

		if p_value_dict[p_value_feature_min] < p_value_min:

			p_value_min = p_value_dict[p_value_feature_min]
			p_value_feature_min_sel = p_value_feature_min
			p_value_dict_sel = p_value_dict
			ordered_p_value_sel = np.sort(p_value_dict_sel)

	if p_value_feature_min_sel == 'const' and features != []:

		p_value_feature_min_sel = p_value_dict_sel[p_value_dict_sel == ordered_p_value_sel[1]].index[0]

	if len(p_value_dict_sel) != 0:

		if p_value_dict_sel[p_value_feature_min_sel] < 0.05:

			features.remove(p_value_feature_min_sel)
			added_features.append(p_value_feature_min_sel)

			x_added = pre_train_df[added_features]
			x_added = sm.add_constant(x_added)
			y = target
			model_added = sm.OLS(y,x_added).fit()
			r_square_dict_added[str(added_features)] = model_added.rsquared
			params_dict_added = model_added.params
			p_value_dict_added = model_added.pvalues

			return forward_selection(features,pre_train_df,p_value_dict_added,r_square_dict_added,params_dict_added,target,added_features)

	return p_value_dict_added,r_square_dict_added,params_dict_added,features,added_features


#DESCRIPTION: Estimates Lasso linear regression
#INPUT:
# features: list of strings with the feature names to be considered for linear regression
# pre_train_df: Dataframe with the corresponding training instances
# target: Dataframe with the corresponding training instances target
# alpha: weight of the L1 regularization term in Lasso (LassoCV is used instead to find best L1 weight)
#OUTPUT:
# r_square_dict: Dictionary containing the coefficient of determination R² for all features
# params_dict: Dictionary containing the linear coefficients for all features
def lasso(features,pre_train_df,target,alpha = 0.001):

	lasso_model = linear_model.LassoCV(fit_intercept=True,tol=1e-05)
	x = pre_train_df[features]
	y = target

	lasso_model.fit(x,y)

	r_square_dict[str(features)] = lasso_model.score(x,y)
	params = lasso_model.coef_
	intercept = lasso_model.intercept_
	params_dict = pd.Series(params,index=features)
	params_dict = params_dict[abs(params_dict) > 0.0000000001]
	pd_inter = pd.Series(intercept, index=['const'])
	params_dict = params_dict.append(pd_inter)

	return r_square_dict,params_dict


#DESCRIPTION: Predicts values for the test data and coeficients given
#INPUT:
# test_df: Dataframe containing nofolds Dataframes, each with the corresponding testing instances
# coefficients: Dataframe containing nofolds Dataframes, each with the corresponding coefficients for every feature
# nofolds: Number of folds to use for cross-validation
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
#OUTPUT:
# prediction_ind: Dataframe containing the prediction and fold number for every instance
def predict(test_df,coefficients,nofolds,target):

	col = ['fold_no','prediction','expected_total_error']
	prediction_ind = pd.DataFrame(columns=col)

	for i in range(nofolds):

		for k in test_df[i].index.tolist():

			prediction_value = 0
			coefficients_list_i = coefficients[i].index.tolist()

			for j in coefficients_list_i:

				if j == 'const':

					prediction_value += coefficients[i][j]

				else:

					instance_feature_value = test_df[i].loc[k,j]
					coefficient_value = coefficients[i][j]
					prediction_value += instance_feature_value*coefficient_value

			prediction_ind = pd.concat([prediction_ind,pd.DataFrame({'fold_no':i,'prediction':prediction_value,'expected_total_error':target[i].loc[k]},index=[k])])

	return prediction_ind


#DESCRIPTION: Calculates the RMSE of a linear prediction vs. the target value of positioning error
#INPUT:
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
# prediction: Dataframe containing the prediction and fold number for every instance
# nofolds: Number of folds to use for cross-validation
#OUTPUT:
# errors: Dataframe containing RMSE for every fold.
def rmse_lin(target,prediction,nofolds):

	col = ['fold_no','error']
	errors = pd.DataFrame(columns=col)

	for i in range(nofolds):

		fold_predictions = prediction.loc[prediction['fold_no'] == i]
		del fold_predictions['fold_no']
		error = np.sqrt(sum((target[i] - fold_predictions['prediction'])**2)/len(target[i]))

		errors = errors.append({'fold_no':i,'error':error},ignore_index=True)

	return errors


#DESCRIPTION: Calculates the RMSE of a random forest prediction vs. the target value of positioning error
#INPUT:
# target: Dataframe containing nofolds Dataframes, each with the corresponding training instances target
# prediction: Dataframe containing the prediction and fold number for every instance
# nofolds: Number of folds to use for cross-validation
#OUTPUT:
# errors: Dataframe containing RMSE for every fold.
def rmse_rf(target,prediction,nofolds):

	col = ['fold_no','error']
	errors = pd.DataFrame(columns=col)

	for i in range(nofolds):

		fold_predictions = prediction[i]
		error = np.sqrt(sum((target[i] - fold_predictions)**2)/len(target[i]))

		errors = errors.append({'fold_no':i,'error':error},ignore_index=True)

	errors = errors.append({'fold_no':-1,'error':np.mean(errors['error'])},ignore_index=True)

	return errors


#DESCRIPTION: Auxiliary method that eliminates features not in the selected features
#INPUT:
# selected: Dataframe containing all the selected features
# train_data: Dataframe with the corresponding training instances
# test_data: Dataframe with the corresponding testing instances
#OUTPUT:
# train_data: Dataframe with the corresponding training instances with selected features
# test_data: Dataframe with the corresponding testing instances with selected features
def eliminateFeatures(selected,train_data,test_data):

	columns_selected = [i for i in train_data.columns if i in selected.index.tolist()]
	train_data = train_data[columns_selected]
	test_data = test_data[columns_selected]

	return train_data,test_data


#DESCRIPTION: Applies the Random Forest algorithm
#INPUT:
# train: Dataframe with the corresponding training instances
# target: Dataframe with the corresponding training instances target
# test: Dataframe with the corresponding testing instances
# target_test: Dataframe with the corresponding testing instances target
# ntrees: Number of trees in the random forest. Default value = 100
# crit: Criteria for split. Default value = 'mse'
# min_sample_split: Minimum number of node samples required for split. Default value = 10
# min_leaf_size: Minimum number of samples at a leaf. Default value = 1
# oob: Whether the forest uses OOB samples for R² estimation (not relevant for the code)
#OUTPUT:
# random_forest_feature_imp: Feature relevance of all features
# rf_prediction: Predicted values for every instance from the random forest
# rf_r_square: random forest r_square
# oob_prediction: prediction values for OOB instances
# all_trees_test_pred: Predicted values for every instance, from every single tree in the forest
def randomForest(train,target,test,target_test,ntrees = 100, crit = 'mse',min_sample_split = 10,min_leaf_size = 1,oob = True):

	regressor = RandomForestRegressor(n_estimators = ntrees, criterion = crit, min_samples_split = min_sample_split, min_samples_leaf = min_leaf_size, oob_score = oob)
	regressor.fit(train,target)
	random_forest_feature_imp = regressor.feature_importances_
	rf_prediction = regressor.predict(test)
	rf_r_square = regressor.score(test,target_test)

	oob_prediction = regressor.oob_prediction_
	all_trees_test_pred = np.array([tree.predict(test) for tree in regressor.estimators_])


	return random_forest_feature_imp,rf_prediction,rf_r_square,oob_prediction,all_trees_test_pred


#DESCRIPTION: Calculates the variance for the nonconformity measure
#INPUT:
# tree_predictions: Predicted values for every instance, from every single tree in the forest
# prediction: Predicted values for every instance from the random forest
# ntrees: Number of trees used
#OUTPUT:
# variance: Variance of all tree predictions with respect to the single forest prediction value
def varianceNonconformity(tree_predictions,prediction,ntrees):

	failed = 0

	try:

		len_instances = len(tree_predictions[0,:])

	except:

		failed = 1
		len_instances = len(prediction)

	finally:

		variance = [0]*len_instances

		for i in range(len_instances):

			if failed:
				variance[i] = np.var(prediction)
			else:
				variance[i] = np.var(tree_predictions[:,i])

	return variance


#DESCRIPTION: Nonconformity calculation 99.999%
#INPUT:
# target: Array with the corresponding training instances target
# prediction: Predicted values for every instance from the random forest
# variance: Variance of all tree predictions with respect to the single forest prediction value
# beta: Constant to avoid division by zero and control the sensitivity to variance
#OUTPUT:
# norm_nonconformity: list with the values of the nonconformity range
def normalizedNonconformityIntegrity(target,prediction,variance,beta = 0.01):

	length = len(target)
	length_c = min(round(0.99999*(length+1)),length-1)
	nonconformity = abs(target.array.astype('float') - prediction)
	nonconformity_sort = np.sort(nonconformity)
	alfa_c = nonconformity_sort[length_c]
	var_length = len(variance)
	norm_nonconformity = [0]*var_length

	for i in range(var_length):

		norm_nonconformity[i] = alfa_c/(variance[i]+beta)

	return norm_nonconformity


#DESCRIPTION: Nonconformity calculation 95%
#INPUT:
# target: Array with the corresponding training instances target
# prediction: Predicted values for every instance from the random forest
# variance: Variance of all tree predictions with respect to the single forest prediction value
# beta: Constant to avoid division by zero and control the sensitivity to variance
#OUTPUT:
# norm_nonconformity: list with the values of the nonconformity range
def normalizedNonconformity_95(target, prediction, variance, beta=0.01):
	length = len(target)
	length_c = min(round(0.95 * (length + 1)), length - 1)
	nonconformity = abs(target.array.astype('float') - prediction)
	nonconformity_sort = np.sort(nonconformity)
	alfa_c = nonconformity_sort[length_c]
	var_length = len(variance)
	norm_nonconformity = [0] * var_length

	for i in range(var_length):
		norm_nonconformity[i] = alfa_c / (variance[i] + beta)

	return norm_nonconformity


#DESCRIPTION: Nonconformity calculation 98%
#INPUT:
# target: Array with the corresponding training instances target
# prediction: Predicted values for every instance from the random forest
# variance: Variance of all tree predictions with respect to the single forest prediction value
# beta: Constant to avoid division by zero and control the sensitivity to variance
#OUTPUT:
# norm_nonconformity: list with the values of the nonconformity range
def normalizedNonconformity_98(target, prediction, variance, beta=0.01):
	length = len(target)
	length_c = min(round(0.98 * (length + 1)), length - 1)
	nonconformity = abs(target.array.astype('float') - prediction)
	nonconformity_sort = np.sort(nonconformity)
	alfa_c = nonconformity_sort[length_c]
	var_length = len(variance)
	norm_nonconformity = [0] * var_length

	for i in range(var_length):
		norm_nonconformity[i] = alfa_c / (variance[i] + beta)

	return norm_nonconformity


#DESCRIPTION: Confidence Interval calculation for random forest prediction
#INPUT:
# norm_nonconformity: list with the values of the nonconformity range
# prediction: Predicted values for every instance from the random forest
# variance: Variance of all tree predictions with respect to the single forest prediction value
# beta: Constant to avoid division by zero and control the sensitivity to variance
# target_test: Dataframe with the corresponding testing instances target
#OUTPUT:
# erri_ci_df: Dataframe with the error bars, confidence interval and the fraction of test instances in or out of the interval
def ci_rf(norm_nonconformity,prediction,variance,beta,target_test):

	length = len(prediction)
	ci = [0]*len(norm_nonconformity)
	erri = [0]*len(norm_nonconformity)
	inside = [0]*len(norm_nonconformity)
	col = ['erri','ci','inside']
	erri_ci_df = pd.DataFrame(columns=col)

	for i in range(length):

		erri[i] = norm_nonconformity[i]*(variance[i]+beta)
		ci[i] = [prediction[i]-erri[i], prediction[i]+erri[i]]

		if target_test.values[i] <= prediction[i]+erri[i] and target_test.values[i] >= prediction[i]-erri[i]:

			inside[i] = 1

		erri_ci_df = pd.concat([erri_ci_df, pd.DataFrame([{'erri': erri[i], 'ci': ci[i], 'inside': inside[i]}])])

	sum_inside = sum(erri_ci_df['inside'])
	fraction_inside = sum_inside/length

	erri_ci_df = pd.concat([erri_ci_df, pd.DataFrame([{'erri': 0, 'ci': 0, 'inside': fraction_inside}])])

	return erri_ci_df


#DESCRIPTION: Integrity Interval average calculation
#INPUT:
# fold_1: Dataframe containing the fraction of instances in fold 1 inside the confidence interval
# fold_2: Dataframe containing the fraction of instances in fold 2 inside the confidence interval
# fold_3: Dataframe containing the fraction of instances in fold 3 inside the confidence interval
# fold_4: Dataframe containing the fraction of instances in fold 4 inside the confidence interval
# fold_5: Dataframe containing the fraction of instances in fold 5 inside the confidence interval
#OUTPUT:
# ci_df_average: Dataframe with the average fraction of instances inside the confidence intervals
def ci_rf_average(fold_1,fold_2,fold_3,fold_4,fold_5):

	col = ['inside_avg']
	ci_df_average = pd.DataFrame(columns=col)

	ci_df_average = pd.concat([ci_df_average, pd.DataFrame([{'inside_avg':(fold_1.loc[-1,'inside']+fold_2.loc[-1,'inside']+fold_3.loc[-1,'inside']+fold_4.loc[-1,'inside']+fold_5.loc[-1,'inside'])/5}])])

	return ci_df_average


#DESCRIPTION: Calculates average instance feature value
#INPUT:
# data: Dataframe with the corresponding training instances
#OUTPUT:
# data_avg_std: Dataframe with the average and standard deviation of features in the training
def averageInstance(data):

	data_avg_std = pd.DataFrame(columns = data.columns)
	data_avg_std.loc[0] = range(len(data.columns))
	data_avg_std.loc[1] = range(len(data.columns))

	for i in data.columns:

		data_avg_std.loc[0,i] = np.mean(data[i])
		data_avg_std.loc[1,i] = np.std(data[i])

	return data_avg_std


#DESCRIPTION: Calculates each instance distance to the average value of the dataset
#INPUT:
# data: Dataframe with the corresponding training instances
# data_avg_std: Dataframe with the average and standard deviation of features in the training
#OUTPUT:
# eu_df: Dataframe with the normalized value for every feature and instance
def normalizedDistance(data,data_avg_std):

	eu_df = pd.DataFrame(index = data.index, columns = data.columns)

	for i in data.index:

		for j in data.columns:

			eu_df.loc[i,j] = abs((data.loc[i,j]-data_avg_std.loc[0,j])/data_avg_std.loc[1,j])

	return eu_df


#DESCRIPTION: Defines Support Vector Regression model
#INPUT:
# train_data: Dataframe with the corresponding training instances
# target: List of testing instances target
# test_data: Dataframe with the corresponding testing instances
#OUTPUT:
# prediction: List of predicted values from the SVR algorithm
def mainSVR(train_data, target, test_data):

	gamma_i = 5
	epsilon_i = 0.001
	C_i = 1
	reg = SVR(gamma=gamma_i, C=C_i, epsilon=epsilon_i)
	reg.fit(train_data, target)
	prediction = reg.predict(test_data)

	return prediction


#DESCRIPTION: Select only the features mentioned as relevant in the list inside the method
#INPUT:
# train_data: Dataframe with the corresponding training instances
# test_data: Dataframe with the corresponding testing instances
#OUTPUT:
# train_data: Dataframe with the corresponding training instances with selected features
# test_data: Dataframe with the corresponding testing instances with selected features
def selectFeatures(train_data,test_data):

	relevant_features = ['cno','nr_used_measurements','elevation','difference_ENU','lsq_residuals','azimuth','innovation_ENU','pdop']
	columns_selected = []

	for i in relevant_features:

		columns_selected.extend([j for j in train_data.columns if j.startswith(i)])

	train_data = train_data[columns_selected]
	test_data = test_data[columns_selected]

	return train_data,test_data


#DESCRIPTION: Sorts and summarizes the data for feature importance
#INPUT:
# feature_importance: Dataframe with the corresponding feature importance for all features
#OUTPUT:
# sort_summarize: Dataframe containing ordered features by importance
def sortSummarize(feature_importance):

	sort_summarize = pd.DataFrame(columns=['feature','importance'])

	for i in feature_importance['feature']:

		str = i[0:4]

		if str in ['cons', 'amb_', 'cycl', 'mult', 'used', 'lsq_', 'trac', 'elev', 'azim', 'cno_']:

			if str not in sort_summarize['feature'].tolist():

				df_sub = feature_importance.loc[(feature_importance['feature'].str[0:4] == str)]
				str_sum = sum(df_sub['importance'])
				sort_summarize = sort_summarize.append(pd.DataFrame({'feature': str, 'importance': str_sum},index=[0]))

			else:

				continue

		else:

			if i not in sort_summarize['feature'].tolist():

				df_sub = feature_importance.loc[(feature_importance['feature'] == i)]
				str_sum = sum(df_sub['importance'])
				sort_summarize = sort_summarize.append(pd.DataFrame({'feature': i, 'importance': str_sum},index=[0]))

			else:

				continue

	sort_summarize = sort_summarize.sort_values(by=['importance'],ascending=False)

	return sort_summarize

# The program below:
# 1) Selects relevant features with linear based regression
# 2) Applies Random Forest to data to obtain high accuracy predictions
# 3) Is able to store the data into a given address

print("Starting...")
start_time = time.time()
file_key_vector = ['synthetic']

r_square_dict = dict()
r_square_dict_added = dict()
p_value_dict = dict()
p_value_dict_added = dict()
params_dict = dict()
params_dict_added = dict()
feature_list = list()
added_features = list()
added_and_removed_list = list()
nofolds = 5

for i in file_key_vector:

	print("Read data for "+str(i)+" started...")
	pre_train_df,pre_test_df,eliminated_f,target,target_test,features = read_data_df(i,nofolds)
	print("Read data for "+str(i)+" finished...")

	"Euclidean distance"
	pre_train_df_avg_std = [0]*nofolds
	pre_test_norm_dist = [0]*nofolds

	"Linear Regression variables"
	base_r_square_dict = list(range(nofolds))
	base_params_dict = list(range(nofolds))
	base_p_value_dict = list(range(nofolds))
	back_p_value_dict = list(range(nofolds))
	back_r_square_dict = list(range(nofolds))
	back_params_dict = list(range(nofolds))
	step_p_value_dict = list(range(nofolds))
	step_r_square_dict = list(range(nofolds))
	step_params_dict = list(range(nofolds))
	for_p_value_dict = list(range(nofolds))
	for_r_square_dict = list(range(nofolds))
	for_params_dict = list(range(nofolds))
	lasso_r_square_dict = list(range(nofolds))
	lasso_params_dict = list(range(nofolds))
	exp_r_square_dict = list(range(nofolds))
	exp_params_dict = list(range(nofolds))

	base_best_r_squared = list(range(nofolds))
	base_best_r_squared_feat = list(range(nofolds))
	back_best_r_squared = list(range(nofolds))
	back_best_r_squared_feat = list(range(nofolds))
	step_best_r_squared = list(range(nofolds))
	step_best_r_squared_feat = list(range(nofolds))
	for_best_r_squared = list(range(nofolds))
	for_best_r_squared_feat = list(range(nofolds))
	lasso_best_r_squared = list(range(nofolds))
	lasso_best_r_squared_feat = list(range(nofolds))
	exp_best_r_squared = list(range(nofolds))
	exp_best_r_squared_feat = list(range(nofolds))

	base_dimensionality_shrink = list(range(nofolds))
	back_dimensionality_shrink = list(range(nofolds))
	step_dimensionality_shrink = list(range(nofolds))
	for_dimensionality_shrink = list(range(nofolds))
	lasso_dimensionality_shrink = list(range(nofolds))
	exp_dimensionality_shrink = list(range(nofolds))

	base_dimensionality_shrink_percent = list(range(nofolds))
	back_dimensionality_shrink_percent = list(range(nofolds))
	step_dimensionality_shrink_percent = list(range(nofolds))
	for_dimensionality_shrink_percent = list(range(nofolds))
	lasso_dimensionality_shrink_percent = list(range(nofolds))
	exp_dimensionality_shrink_percent = list(range(nofolds))

	"Random Forest variables"
	base_random_forest_feature_imp = [0]*nofolds
	base_rf_prediction = [0]*nofolds
	base_rf_r_square = [0]*nofolds
	base_rf_feature_importance = [0]*nofolds
	sorted_summarized_base_rf = [0]*nofolds
	base_rf_oob_prediction = [0]*nofolds
	base_all_tree_pred = [0]*nofolds
	base_variance_rf = [0]*nofolds
	base_norm_nonconformity_95 = [0]*nofolds
	base_norm_nonconformity_98 = [0]*nofolds
	base_norm_nonconformityInt = [0]*nofolds
	base_ci_95_rf = [0]*nofolds
	base_ci_98_rf = [0]*nofolds
	base_int_rf = [0]*nofolds

	back_train_df = [0]*nofolds
	back_test_df = [0]*nofolds
	back_random_forest_feature_imp = [0]*nofolds
	back_rf_prediction = [0]*nofolds
	back_rf_r_square = [0]*nofolds
	back_rf_feature_importance = [0]*nofolds
	sorted_summarized_back_rf = [0]*nofolds
	back_rf_oob_prediction = [0]*nofolds
	back_all_tree_pred = [0]*nofolds
	back_variance_rf = [0]*nofolds
	back_norm_nonconformity_95 = [0]*nofolds
	back_norm_nonconformity_98 = [0]*nofolds
	back_norm_nonconformityInt = [0]*nofolds
	back_ci_95_rf = [0]*nofolds
	back_ci_98_rf = [0]*nofolds
	back_int_rf = [0]*nofolds

	step_train_df = [0]*nofolds
	step_test_df = [0]*nofolds
	step_random_forest_feature_imp = [0]*nofolds
	step_rf_prediction = [0]*nofolds
	step_rf_r_square = [0]*nofolds
	step_rf_feature_importance = [0]*nofolds
	sorted_summarized_step_rf = [0]*nofolds
	step_rf_oob_prediction = [0]*nofolds
	step_all_tree_pred = [0]*nofolds
	step_variance_rf = [0]*nofolds
	step_norm_nonconformity_95 = [0]*nofolds
	step_norm_nonconformity_98 = [0]*nofolds
	step_norm_nonconformityInt = [0]*nofolds
	step_ci_95_rf = [0]*nofolds
	step_ci_98_rf = [0]*nofolds
	step_int_rf = [0]*nofolds

	for_train_df = [0]*nofolds
	for_test_df = [0]*nofolds
	for_random_forest_feature_imp = [0]*nofolds
	for_rf_prediction = [0]*nofolds
	for_rf_r_square = [0]*nofolds
	for_rf_feature_importance = [0]*nofolds
	sorted_summarized_for_rf = [0]*nofolds
	for_rf_oob_prediction = [0]*nofolds
	for_all_tree_pred = [0]*nofolds
	for_variance_rf = [0]*nofolds
	for_norm_nonconformity_95 = [0]*nofolds
	for_norm_nonconformity_98 = [0]*nofolds
	for_norm_nonconformityInt = [0]*nofolds
	for_ci_95_rf = [0]*nofolds
	for_ci_98_rf = [0]*nofolds
	for_int_rf = [0]*nofolds

	lasso_train_df = [0]*nofolds
	lasso_test_df = [0]*nofolds
	lasso_random_forest_feature_imp = [0]*nofolds
	lasso_rf_prediction = [0]*nofolds
	lasso_rf_r_square = [0]*nofolds
	lasso_rf_feature_importance = [0]*nofolds
	sorted_summarized_lasso_rf = [0]*nofolds
	lasso_rf_oob_prediction = [0]*nofolds
	lasso_all_tree_pred = [0]*nofolds
	lasso_variance_rf = [0]*nofolds
	lasso_norm_nonconformity_95 = [0]*nofolds
	lasso_norm_nonconformity_98 = [0]*nofolds
	lasso_norm_nonconformityInt = [0]*nofolds
	lasso_ci_95_rf = [0]*nofolds
	lasso_ci_98_rf = [0]*nofolds
	lasso_int_rf = [0]*nofolds

	exp_train_df = [0]*nofolds
	exp_test_df = [0]*nofolds
	exp_random_forest_feature_imp = [0]*nofolds
	exp_rf_prediction = [0]*nofolds
	exp_rf_r_square = [0]*nofolds
	exp_rf_feature_importance = [0]*nofolds
	sorted_summarized_exp_rf = [0]*nofolds
	exp_rf_oob_prediction = [0]*nofolds
	exp_all_tree_pred = [0]*nofolds
	exp_variance_rf = [0]*nofolds
	exp_norm_nonconformity_95 = [0]*nofolds
	exp_norm_nonconformity_98 = [0]*nofolds
	exp_norm_nonconformityInt = [0]*nofolds
	exp_ci_95_rf = [0]*nofolds
	exp_ci_98_rf = [0]*nofolds
	exp_int_rf = [0]*nofolds

	"SVR predictions"
	svr_prediction = [0]*nofolds
	col = ['fold_no','prediction','expected_total_error']
	svr_prediction_df = pd.DataFrame(columns=col)

	for j in range(nofolds):

		if i == 'all': # Change to 'synthetic' if wishing to apply SVR

			pre_train_df[j], pre_test_df[j] = selectFeatures(pre_train_df[j], pre_test_df[j])
			svr_prediction[j] = mainSVR(pre_train_df[j],target[j],pre_test_df[j])
			svr_prediction_df = pd.concat([svr_prediction_df,pd.DataFrame({'fold_no':j,'prediction':svr_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])

		number_features_used = len(features[j])

		print("Started baseline regression for "+str(i)+"...")
		features_j = pre_train_df[j].columns.tolist()
		base_p_value_dict[j],base_r_square_dict[j],base_params_dict[j] = baseline_lin(features_j,pre_train_df[j],target[j])
		print("Finished baseline regression for "+str(i)+"...")

		print("Started exp regression for "+str(i)+"...")
		exp_r_square_dict[j],exp_params_dict[j] = expertSelection(pre_train_df[j],target[j])
		print("Finished exp regression for "+str(i)+"...")

		print("Started lasso regression for "+str(i)+"...")
		features_j = pre_train_df[j].columns.tolist()
		lasso_r_square_dict[j],lasso_params_dict[j] = lasso(features_j,pre_train_df[j],target[j])
		print("Finished lasso regression for "+str(i)+"...")

		print("Started backward regression for "+str(i)+"...")
		features_j = pre_train_df[j].columns.tolist()
		back_p_value_dict[j],back_r_square_dict[j],back_params_dict[j],back_relevant_features,back_deleted_features = backward_selection(features_j,pre_train_df[j],r_square_dict,target[j],feature_list)
		print("Finished backward regression for "+str(i)+"...")

		print("Started stepwise regression for "+str(i)+"...")
		features_j = pre_train_df[j].columns.tolist()
		step_p_value_dict[j],step_r_square_dict[j],step_params_dict[j],step_relevant_features = stepwise_selection(features_j,pre_train_df[j],p_value_dict_added,r_square_dict_added,params_dict_added,added_features,target[j])
		print("Finished stepwise regression for "+str(i)+"...")

		features_j = pre_train_df[j].columns.tolist()

		r_square_dict = {}
		r_square_dict_added = {}
		p_value_dict = {}
		p_value_dict_added = {}
		params_dict = {}
		params_dict_added = {}
		feature_list = []
		added_features = []

		print("Started forward regression for "+str(i)+"...")
		for_p_value_dict[j],for_r_square_dict[j],for_params_dict[j],for_relevant_features,for_deleted_features = forward_selection(features_j,pre_train_df[j],p_value_dict_added,r_square_dict_added,params_dict_added,target[j],added_features)
		print("Finished forward regression for "+str(i)+"...")

		max_r_squared = max(list(base_r_square_dict[j].values()))
		key_max_r_squared = max(base_r_square_dict[j], key=base_r_square_dict[j].get)
		base_best_r_squared[j] = max_r_squared
		base_best_r_squared_feat[j] = key_max_r_squared

		max_r_squared = max(list(lasso_r_square_dict[j].values()))
		key_max_r_squared = max(lasso_r_square_dict[j], key=lasso_r_square_dict[j].get)
		lasso_best_r_squared[j] = max_r_squared
		lasso_best_r_squared_feat[j] = key_max_r_squared

		max_r_squared = max(list(exp_r_square_dict[j].values()))
		key_max_r_squared = max(exp_r_square_dict[j], key=exp_r_square_dict[j].get)
		exp_best_r_squared[j] = max_r_squared
		exp_best_r_squared_feat[j] = key_max_r_squared

		min_r_squared = min(list(back_r_square_dict[j].values()))
		key_min_r_squared = min(back_r_square_dict[j], key=back_r_square_dict[j].get)
		back_best_r_squared[j] = min_r_squared
		back_best_r_squared_feat[j] = key_min_r_squared

		max_r_squared = max(list(step_r_square_dict[j].values()))
		key_max_r_squared = max(step_r_square_dict[j], key=step_r_square_dict[j].get)
		step_best_r_squared[j] = max_r_squared
		step_best_r_squared_feat[j] = key_max_r_squared

		max_r_squared = max(list(for_r_square_dict[j].values()))
		key_max_r_squared = max(for_r_square_dict[j], key=for_r_square_dict[j].get)
		for_best_r_squared[j] = max_r_squared
		for_best_r_squared_feat[j] = key_max_r_squared

		base_dimensionality_shrink[j] = number_features_used - len(base_params_dict[j]) + 1
		back_dimensionality_shrink[j] = number_features_used - len(back_params_dict[j]) + 1
		step_dimensionality_shrink[j] = number_features_used - len(step_params_dict[j]) + 1
		for_dimensionality_shrink[j] = number_features_used - len(for_params_dict[j]) + 1
		lasso_dimensionality_shrink[j] = number_features_used - len(lasso_params_dict[j]) + 1
		exp_dimensionality_shrink[j] = number_features_used - len(exp_params_dict[j]) + 1

		base_dimensionality_shrink_percent[j] = base_dimensionality_shrink[j]/number_features_used
		back_dimensionality_shrink_percent[j] = back_dimensionality_shrink[j]/number_features_used
		step_dimensionality_shrink_percent[j] = step_dimensionality_shrink[j]/number_features_used
		for_dimensionality_shrink_percent[j] = for_dimensionality_shrink[j]/number_features_used
		lasso_dimensionality_shrink_percent[j] = lasso_dimensionality_shrink[j]/number_features_used
		exp_dimensionality_shrink_percent[j] = exp_dimensionality_shrink[j]/number_features_used

		r_square_dict = {}
		r_square_dict_added = {}
		p_value_dict = {}
		p_value_dict_added = {}
		params_dict = {}
		params_dict_added = {}
		feature_list = []
		added_features = []

	"7. For every single run and method (every fold, method and file) calculate the RMSE of the predictions. Find the average throughout the folds (for every file and method)"
	"Also plot linear RMSE of prediction vs. Dimensionality reduction for each method for every file"

	print("Started baseline linear predictions "+str(i)+"...")
	base_prediction = predict(pre_test_df,base_params_dict,nofolds,target_test)
	print("Finished baseline linear predictions "+str(i)+"...")

	print("Started backward linear predictions "+str(i)+"...")
	back_prediction = predict(pre_test_df,back_params_dict,nofolds,target_test)
	print("Finished backward linear predictions "+str(i)+"...")

	print("Started stepwise linear predictions "+str(i)+"...")
	step_prediction = predict(pre_test_df,step_params_dict,nofolds,target_test)
	print("Finished stepwise linear predictions "+str(i)+"...")

	print("Started forward linear predictions "+str(i)+"...")
	for_prediction = predict(pre_test_df,for_params_dict,nofolds,target_test)
	print("Finished forward linear predictions "+str(i)+"...")

	print("Started lasso linear predictions "+str(i)+"...")
	lasso_prediction = predict(pre_test_df,lasso_params_dict,nofolds,target_test)
	print("Finished lasso linear predictions "+str(i)+"...")

	print("Started exp linear predictions "+str(i)+"...")
	exp_prediction = predict(pre_test_df,exp_params_dict,nofolds,target_test)
	print("Finished exp linear predictions "+str(i)+"...")

	print("Started baseline linear RMSE calculations "+str(i)+"...")
	base_RMSE = rmse_lin(target_test,base_prediction,nofolds)
	print("Finished baseline linear RMSE calculations "+str(i)+"...")

	print("Started backward linear RMSE calculations "+str(i)+"...")
	back_RMSE = rmse_lin(target_test,back_prediction,nofolds)
	print("Finished backward linear RMSE calculations "+str(i)+"...")

	print("Started stepwise linear RMSE calculations "+str(i)+"...")
	step_RMSE = rmse_lin(target_test,step_prediction,nofolds)
	print("Finished stepwise linear RMSE calculations "+str(i)+"...")

	print("Started forward linear RMSE calculations "+str(i)+"...")
	for_RMSE = rmse_lin(target_test,for_prediction,nofolds)
	print("Finished forward linear RMSE calculations "+str(i)+"...")

	print("Started lasso linear RMSE calculations "+str(i)+"...")
	lasso_RMSE = rmse_lin(target_test,lasso_prediction,nofolds)
	print("Finished lasso linear RMSE calculations "+str(i)+"...")

	print("Started exp linear RMSE calculations "+str(i)+"...")
	exp_RMSE = rmse_lin(target_test,exp_prediction,nofolds)
	print("Finished exp linear RMSE calculations "+str(i)+"...")

	"8. Random Forests: RMSE, R^2 and Stanford diagrams plot for all 3 selection methods and Baseline (no Feature Selection applied)"
	"Parameters for random forest regressor"
	ntrees = 100
	crit = 'mse'
	min_sample_split = 10
	min_leaf_size = 1
	oob = True

	print("Started all methods random forests training and predictions "+str(i)+"...")
	for j in range(nofolds):

		"Random Forest baseline method call"
		base_random_forest_feature_imp[j],base_rf_prediction[j],base_rf_r_square[j],base_rf_oob_prediction[j],base_all_tree_pred[j] = randomForest(pre_train_df[j],target[j],pre_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
		base_rf_feature_importance[j] = pd.DataFrame({'feature':pre_train_df[j].columns.tolist(),'importance':base_random_forest_feature_imp[j]})

		"Random Forest backward method call"
		back_train_df[j],back_test_df[j] = eliminateFeatures(back_params_dict[j],pre_train_df[j],pre_test_df[j])
		if len(back_train_df[j].columns) > 0:
			back_random_forest_feature_imp[j],back_rf_prediction[j],back_rf_r_square[j],back_rf_oob_prediction[j],back_all_tree_pred[j] = randomForest(back_train_df[j],target[j],back_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
			back_rf_feature_importance[j] = pd.DataFrame({'feature':back_train_df[j].columns.tolist(),'importance':back_random_forest_feature_imp[j]})
		else:
			back_rf_prediction[j] = np.zeros(len(target_test[j]))

		"Random Forest stepwise method call"
		step_train_df[j],step_test_df[j] = eliminateFeatures(step_params_dict[j],pre_train_df[j],pre_test_df[j])
		if len(step_train_df[j].columns) > 0:
			step_random_forest_feature_imp[j],step_rf_prediction[j],step_rf_r_square[j],step_rf_oob_prediction[j],step_all_tree_pred[j] = randomForest(step_train_df[j],target[j],step_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
			step_rf_feature_importance[j] = pd.DataFrame({'feature':step_train_df[j].columns.tolist(),'importance':step_random_forest_feature_imp[j]})
		else:
			step_rf_prediction[j] = np.zeros(len(target_test[j]))

		"Random Forest forward method call"
		for_train_df[j],for_test_df[j] = eliminateFeatures(for_params_dict[j],pre_train_df[j],pre_test_df[j])
		if len(for_train_df[j].columns) > 0:
			for_random_forest_feature_imp[j],for_rf_prediction[j],for_rf_r_square[j],for_rf_oob_prediction[j],for_all_tree_pred[j] = randomForest(for_train_df[j],target[j],for_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
			for_rf_feature_importance[j] = pd.DataFrame({'feature':for_train_df[j].columns.tolist(),'importance':for_random_forest_feature_imp[j]})
		else:
			for_rf_prediction[j] = np.zeros(len(target_test[j]))

		"Random Forest lasso method call"
		lasso_train_df[j],lasso_test_df[j] = eliminateFeatures(lasso_params_dict[j],pre_train_df[j],pre_test_df[j])
		if len(lasso_train_df[j].columns) > 0:
			lasso_random_forest_feature_imp[j],lasso_rf_prediction[j],lasso_rf_r_square[j],lasso_rf_oob_prediction[j],lasso_all_tree_pred[j] = randomForest(lasso_train_df[j],target[j],lasso_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
			lasso_rf_feature_importance[j] = pd.DataFrame({'feature':lasso_train_df[j].columns.tolist(),'importance':lasso_random_forest_feature_imp[j]})
		else:
			lasso_rf_prediction[j] = np.zeros(len(target_test[j]))

		"Random Forest exp method call"
		exp_train_df[j],exp_test_df[j] = eliminateFeatures(exp_params_dict[j],pre_train_df[j],pre_test_df[j])
		if len(exp_train_df[j].columns) > 0:
			exp_random_forest_feature_imp[j],exp_rf_prediction[j],exp_rf_r_square[j],exp_rf_oob_prediction[j],exp_all_tree_pred[j] = randomForest(exp_train_df[j],target[j],exp_test_df[j],target_test[j],ntrees,crit,min_sample_split,min_leaf_size,oob)
			exp_rf_feature_importance[j] = pd.DataFrame({'feature':exp_train_df[j].columns.tolist(),'importance':exp_random_forest_feature_imp[j]})
		else:
			exp_rf_prediction[j] = np.zeros(len(target_test[j]))

		base_variance_rf[j] = varianceNonconformity(base_all_tree_pred[j],base_rf_prediction[j],ntrees)
		back_variance_rf[j] = varianceNonconformity(back_all_tree_pred[j],base_rf_prediction[j],ntrees)
		step_variance_rf[j] = varianceNonconformity(step_all_tree_pred[j],base_rf_prediction[j],ntrees)
		for_variance_rf[j] = varianceNonconformity(for_all_tree_pred[j],base_rf_prediction[j],ntrees)
		lasso_variance_rf[j] = varianceNonconformity(lasso_all_tree_pred[j],base_rf_prediction[j],ntrees)
		exp_variance_rf[j] = varianceNonconformity(exp_all_tree_pred[j],base_rf_prediction[j],ntrees)

		beta = 0.001
		base_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],base_rf_oob_prediction[j],base_variance_rf[j])
		back_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],back_rf_oob_prediction[j],back_variance_rf[j])
		step_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],step_rf_oob_prediction[j],step_variance_rf[j])
		for_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],for_rf_oob_prediction[j],for_variance_rf[j])
		lasso_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],lasso_rf_oob_prediction[j],lasso_variance_rf[j])
		exp_norm_nonconformity_95[j] = normalizedNonconformity_95(target[j],exp_rf_oob_prediction[j],exp_variance_rf[j])

		base_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],base_rf_oob_prediction[j],base_variance_rf[j])
		back_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],back_rf_oob_prediction[j],back_variance_rf[j])
		step_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],step_rf_oob_prediction[j],step_variance_rf[j])
		for_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],for_rf_oob_prediction[j],for_variance_rf[j])
		lasso_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],lasso_rf_oob_prediction[j],lasso_variance_rf[j])
		exp_norm_nonconformity_98[j] = normalizedNonconformity_98(target[j],exp_rf_oob_prediction[j],exp_variance_rf[j])

		base_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],base_rf_oob_prediction[j],base_variance_rf[j])
		back_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],back_rf_oob_prediction[j],back_variance_rf[j])
		step_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],step_rf_oob_prediction[j],step_variance_rf[j])
		for_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],for_rf_oob_prediction[j],for_variance_rf[j])
		lasso_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],lasso_rf_oob_prediction[j],lasso_variance_rf[j])
		exp_norm_nonconformityInt[j] = normalizedNonconformityIntegrity(target[j],exp_rf_oob_prediction[j],exp_variance_rf[j])

		temp = target_test[j].index.tolist()
		temp.append(-1)
		base_ci_95_rf[j] = ci_rf(base_norm_nonconformity_95[j],base_rf_prediction[j],base_variance_rf[j],beta,target_test[j])
		base_ci_95_rf[j]['target_index'] = temp
		base_ci_95_rf[j].set_index(['target_index'],inplace=True)
		back_ci_95_rf[j] = ci_rf(back_norm_nonconformity_95[j],back_rf_prediction[j],back_variance_rf[j],beta,target_test[j])
		back_ci_95_rf[j]['target_index'] = temp
		back_ci_95_rf[j].set_index(['target_index'],inplace=True)
		step_ci_95_rf[j] = ci_rf(step_norm_nonconformity_95[j],step_rf_prediction[j],step_variance_rf[j],beta,target_test[j])
		step_ci_95_rf[j]['target_index'] = temp
		step_ci_95_rf[j].set_index(['target_index'],inplace=True)
		for_ci_95_rf[j] = ci_rf(for_norm_nonconformity_95[j],for_rf_prediction[j],for_variance_rf[j],beta,target_test[j])
		for_ci_95_rf[j]['target_index'] = temp
		for_ci_95_rf[j].set_index(['target_index'],inplace=True)
		lasso_ci_95_rf[j] = ci_rf(lasso_norm_nonconformity_95[j],lasso_rf_prediction[j],lasso_variance_rf[j],beta,target_test[j])
		lasso_ci_95_rf[j]['target_index'] = temp
		lasso_ci_95_rf[j].set_index(['target_index'],inplace=True)
		exp_ci_95_rf[j] = ci_rf(exp_norm_nonconformity_95[j],exp_rf_prediction[j],exp_variance_rf[j],beta,target_test[j])
		exp_ci_95_rf[j]['target_index'] = temp
		exp_ci_95_rf[j].set_index(['target_index'],inplace=True)

		base_ci_98_rf[j] = ci_rf(base_norm_nonconformity_98[j],base_rf_prediction[j],base_variance_rf[j],beta,target_test[j])
		base_ci_98_rf[j]['target_index'] = temp
		base_ci_98_rf[j].set_index(['target_index'],inplace=True)
		back_ci_98_rf[j] = ci_rf(back_norm_nonconformity_98[j],back_rf_prediction[j],back_variance_rf[j],beta,target_test[j])
		back_ci_98_rf[j]['target_index'] = temp
		back_ci_98_rf[j].set_index(['target_index'],inplace=True)
		step_ci_98_rf[j] = ci_rf(step_norm_nonconformity_98[j],step_rf_prediction[j],step_variance_rf[j],beta,target_test[j])
		step_ci_98_rf[j]['target_index'] = temp
		step_ci_98_rf[j].set_index(['target_index'],inplace=True)
		for_ci_98_rf[j] = ci_rf(for_norm_nonconformity_98[j],for_rf_prediction[j],for_variance_rf[j],beta,target_test[j])
		for_ci_98_rf[j]['target_index'] = temp
		for_ci_98_rf[j].set_index(['target_index'],inplace=True)
		lasso_ci_98_rf[j] = ci_rf(lasso_norm_nonconformity_98[j],lasso_rf_prediction[j],lasso_variance_rf[j],beta,target_test[j])
		lasso_ci_98_rf[j]['target_index'] = temp
		lasso_ci_98_rf[j].set_index(['target_index'],inplace=True)
		exp_ci_98_rf[j] = ci_rf(exp_norm_nonconformity_98[j],exp_rf_prediction[j],exp_variance_rf[j],beta,target_test[j])
		exp_ci_98_rf[j]['target_index'] = temp
		exp_ci_98_rf[j].set_index(['target_index'],inplace=True)

		base_int_rf[j] = ci_rf(base_norm_nonconformityInt[j],base_rf_prediction[j],base_variance_rf[j],beta,target_test[j])
		base_int_rf[j]['target_index'] = temp
		base_int_rf[j].set_index(['target_index'],inplace=True)
		back_int_rf[j] = ci_rf(back_norm_nonconformityInt[j],back_rf_prediction[j],back_variance_rf[j],beta,target_test[j])
		back_int_rf[j]['target_index'] = temp
		back_int_rf[j].set_index(['target_index'],inplace=True)
		step_int_rf[j] = ci_rf(step_norm_nonconformityInt[j],step_rf_prediction[j],step_variance_rf[j],beta,target_test[j])
		step_int_rf[j]['target_index'] = temp
		step_int_rf[j].set_index(['target_index'],inplace=True)
		for_int_rf[j] = ci_rf(for_norm_nonconformityInt[j],for_rf_prediction[j],for_variance_rf[j],beta,target_test[j])
		for_int_rf[j]['target_index'] = temp
		for_int_rf[j].set_index(['target_index'],inplace=True)
		lasso_int_rf[j] = ci_rf(lasso_norm_nonconformityInt[j],lasso_rf_prediction[j],lasso_variance_rf[j],beta,target_test[j])
		lasso_int_rf[j]['target_index'] = temp
		lasso_int_rf[j].set_index(['target_index'],inplace=True)
		exp_int_rf[j] = ci_rf(exp_norm_nonconformityInt[j],exp_rf_prediction[j],exp_variance_rf[j],beta,target_test[j])
		exp_int_rf[j]['target_index'] = temp
		exp_int_rf[j].set_index(['target_index'],inplace=True)

	print("Finished all methods random forests training and predictions "+str(i)+"...")

	print("Started baseline rf RMSE calculations "+str(i)+"...")
	base_rf_RMSE = rmse_rf(target_test,base_rf_prediction,nofolds)
	print("Finished baseline rf RMSE calculations "+str(i)+"...")

	print("Started backward rf RMSE calculations "+str(i)+"...")
	back_rf_RMSE = rmse_rf(target_test,back_rf_prediction,nofolds)
	print("Finished backward rf RMSE calculations "+str(i)+"...")

	print("Started stepwise rf RMSE calculations "+str(i)+"...")
	step_rf_RMSE = rmse_rf(target_test,step_rf_prediction,nofolds)
	print("Finished stepwise rf RMSE calculations "+str(i)+"...")

	print("Started forward rf RMSE calculations "+str(i)+"...")
	for_rf_RMSE = rmse_rf(target_test,for_rf_prediction,nofolds)
	print("Finished forward rf RMSE calculations "+str(i)+"...")

	print("Started lasso rf RMSE calculations "+str(i)+"...")
	lasso_rf_RMSE = rmse_rf(target_test,lasso_rf_prediction,nofolds)
	print("Finished lasso rf RMSE calculations "+str(i)+"...")

	print("Started exp rf RMSE calculations "+str(i)+"...")
	exp_rf_RMSE = rmse_rf(target_test,exp_rf_prediction,nofolds)
	print("Finished exp rf RMSE calculations "+str(i)+"...")

	if i == 'all':

		print("Started svr RMSE calculations "+str(i)+"...")
		svr_RMSE = rmse_rf(target_test,svr_prediction,nofolds)
		print("Finished svr RMSE calculations "+str(i)+"...")

	"11. Save all information required for analysis"
	"1. Average RMSE for all methods, for every fold, for every regressor for every file."
	"2. Total Average RMSE with standard deviation (for every file)."
	"3. All feature importances (all linear params dict for every fold, for every feature selection method. All feature_rf_importance for every fold, for every feature selection method and other regressors used the same)."
	"4. Print instance number (careful of the index number preservation) with total_expected_error and predicted error for every fold, every feature selection method, every regressor, every file"

	linear_RMSE = pd.concat([base_RMSE,back_RMSE,step_RMSE,for_RMSE,lasso_RMSE,exp_RMSE])
	linear_RMSE_mean = np.mean(linear_RMSE['error'])
	linear_RMSE_std = np.std(linear_RMSE['error'],ddof=1)
	linear_RMSE_df = pd.DataFrame({'mean':linear_RMSE_mean,'std':linear_RMSE_std},index=[1])

	rf_RMSE = pd.concat([base_rf_RMSE,back_rf_RMSE,step_rf_RMSE,for_rf_RMSE,lasso_rf_RMSE,exp_rf_RMSE])
	rf_RMSE_mean = np.mean(rf_RMSE['error'])
	rf_RMSE_std = np.std(rf_RMSE['error'],ddof=1)
	rf_RMSE_df = pd.DataFrame({'mean':rf_RMSE_mean,'std':rf_RMSE_std},index=[1])

	col = ['fold_no','prediction','expected_total_error']
	base_rf_prediction_df = pd.DataFrame(columns=col)
	back_rf_prediction_df = pd.DataFrame(columns=col)
	step_rf_prediction_df = pd.DataFrame(columns=col)
	for_rf_prediction_df = pd.DataFrame(columns=col)
	lasso_rf_prediction_df = pd.DataFrame(columns=col)
	exp_rf_prediction_df = pd.DataFrame(columns=col)

	for j in range(nofolds):

		base_rf_prediction_df = pd.concat([base_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':base_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		back_rf_prediction_df = pd.concat([back_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':back_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		step_rf_prediction_df = pd.concat([step_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':step_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		for_rf_prediction_df = pd.concat([for_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':for_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		lasso_rf_prediction_df = pd.concat([lasso_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':lasso_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		exp_rf_prediction_df = pd.concat([exp_rf_prediction_df,pd.DataFrame({'fold_no':j,'prediction':exp_rf_prediction[j],'expected_total_error':target_test[j]},index=target_test[j].index.tolist())])
		sorted_summarized_base_rf[j] = sortSummarize(base_rf_feature_importance[j])
		sorted_summarized_back_rf[j] = sortSummarize(back_rf_feature_importance[j])
		sorted_summarized_step_rf[j] = sortSummarize(step_rf_feature_importance[j])
		sorted_summarized_for_rf[j] = sortSummarize(for_rf_feature_importance[j])
		sorted_summarized_lasso_rf[j] = sortSummarize(lasso_rf_feature_importance[j])
		sorted_summarized_exp_rf[j] = sortSummarize(exp_rf_feature_importance[j])

	base_int_rf_average = ci_rf_average(base_int_rf[0],base_int_rf[1],base_int_rf[2],base_int_rf[3],base_int_rf[4])
	back_int_rf_average = ci_rf_average(back_int_rf[0], back_int_rf[1], back_int_rf[2], back_int_rf[3], back_int_rf[4])
	step_int_rf_average = ci_rf_average(step_int_rf[0], step_int_rf[1], step_int_rf[2], step_int_rf[3], step_int_rf[4])
	for_int_rf_average = ci_rf_average(for_int_rf[0], for_int_rf[1], for_int_rf[2], for_int_rf[3], for_int_rf[4])
	lasso_int_rf_average = ci_rf_average(lasso_int_rf[0], lasso_int_rf[1], lasso_int_rf[2], lasso_int_rf[3], lasso_int_rf[4])
	exp_int_rf_average = ci_rf_average(exp_int_rf[0], exp_int_rf[1], exp_int_rf[2], exp_int_rf[3], exp_int_rf[4])


	for j in range(nofolds):

		pre_train_df_avg_std[j] = averageInstance(pre_train_df[j])
		pre_test_norm_dist[j] = normalizedDistance(pre_test_df[j],pre_train_df_avg_std[j])

	print("Started printing all variables of RMSE and parameters for "+str(i)+"...")
	writer = pd.ExcelWriter(df_pre.ABS_DIR+' RMSE '+str(i)+'.xlsx', engine='xlsxwriter')

	base_RMSE.to_excel(writer,sheet_name ='lin_baseline')
	back_RMSE.to_excel(writer,sheet_name ='lin_backward')
	step_RMSE.to_excel(writer,sheet_name ='lin_stepwise')
	for_RMSE.to_excel(writer,sheet_name ='lin_forward')
	lasso_RMSE.to_excel(writer,sheet_name ='lin_lasso')
	exp_RMSE.to_excel(writer,sheet_name ='lin_exp')

	base_rf_RMSE.to_excel(writer,sheet_name ='rf_baseline')
	back_rf_RMSE.to_excel(writer,sheet_name ='rf_backward')
	step_rf_RMSE.to_excel(writer,sheet_name ='rf_stepwise')
	for_rf_RMSE.to_excel(writer,sheet_name ='rf_forward')
	lasso_rf_RMSE.to_excel(writer,sheet_name ='rf_lasso')
	exp_rf_RMSE.to_excel(writer,sheet_name ='rf_exp')

	if i == 'all': # Change to 'synthetic' if wishing to apply SVR
		svr_RMSE.to_excel(writer,sheet_name='svr')

	linear_RMSE_df.to_excel(writer,sheet_name ='linear_RMSE')
	rf_RMSE_df.to_excel(writer,sheet_name ='rf_RMSE')

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' params_feature_importance '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		base_params_dict[j].to_excel(writer,sheet_name ='lin_base_fold '+str(j))
		back_params_dict[j].to_excel(writer,sheet_name ='lin_back_fold '+str(j))
		step_params_dict[j].to_excel(writer,sheet_name ='lin_step_fold '+str(j))
		for_params_dict[j].to_excel(writer,sheet_name ='lin_for_fold '+str(j))
		lasso_params_dict[j].to_excel(writer,sheet_name ='lin_lasso_fold '+str(j))
		exp_params_dict[j].to_excel(writer,sheet_name ='lin_exp_fold '+str(j))

		base_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_base_fold '+str(j))
		back_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_back_fold '+str(j))

		try:
			step_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_step_fold '+str(j))
		except:
			base_rf_feature_importance[j].to_excel(writer, sheet_name='rf_step_NA_fold ' + str(j))

		for_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_for_fold '+str(j))
		lasso_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_lasso_fold '+str(j))
		exp_rf_feature_importance[j].to_excel(writer,sheet_name ='rf_exp_fold '+str(j))

	for j in range(nofolds):

		sorted_summarized_base_rf[j].to_excel(writer,sheet_name ='rf_base_ss_fold '+str(j))
		sorted_summarized_back_rf[j].to_excel(writer,sheet_name ='rf_back_ss_fold '+str(j))
		sorted_summarized_step_rf[j].to_excel(writer,sheet_name ='rf_step_ss_fold '+str(j))
		sorted_summarized_for_rf[j].to_excel(writer,sheet_name ='rf_for_ss_fold '+str(j))
		sorted_summarized_lasso_rf[j].to_excel(writer,sheet_name ='rf_lasso_ss_fold '+str(j))
		sorted_summarized_exp_rf[j].to_excel(writer,sheet_name ='rf_exp_ss_fold '+str(j))

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' instances_predictions '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		base_prediction[base_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_base_fold '+str(j))
		back_prediction[back_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_back_fold '+str(j))
		step_prediction[step_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_step_fold '+str(j))
		for_prediction[for_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_for_fold '+str(j))
		lasso_prediction[lasso_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_lasso_fold '+str(j))
		exp_prediction[exp_prediction['fold_no'] == j].to_excel(writer,sheet_name ='lin_exp_fold '+str(j))

		base_rf_prediction_df[base_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_base_fold '+str(j))
		back_rf_prediction_df[back_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_back_fold '+str(j))
		step_rf_prediction_df[step_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_step_fold '+str(j))
		for_rf_prediction_df[for_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_for_fold '+str(j))
		lasso_rf_prediction_df[lasso_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_lasso_fold '+str(j))
		exp_rf_prediction_df[exp_rf_prediction_df['fold_no'] == j].to_excel(writer,sheet_name ='rf_exp_fold '+str(j))

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' conformal prediction 95 range '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		base_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_base_fold '+str(j))
		back_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_back_fold '+str(j))
		step_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_step_fold '+str(j))
		for_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_for_fold '+str(j))
		lasso_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_lasso_fold '+str(j))
		exp_ci_95_rf[j].to_excel(writer,sheet_name ='ci_rf_exp_fold '+str(j))

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' conformal prediction 98 range '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		base_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_base_fold '+str(j))
		back_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_back_fold '+str(j))
		step_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_step_fold '+str(j))
		for_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_for_fold '+str(j))
		lasso_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_lasso_fold '+str(j))
		exp_ci_98_rf[j].to_excel(writer,sheet_name ='ci_rf_exp_fold '+str(j))

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' conformal integrity prediction range '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		base_int_rf[j].to_excel(writer,sheet_name ='int_rf_base_fold '+str(j))
		back_int_rf[j].to_excel(writer,sheet_name ='int_rf_back_fold '+str(j))
		step_int_rf[j].to_excel(writer,sheet_name ='int_rf_step_fold '+str(j))
		for_int_rf[j].to_excel(writer,sheet_name ='int_rf_for_fold '+str(j))
		lasso_int_rf[j].to_excel(writer,sheet_name ='int_rf_lasso_fold '+str(j))
		exp_int_rf[j].to_excel(writer,sheet_name ='int_rf_exp_fold '+str(j))

	base_int_rf_average.to_excel(writer,sheet_name ='int_rf_base_avg_fold '+str(j))
	back_int_rf_average.to_excel(writer, sheet_name='int_rf_back_avg_fold ' + str(j))
	step_int_rf_average.to_excel(writer, sheet_name='int_rf_step_avg_fold ' + str(j))
	for_int_rf_average.to_excel(writer, sheet_name='int_rf_for_avg_fold ' + str(j))
	lasso_int_rf_average.to_excel(writer, sheet_name='int_rf_lasso_avg_fold ' + str(j))
	exp_int_rf_average.to_excel(writer, sheet_name='int_rf_exp_avg_fold ' + str(j))

	writer.save()

	writer = pd.ExcelWriter(df_pre.ABS_DIR+' normalized distance '+str(i)+'.xlsx', engine='xlsxwriter')

	for j in range(nofolds):

		pre_train_df_avg_std[j].to_excel(writer,sheet_name ='avg_std_train_inst_fold '+str(j))
		pre_test_norm_dist[j].to_excel(writer,sheet_name ='norm_dist_fold '+str(j))

	writer.save()

	print("Finished printing all variables of RMSE and parameters for "+str(i)+"...")

	print("Started all plots for  "+str(i)+"...")
	labels = ['Baseline','Backward','Stepwise','Forward','Lasso','Exp']

	fboxplot, ax = plt.subplots()
	plt.boxplot([base_best_r_squared,back_best_r_squared,step_best_r_squared,for_best_r_squared,lasso_best_r_squared,exp_best_r_squared])
	# plt.title(str(i)+' R^2 Boxplot')
	plt.xlabel('Selection Method')
	ax.set_xticklabels(labels)
	plt.legend()
	plt.ylabel('R^2')
	plt.savefig(df_pre.ABS_DIR+str(i)+'linear boxplot '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fboxplot)

	fboxplot1, ax1 = plt.subplots()
	plt.boxplot([base_dimensionality_shrink,back_dimensionality_shrink,step_dimensionality_shrink,for_dimensionality_shrink,lasso_dimensionality_shrink,exp_dimensionality_shrink])
	# plt.title(str(i)+' Dimensionality Reduction Boxplot')
	plt.xlabel('Selection Method')
	ax1.set_xticklabels(labels)
	plt.legend()
	plt.ylabel('Number of Features eliminated')
	plt.savefig(df_pre.ABS_DIR+'dim_red '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fboxplot1)

	fboxplot2, ax2 = plt.subplots()
	plt.boxplot([base_dimensionality_shrink_percent,back_dimensionality_shrink_percent,step_dimensionality_shrink_percent,for_dimensionality_shrink_percent,lasso_dimensionality_shrink_percent,exp_dimensionality_shrink_percent])
	# plt.title(str(i)+' Dimensionality Reduction Percentage Boxplot')
	plt.xlabel('Selection Method')
	ax2.set_xticklabels(labels)
	plt.legend()
	plt.ylabel('Percentage of Features eliminated')
	plt.savefig(df_pre.ABS_DIR+'linear dim_red_percent '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fboxplot2)

	"6. Plot linear R^2 vs. Dimensionality reduction for each method for every file (average throughout the 5 folds) in one single scatter plot"
	fig = plt.figure()
	ax3 = fig.add_subplot(111)

	ax3.scatter(base_dimensionality_shrink_percent, base_best_r_squared, s=10, c='black', marker="h", label='Baseline')
	ax3.scatter(back_dimensionality_shrink_percent, back_best_r_squared, s=10, c='b', marker="x", label='Backward')
	ax3.scatter(step_dimensionality_shrink_percent, step_best_r_squared, s=10, c='r', marker="o", label='Stepwise')
	ax3.scatter(for_dimensionality_shrink_percent, for_best_r_squared, s=10, c='g', marker="s", label='Forward')
	ax3.scatter(lasso_dimensionality_shrink_percent, lasso_best_r_squared, s=10, c='pink', marker="p", label='Lasso')
	ax3.scatter(exp_dimensionality_shrink_percent, exp_best_r_squared, s=10, c='gray', marker="v", label='Exp')
	# plt.title(str(i)+'Linear R^2 vs. Dimension. Red. Percent. Scatter plot')
	plt.xlabel('Dimensionality Reduction Percentage')
	plt.ylabel('R^2')
	plt.legend()
	plt.savefig(df_pre.ABS_DIR+'Linear R2_dim_red_percent '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fig)
	#	plt.show()

	fig = plt.figure()
	ax4 = fig.add_subplot(111)

	ax4.scatter(base_dimensionality_shrink_percent, base_RMSE['error'], s=10, c='black', marker="h", label='Baseline')
	ax4.scatter(back_dimensionality_shrink_percent, back_RMSE['error'], s=10, c='b', marker="x", label='Backward')
	ax4.scatter(step_dimensionality_shrink_percent, step_RMSE['error'], s=10, c='r', marker="o", label='Stepwise')
	ax4.scatter(for_dimensionality_shrink_percent, for_RMSE['error'], s=10, c='g', marker="s", label='Forward')
	ax4.scatter(lasso_dimensionality_shrink_percent, for_RMSE['error'], s=10, c='pink', marker="p", label='Lasso')
	ax4.scatter(exp_dimensionality_shrink_percent, for_RMSE['error'], s=10, c='gray', marker="v", label='Exp')
	# plt.title(str(i)+'Linear RMSE vs. Dimension. Red. Percent. Scatter plot')
	plt.xlabel('Dimensionality Reduction Percentage')
	plt.ylabel('RMSE')
	plt.legend();
	plt.savefig(df_pre.ABS_DIR+'Linear RMSE_dim_red_percent '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fig)
	#	plt.show()

	fboxplot3, ax3_1 = plt.subplots()
	plt.boxplot([base_RMSE['error'],back_RMSE['error'],step_RMSE['error'],for_RMSE['error'],lasso_RMSE['error'],exp_RMSE['error']])
	# plt.title(str(i)+' Linear RMSE Boxplot')
	plt.xlabel('Selection Method')
	ax3_1.set_xticklabels(labels)
	plt.legend()
	plt.ylabel('RMSE')
	plt.savefig(df_pre.ABS_DIR+'Linear RMSE_all_methods. '+str(i)+' png', format='png', dpi = 800)
	plt.close(fboxplot3)
	#	plt.show()

	fig = plt.figure()
	ax5 = fig.add_subplot(111)
	ax5.scatter(base_dimensionality_shrink_percent, base_rf_r_square, s=10, c='black', marker="h", label='Baseline')
	ax5.scatter(back_dimensionality_shrink_percent, back_rf_r_square, s=10, c='b', marker="x", label='Backward')
	ax5.scatter(step_dimensionality_shrink_percent, step_rf_r_square, s=10, c='r', marker="o", label='Stepwise')
	ax5.scatter(for_dimensionality_shrink_percent, for_rf_r_square, s=10, c='g', marker="s", label='Forward')
	ax5.scatter(lasso_dimensionality_shrink_percent, lasso_rf_r_square, s=10, c='pink', marker="p", label='Lasso')
	ax5.scatter(exp_dimensionality_shrink_percent, exp_rf_r_square, s=10, c='gray', marker="v", label='Exp')
	# plt.title(str(i)+' Random For. R^2 vs. Dimension. Red. Percent. Scatter plot')
	plt.xlabel('Dimensionality Reduction Percentage')
	plt.ylabel('R^2')
	plt.legend();
	plt.savefig(df_pre.ABS_DIR+'Random For R2_dim_red_percent '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fig)

	fig = plt.figure()
	ax6 = fig.add_subplot(111)
	ax6.scatter(base_dimensionality_shrink_percent, base_rf_RMSE['error'][0:5], s=10, c='black', marker="h", label='Baseline')
	ax6.scatter(back_dimensionality_shrink_percent, back_rf_RMSE['error'][0:5], s=10, c='b', marker="x", label='Backward')
	ax6.scatter(step_dimensionality_shrink_percent, step_rf_RMSE['error'][0:5], s=10, c='r', marker="o", label='Stepwise')
	ax6.scatter(for_dimensionality_shrink_percent, for_rf_RMSE['error'][0:5], s=10, c='g', marker="s", label='Forward')
	ax6.scatter(lasso_dimensionality_shrink_percent, lasso_rf_RMSE['error'][0:5], s=10, c='pink', marker="p", label='Lasso')
	ax6.scatter(exp_dimensionality_shrink_percent, exp_rf_RMSE['error'][0:5], s=10, c='gray', marker="v", label='Exp')
	# plt.title(str(i)+' Random For. RMSE vs. Dimension. Red. Percent. Scatter plot')
	plt.xlabel('Dimensionality Reduction Percentage')
	plt.ylabel('RMSE')
	plt.legend();
	plt.savefig(df_pre.ABS_DIR+'Random For RMSE_dim_red_percent '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fig)

	fboxplot4, ax4_1 = plt.subplots()
	plt.boxplot([base_rf_RMSE['error'][0:5],back_rf_RMSE['error'][0:5],step_rf_RMSE['error'][0:5],for_rf_RMSE['error'][0:5],lasso_rf_RMSE['error'][0:5],exp_rf_RMSE['error'][0:5]])
	# plt.title(str(i)+' Random For. RMSE Boxplot')
	plt.xlabel('Selection Method')
	ax4_1.set_xticklabels(labels)
	plt.legend()
	plt.ylabel('RMSE')
	plt.savefig(df_pre.ABS_DIR+'Random For RMSE_all_methods '+str(i)+' .png', format='png', dpi = 800)
	plt.close(fboxplot4)

	for j in range(nofolds):

		base_prediction_values_fold = base_prediction[base_prediction['fold_no'] == j]['prediction']
		back_prediction_values_fold = back_prediction[back_prediction['fold_no'] == j]['prediction']
		step_prediction_values_fold = step_prediction[step_prediction['fold_no'] == j]['prediction']
		for_prediction_values_fold = for_prediction[for_prediction['fold_no'] == j]['prediction']
		lasso_prediction_values_fold = lasso_prediction[lasso_prediction['fold_no'] == j]['prediction']
		exp_prediction_values_fold = exp_prediction[exp_prediction['fold_no'] == j]['prediction']

		fig, axs_lin = plt.subplots(1,6,figsize = (22, 5))
		max_value_base = max(max(target_test[j]),max(base_rf_prediction[j]))
		min_value_base = min(min(target_test[j]),min(base_rf_prediction[j]))
		line_base = (min_value_base,max_value_base)
		max_value_back = max(max(target_test[j]),max(back_prediction_values_fold))
		min_value_back = min(min(target_test[j]),min(back_prediction_values_fold))
		line_back = (min_value_back,max_value_back)
		max_value_step = max(max(target_test[j]),max(step_prediction_values_fold))
		min_value_step = min(min(target_test[j]),min(step_prediction_values_fold))
		line_step = (min_value_step,max_value_step)
		max_value_for = max(max(target_test[j]),max(for_prediction_values_fold))
		min_value_for = min(min(target_test[j]),min(for_prediction_values_fold))
		line_for = (min_value_for,max_value_for)
		max_value_lasso = max(max(target_test[j]),max(lasso_prediction_values_fold))
		min_value_lasso = min(min(target_test[j]),min(lasso_prediction_values_fold))
		line_lasso = (min_value_lasso,max_value_lasso)
		max_value_exp = max(max(target_test[j]),max(exp_prediction_values_fold))
		min_value_exp = min(min(target_test[j]),min(exp_prediction_values_fold))
		line_exp = (min_value_exp,max_value_exp)
		axs_lin[0].scatter(target_test[j], base_prediction_values_fold, s=2, c='black', marker="o", label=labels[0])
		axs_lin[0].plot(line_base,line_base,'r--')
		axs_lin[0].legend()
		axs_lin[1].scatter(target_test[j], back_prediction_values_fold, s=2, c='black', marker="o", label=labels[1])
		axs_lin[1].plot(line_back,line_back,'r--')
		axs_lin[1].legend()
		axs_lin[2].scatter(target_test[j], step_prediction_values_fold, s=2, c='black', marker="o", label=labels[2])
		axs_lin[2].plot(line_step,line_step, 'r--')
		axs_lin[2].legend()
		axs_lin[3].scatter(target_test[j], for_prediction_values_fold, s=2, c='black', marker="o", label=labels[3])
		axs_lin[3].plot(line_for,line_for, 'r--')
		axs_lin[3].legend()
		axs_lin[4].scatter(target_test[j], lasso_prediction_values_fold, s=2, c='black', marker="o", label=labels[4])
		axs_lin[4].plot(line_lasso,line_for, 'r--')
		axs_lin[4].legend()
		axs_lin[5].scatter(target_test[j], exp_prediction_values_fold, s=2, c='black', marker="o", label=labels[5])
		axs_lin[5].plot(line_exp,line_for, 'r--')
		axs_lin[5].legend()

		# fig.suptitle(str(i)+' Linear Reg. Stanford Fold # '+str(j+1))
		for ax in axs_lin.flat:
			ax.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
		plt.tight_layout()

		plt.savefig(df_pre.ABS_DIR+'Lin_Stanford_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
		plt.close(fig)

		fig, axs_rf = plt.subplots(1, 6, figsize=(22, 5))
		max_value_base = max(max(target_test[j]), max(base_rf_prediction[j]))
		min_value_base = min(min(target_test[j]), min(base_rf_prediction[j]))
		line_base = (min_value_base, max_value_base)
		max_value_back = max(max(target_test[j]), max(back_rf_prediction[j]))
		min_value_back = min(min(target_test[j]), min(back_rf_prediction[j]))
		line_back = (min_value_back, max_value_back)
		max_value_step = max(max(target_test[j]), max(step_rf_prediction[j]))
		min_value_step = min(min(target_test[j]), min(step_rf_prediction[j]))
		line_step = (min_value_step, max_value_step)
		max_value_for = max(max(target_test[j]), max(for_rf_prediction[j]))
		min_value_for = min(min(target_test[j]), min(for_rf_prediction[j]))
		line_for = (min_value_for, max_value_for)
		max_value_lasso = max(max(target_test[j]), max(lasso_rf_prediction[j]))
		min_value_lasso = min(min(target_test[j]), min(lasso_rf_prediction[j]))
		line_lasso = (min_value_lasso, max_value_lasso)
		max_value_exp = max(max(target_test[j]), max(exp_rf_prediction[j]))
		min_value_exp = min(min(target_test[j]), min(exp_rf_prediction[j]))
		line_exp = (min_value_exp, max_value_exp)
		axs_rf[0].scatter(target_test[j], base_rf_prediction[j], s=2, c='black', marker="o", label=labels[0])
		axs_rf[0].plot(line_base, line_base, 'r--')
		axs_rf[0].legend()
		axs_rf[1].scatter(target_test[j], back_rf_prediction[j], s=2, c='black', marker="o", label=labels[1])
		axs_rf[1].plot(line_back, line_back, 'r--')
		axs_rf[1].legend()
		axs_rf[2].scatter(target_test[j], step_rf_prediction[j], s=2, c='black', marker="o", label=labels[2])
		axs_rf[2].plot(line_step, line_step, 'r--')
		axs_rf[2].legend()
		axs_rf[3].scatter(target_test[j], for_rf_prediction[j], s=2, c='black', marker="o", label=labels[3])
		axs_rf[3].plot(line_for, line_for, 'r--')
		axs_rf[3].legend()
		axs_rf[4].scatter(target_test[j], lasso_rf_prediction[j], s=2, c='black', marker="o", label=labels[4])
		axs_rf[4].plot(line_lasso, line_lasso, 'r--')
		axs_rf[4].legend()
		axs_rf[5].scatter(target_test[j], exp_rf_prediction[j], s=2, c='black', marker="o", label=labels[5])
		axs_rf[5].plot(line_exp, line_exp, 'r--')
		axs_rf[5].legend()

		for ax in axs_rf.flat:
			ax.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
		plt.tight_layout()

		plt.savefig(df_pre.ABS_DIR+'RF_Stanford_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
		plt.close(fig)

		if i == 'all': # Change to 'synthetic' if wishing to apply SVR

			fig = plt.figure()
			axs_svr = fig.add_subplot(111)
			max_value_svr = max(max(target_test[j]), max(svr_prediction[j]))
			min_value_svr = min(min(target_test[j]), min(svr_prediction[j]))
			line_svr = (min_value_svr, max_value_svr)
			axs_svr.scatter(target_test[j], svr_prediction[j], s=2, c='black', marker="o", label=str(i))
			axs_svr.plot(line_svr, line_svr, 'r--')
			axs_svr.legend()
			axs_svr.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
			plt.tight_layout()

			plt.savefig(df_pre.ABS_DIR+'SVR_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
			plt.close(fig)

		fig, axs_rf = plt.subplots(1,6,figsize = (22, 5))
		max_value_base = max(max(target_test[j]),max(base_rf_prediction[j]))
		min_value_base = min(min(target_test[j]),min(base_rf_prediction[j]))
		line_base = (min_value_base,max_value_base)
		max_value_back = max(max(target_test[j]),max(back_rf_prediction[j]))
		min_value_back = min(min(target_test[j]),min(back_rf_prediction[j]))
		line_back = (min_value_back,max_value_back)
		max_value_step = max(max(target_test[j]),max(step_rf_prediction[j]))
		min_value_step = min(min(target_test[j]),min(step_rf_prediction[j]))
		line_step = (min_value_step,max_value_step)
		max_value_for = max(max(target_test[j]),max(for_rf_prediction[j]))
		min_value_for = min(min(target_test[j]),min(for_rf_prediction[j]))
		line_for = (min_value_for,max_value_for)
		max_value_lasso = max(max(target_test[j]),max(lasso_rf_prediction[j]))
		min_value_lasso = min(min(target_test[j]),min(lasso_rf_prediction[j]))
		line_lasso = (min_value_lasso,max_value_lasso)
		max_value_exp = max(max(target_test[j]),max(exp_rf_prediction[j]))
		min_value_exp = min(min(target_test[j]),min(exp_rf_prediction[j]))
		line_exp = (min_value_exp,max_value_exp)
		axs_rf[0].scatter(target_test[j], base_rf_prediction[j], s=2, c='black', marker="o", label=labels[0])
		axs_rf[0].errorbar(target_test[j], base_rf_prediction[j],yerr=base_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[0].plot(line_base, line_base, 'r--')
		axs_rf[0].legend()
		axs_rf[1].scatter(target_test[j], back_rf_prediction[j], s=2, c='black', marker="o", label=labels[1])
		axs_rf[1].errorbar(target_test[j], back_rf_prediction[j],yerr=back_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[1].plot(line_back,line_back,'r--')
		axs_rf[1].legend()
		axs_rf[2].scatter(target_test[j], step_rf_prediction[j], s=2, c='black', marker="o", label=labels[2])
		axs_rf[2].errorbar(target_test[j], step_rf_prediction[j],yerr=step_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[2].plot(line_step,line_step, 'r--')
		axs_rf[2].legend()
		axs_rf[3].scatter(target_test[j], for_rf_prediction[j], s=2, c='black', marker="o", label=labels[3])
		axs_rf[3].errorbar(target_test[j], for_rf_prediction[j],yerr=for_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[3].plot(line_for,line_for, 'r--')
		axs_rf[3].legend()
		axs_rf[4].scatter(target_test[j], lasso_rf_prediction[j], s=2, c='black', marker="o", label=labels[4])
		axs_rf[4].errorbar(target_test[j], lasso_rf_prediction[j],yerr=lasso_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[4].plot(line_lasso,line_lasso, 'r--')
		axs_rf[4].legend()
		axs_rf[5].scatter(target_test[j], exp_rf_prediction[j], s=2, c='black', marker="o", label=labels[5])
		axs_rf[5].errorbar(target_test[j], exp_rf_prediction[j],yerr=exp_ci_95_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[5].plot(line_exp,line_exp, 'r--')
		axs_rf[5].legend()

		# fig.suptitle(str(i)+' RandomForest Conformal Pred. Stanford Fold # '+str(j+1))
		for ax in axs_rf.flat:
			ax.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
		plt.tight_layout()

		plt.savefig(df_pre.ABS_DIR+'RF_Stanford_Conformal_95_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
		plt.close(fig)

		fig, axs_rf = plt.subplots(1,6,figsize = (22, 5))
		max_value_base = max(max(target_test[j]),max(base_rf_prediction[j]))
		min_value_base = min(min(target_test[j]),min(base_rf_prediction[j]))
		line_base = (min_value_base,max_value_base)
		max_value_back = max(max(target_test[j]),max(back_rf_prediction[j]))
		min_value_back = min(min(target_test[j]),min(back_rf_prediction[j]))
		line_back = (min_value_back,max_value_back)
		max_value_step = max(max(target_test[j]),max(step_rf_prediction[j]))
		min_value_step = min(min(target_test[j]),min(step_rf_prediction[j]))
		line_step = (min_value_step,max_value_step)
		max_value_for = max(max(target_test[j]),max(for_rf_prediction[j]))
		min_value_for = min(min(target_test[j]),min(for_rf_prediction[j]))
		line_for = (min_value_for,max_value_for)
		max_value_lasso = max(max(target_test[j]),max(lasso_rf_prediction[j]))
		min_value_lasso = min(min(target_test[j]),min(lasso_rf_prediction[j]))
		line_lasso = (min_value_lasso,max_value_lasso)
		max_value_exp = max(max(target_test[j]),max(exp_rf_prediction[j]))
		min_value_exp = min(min(target_test[j]),min(exp_rf_prediction[j]))
		line_exp = (min_value_exp,max_value_exp)
		axs_rf[0].scatter(target_test[j], base_rf_prediction[j], s=2, c='black', marker="o", label=labels[0])
		axs_rf[0].errorbar(target_test[j], base_rf_prediction[j],yerr=base_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[0].plot(line_base, line_base, 'r--')
		axs_rf[0].legend()
		axs_rf[1].scatter(target_test[j], back_rf_prediction[j], s=2, c='black', marker="o", label=labels[1])
		axs_rf[1].errorbar(target_test[j], back_rf_prediction[j],yerr=back_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[1].plot(line_back,line_back,'r--')
		axs_rf[1].legend()
		axs_rf[2].scatter(target_test[j], step_rf_prediction[j], s=2, c='black', marker="o", label=labels[2])
		axs_rf[2].errorbar(target_test[j], step_rf_prediction[j],yerr=step_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[2].plot(line_step,line_step, 'r--')
		axs_rf[2].legend()
		axs_rf[3].scatter(target_test[j], for_rf_prediction[j], s=2, c='black', marker="o", label=labels[3])
		axs_rf[3].errorbar(target_test[j], for_rf_prediction[j],yerr=for_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[3].plot(line_for,line_for, 'r--')
		axs_rf[3].legend()
		axs_rf[4].scatter(target_test[j], lasso_rf_prediction[j], s=2, c='black', marker="o", label=labels[4])
		axs_rf[4].errorbar(target_test[j], lasso_rf_prediction[j],yerr=lasso_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[4].plot(line_lasso,line_lasso, 'r--')
		axs_rf[4].legend()
		axs_rf[5].scatter(target_test[j], exp_rf_prediction[j], s=2, c='black', marker="o", label=labels[5])
		axs_rf[5].errorbar(target_test[j], exp_rf_prediction[j],yerr=exp_ci_98_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[5].plot(line_exp,line_exp, 'r--')
		axs_rf[5].legend()

		# fig.suptitle(str(i)+' RandomForest Conformal Pred. Stanford Fold # '+str(j+1))
		for ax in axs_rf.flat:
			ax.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
		plt.tight_layout()

		plt.savefig(df_pre.ABS_DIR+'RF_Stanford_Conformal_98_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
		plt.close(fig)

		fig, axs_rf = plt.subplots(1,6,figsize = (22, 5))
		max_value_base = max(max(target_test[j]),max(base_rf_prediction[j]))
		min_value_base = min(min(target_test[j]),min(base_rf_prediction[j]))
		line_base = (min_value_base,max_value_base)
		max_value_back = max(max(target_test[j]),max(back_rf_prediction[j]))
		min_value_back = min(min(target_test[j]),min(back_rf_prediction[j]))
		line_back = (min_value_back,max_value_back)
		max_value_step = max(max(target_test[j]),max(step_rf_prediction[j]))
		min_value_step = min(min(target_test[j]),min(step_rf_prediction[j]))
		line_step = (min_value_step,max_value_step)
		max_value_for = max(max(target_test[j]),max(for_rf_prediction[j]))
		min_value_for = min(min(target_test[j]),min(for_rf_prediction[j]))
		line_for = (min_value_for,max_value_for)
		max_value_lasso = max(max(target_test[j]),max(lasso_rf_prediction[j]))
		min_value_lasso = min(min(target_test[j]),min(lasso_rf_prediction[j]))
		line_lasso = (min_value_lasso,max_value_lasso)
		max_value_exp = max(max(target_test[j]),max(exp_rf_prediction[j]))
		min_value_exp = min(min(target_test[j]),min(exp_rf_prediction[j]))
		line_exp = (min_value_exp,max_value_exp)
		axs_rf[0].scatter(target_test[j], base_rf_prediction[j], s=2, c='black', marker="o", label=labels[0])
		axs_rf[0].errorbar(target_test[j], base_rf_prediction[j],yerr=base_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[0].plot(line_base, line_base, 'r--')
		axs_rf[0].legend()
		axs_rf[1].scatter(target_test[j], back_rf_prediction[j], s=2, c='black', marker="o", label=labels[1])
		axs_rf[1].errorbar(target_test[j], back_rf_prediction[j],yerr=back_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[1].plot(line_back,line_back,'r--')
		axs_rf[1].legend()
		axs_rf[2].scatter(target_test[j], step_rf_prediction[j], s=2, c='black', marker="o", label=labels[2])
		axs_rf[2].errorbar(target_test[j], step_rf_prediction[j],yerr=step_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[2].plot(line_step,line_step, 'r--')
		axs_rf[2].legend()
		axs_rf[3].scatter(target_test[j], for_rf_prediction[j], s=2, c='black', marker="o", label=labels[3])
		axs_rf[3].errorbar(target_test[j], for_rf_prediction[j],yerr=for_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[3].plot(line_for,line_for, 'r--')
		axs_rf[3].legend()
		axs_rf[4].scatter(target_test[j], lasso_rf_prediction[j], s=2, c='black', marker="o", label=labels[4])
		axs_rf[4].errorbar(target_test[j], lasso_rf_prediction[j],yerr=lasso_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[4].plot(line_lasso,line_lasso, 'r--')
		axs_rf[4].legend()
		axs_rf[5].scatter(target_test[j], exp_rf_prediction[j], s=2, c='black', marker="o", label=labels[5])
		axs_rf[5].errorbar(target_test[j], exp_rf_prediction[j],yerr=exp_int_rf[j]['erri'].values[:-1],linestyle="None")
		axs_rf[5].plot(line_exp,line_exp, 'r--')
		axs_rf[5].legend()

		# fig.suptitle(str(i)+' RandomForest Conformal Pred. Stanford Fold # '+str(j+1))
		for ax in axs_rf.flat:
			ax.set(xlabel='True positioning error (m)', ylabel='Predicted positioning error (m)')
		plt.tight_layout()

		plt.savefig(df_pre.ABS_DIR+'RF_Stanford_Conformal_Integrity_'+str(i)+'_Fold_'+str(j+1)+'.png', format='png', dpi = 800)
		plt.close(fig)

	print("Finished all plots for  "+str(i)+"...")

print("Finishing...")
end_time = time.time()
duration = end_time - start_time
print("Finished: Execution time: "+str(duration))