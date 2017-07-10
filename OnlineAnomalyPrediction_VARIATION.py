import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

'''
VARIATION:
Feature set created using data before and after the newspaper reported date, rather than just before it.

Code is similar to OnlineAnomalyPrediction.py.

'''
from OnlineAnomalyPrediction import *

def featureSetCreation(cname, arr, anom):
	y = retail[cname]
	tmp = arr.reset_index(drop=True).copy(deep=True)
	#Detrend
	regr = linear_model.LinearRegression()
	x = np.arange(3461).reshape(-1,1)
	regr.fit(x, y)
	t = regr.predict(np.arange(3461).reshape(-1,1))
	y = y - t
	regr.fit(x, tmp)
	t = regr.predict(np.arange(3461).reshape(-1,1))
	tmp = tmp - t
	#Uncomment to disbale detrending
	#y = retail[cname]
	#tmp = exx.reset_index(drop=True).copy(deep=True)
	center = y.reset_index(drop=True).copy(deep=True)
	arrival = tmp.copy(deep=True)
	center = (center - center.mean())/center.std()
	arrival = (arrival - arrival.mean())/arrival.std()
	#Create features
	train = pd.DataFrame(columns=np.arange(4+(7*2)))
	train_label = pd.DataFrame(np.zeros(((3461-days+1-days), 1)))
	for i in range(days-1, 3461-days):
		row = []
		x = center.ix[i-days+1:i+days].reset_index(drop=True)
		y = arrival.ix[i-days+1:i+days].reset_index(drop=True)
		row.append(x.mean())
		row.append(x.skew())
		row.append(x.max())
		row.append(x.min())
		row.append(x.std())
		if max(x) != min(x):
			row.append(x.kurtosis())
		else:
			row.append(0)
		row.append(len(signal.find_peaks_cwt(x, np.arange(1,10))))
		row.append(y.mean())
		row.append(y.skew())
		row.append(y.max())
		row.append(y.min())
		row.append(y.std())
		if max(y) != min(y):
			row.append(y.kurtosis())
		else:
			row.append(0)
		row.append(len(signal.find_peaks_cwt(y, np.arange(1,10))))
		l = regr.fit(np.arange(days*2).reshape(-1,1), np.array(x)).predict(np.arange(days*2).reshape(-1,1)).tolist()
		row.append((l[2*days-1] - l[0])/(2*days))
		l = np.diff(x,1)
		l = regr.fit(np.arange(2*days-1).reshape(-1,1), np.array(l)).predict(np.arange(2*days-1).reshape(-1,1)).tolist()
		row.append((l[2*days-2] - l[0])/(2*days))
		l = regr.fit(np.arange(2*days).reshape(-1,1), np.array(y)).predict(np.arange(2*days).reshape(-1,1)).tolist()
		row.append((l[2*days-1] - l[0])/(2*days))
		l = np.diff(y,1)
		l = regr.fit(np.arange(2*days-1).reshape(-1,1), np.array(l)).predict(np.arange(2*days-1).reshape(-1,1)).tolist()
		row.append((l[2*days-2] - l[0])/(2*days))
		train.loc[i-days+1] = row
		for ele in anom:
			if ele == i:
				train_label.ix[i-days+1] = 1 
				break
	return train, train_label


days = 15
'''
LUCKNOW, DELHI, MUMBAI
Note: CAPS ON for cname

anom: anom_lucknow, anom_mumbai, anom_delhi
'''
cname = "LUCKNOW"
wndw = 21
anom = anom_lucknow

def function():
	#Delhi, Lucknow: arrival_india
	#mumbai: arrival_mumbai
	X, Y = featureSetCreation(cname, arrival_india, anom)
	X = X.ix[0:3272,:]
	Y = Y.ix[0:3272]
	#test train division
	train = X.ix[0:2556,:]
	train_label = Y.ix[0:2556,:]
	test = X.ix[2557:,:].reset_index(drop=True)
	dtree_result = DT(train, train_label, test)
	visualizeAnom(cname, dtree_result, anom)
	#calculate precision and recall
	precision_recall(dtree_result, wndw, anom, days)