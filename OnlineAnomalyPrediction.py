import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import tree


'''
VARIABLES Description
1. days: Length of sliding window
2. cname: DELHI, LUCKNOW, MUMBAI
3. wndw: It is a Validation window. Bound the number of days within which a newspaper article is present from a predicted anomlous date. We consider if a newspaper is present within 21 days of predicted anomaly date, then it is classified correcly.
4. center_arrival: Summation of arrival for mentioned state. For Mumbai we have taken summation of Maharastra's arrival, Delhi and lucknow use the summation of arrivals for the entire nations produce. The idea here is that the mandi arrivals should also show seasonal behavior which is not the case for delhi (Huge Imports). Similarly, lucknow's arrivals were not convincing enough, So we sum the arrivals of all the mandis arrival in india for such centers. More qualitative explation is if contibution of a state in total production in india is less than 10% and doesn't show seasonal behavior then use Nations Produce instead.

'''


#Interpolated Time series in retail. Set date as index.
retail = pd.read_csv("processed_retail.csv")
retail['date'] =  pd.to_datetime(retail['date'], format='%Y-%m-%d')
retail.set_index('date', inplace=True)

#read_arrivals
arrival_india = pd.read_csv("NationalArrivals.csv", header=None)
arrival_mumbai = pd.read_csv("MaharastraArrivals.csv", header=None)

#preprocessing
arrival_india[0] =  pd.to_datetime(arrival_india[0], format='%Y-%m-%d')
arrival_india.set_index(0, inplace=True)
arrival_india = arrival_india.squeeze()
arrival_mumbai[0] =  pd.to_datetime(arrival_mumbai[0], format='%Y-%m-%d')
arrival_mumbai.set_index(0, inplace=True)
arrival_mumbai = arrival_mumbai.squeeze()



#Anomalies list from newspaper database. Each point repesent the index in the time series
anom_delhi = [70, 102, 160, 306, 390, 405, 406, 409, 518, 521, 551, 597, 637, 819, 826, 887, 915, 1000, 1014, 1102, 1103, 1104, 1106, 1107, 1280, 1317, 1362, 1376, 1406, 1413, 1428, 1434, 1441, 1445, 1448, 1462, 1483, 1560, 1581, 1658, 1711, 1724, 1758, 1816, 1817, 1818, 1819, 1820, 1822, 1824, 1826, 1828, 1829, 1833, 1834, 1837, 1838, 1839, 1841, 1854, 1861, 1875, 1896, 1945, 2050, 2064, 2071, 2134, 2190, 2204, 2299, 2394, 2553, 2589, 2594, 2757, 2781, 2782, 2783, 2786, 2789, 2791, 2792, 2794, 2798, 2814, 2817, 2818, 2819, 2820, 2821, 2845, 2852, 2853, 2854, 2855, 2856, 2861, 2867, 2874, 2876, 2894, 2895, 2909, 3030, 3070, 3084, 3091, 3093, 3097, 3100, 3103, 3104, 3106, 3108, 3110, 3129, 3176]
anom_mumbai = [70, 96,	100, 102, 103, 167, 180, 191, 316, 406, 410, 412, 423, 543, 597, 601, 630, 632, 638, 639, 641, 643, 645, 647, 652, 826, 828, 887, 1057, 1074, 1101,	1102, 1279, 1376, 1406, 1413, 1427, 1434, 1439, 1441, 1448, 1459, 1462, 1476, 1550, 1581, 1588, 1599, 1602, 1637, 1638, 1759, 1781, 1784, 1804, 1805, 1810, 1816, 1817, 1818, 1819, 1820, 1824, 1826, 1829, 1830, 1832, 1833, 1834, 1835, 1837, 1838, 1839, 1840, 1841, 1847, 1848, 1852, 1854, 1856, 1861, 1867, 1869, 1873, 1888, 1938, 1945, 2022, 2023, 2043, 2050, 2064, 2074, 2092, 2099, 2129, 2134, 2185, 2201, 2219, 2569, 2738, 2745, 2748, 2754, 2756, 2780, 2782, 2783, 2784, 2786, 2793, 2808, 2814, 2817, 2818, 2820, 2821, 2832, 2845, 2847, 2854, 2855, 2860, 2864, 2895, 2906, 2909, 2962, 3030, 3073, 3074, 3084, 3085, 3093, 3098, 3099, 3103, 3104, 3106, 3107, 3108, 3110, 3125, 3167, 3200]
anom_lucknow = [102, 171, 176, 191, 308, 364, 406, 409, 600, 601, 628, 642, 822, 825, 835, 848, 892, 929, 1102, 1293, 1296, 1406, 1410, 1441, 1445, 1448, 1449, 1462, 1475, 1476, 1487, 1525, 1539, 1581, 1623, 1637, 1651, 1658, 1708, 1721, 1729, 1777, 1816, 1817, 1818, 1819, 1826, 1833, 1834, 1835, 1836, 1837, 1839, 1840, 1842, 1846, 1847, 1854, 1861, 1875, 1957, 1963, 2023, 2050, 2064, 2090, 2298, 2328, 2401, 2450, 2591, 2782, 2784, 2786, 2789, 2794, 2814, 2817, 2818, 2819, 2820, 2849, 2852, 2853, 2854, 2855, 2863, 2874, 2876, 2894, 2909, 2933, 2988, 3030, 3070]

def featureSetCreation(cname, arrivals, days, anom):
	y = retail[cname]
	tmp = arrivals.reset_index(drop=True).copy(deep=True)
	#Detrend: Fit a linear line on TS and subtract. This accounts for increasing purchasing power of people and changed scales of anomalies over the years.
	regr = linear_model.LinearRegression()
	x = np.arange(3461).reshape(-1,1)
	regr.fit(x, y)
	t = regr.predict(np.arange(3461).reshape(-1,1))
	y = y - t
	regr.fit(x, tmp)
	t = regr.predict(np.arange(3461).reshape(-1,1))
	tmp = tmp - t
	center = y.reset_index(drop=True).copy(deep=True)
	arrival = tmp.copy(deep=True)
	#Normalize to get center and arrival time series on same scale
	center = (center - center.mean())/center.std()
	arrival = (arrival - arrival.mean())/arrival.std()
	#Data structure defination
	train = pd.DataFrame(columns=np.arange(4+(7*2)))
	train_label = pd.DataFrame(np.zeros(((3461-days+1), 1)))
	#Implementation of Sliding window
	for i in range(days-1, 3461):
		row = []
		x = center.ix[i-days+1:i].reset_index(drop=True)
		y = arrival.ix[i-days+1:i].reset_index(drop=True)
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
		#Slope of Price time series
		l = regr.fit(np.arange(days).reshape(-1,1), np.array(x)).predict(np.arange(days).reshape(-1,1)).tolist()
		row.append((l[days-1] - l[0])/days)
		#Rate of change of slope for Price TS
		l = np.diff(x,1)
		l = regr.fit(np.arange(days-1).reshape(-1,1), np.array(l)).predict(np.arange(days-1).reshape(-1,1)).tolist()
		row.append((l[days-2] - l[0])/days)
		#Slope of Arrival TS
		l = regr.fit(np.arange(days).reshape(-1,1), np.array(y)).predict(np.arange(days).reshape(-1,1)).tolist()
		row.append((l[days-1] - l[0])/days)
		#Rate of change of slope for Arrival TS
		l = np.diff(y,1)
		l = regr.fit(np.arange(days-1).reshape(-1,1), np.array(l)).predict(np.arange(days-1).reshape(-1,1)).tolist()
		row.append((l[days-2] - l[0])/days)
		train.loc[i-days+1] = row
		#Labeling based on newpspaper databse
		for ele in anom:
			if ele == i:
				train_label.ix[i-days+1] = 1 
				break
	return train, train_label

def feature_importance(clf):
	pd.DataFrame(clf.feature_importances_).plot()
	labels = ['Pr_Mean', 'Pr_skew', 'Pr_Max', 'Pr_Min', 'Pr_std', 'Pr_Kurt', 'Pr_Peaks', 'Ar_Mean', 'Ar_skew', 'Ar_Max', 'Ar_Min', 'Ar_std', 'Ar_Kurt', 'Ar_Peaks', 'Pr_Slope', 'Pr_RoC_RoC', 'Ar_Slope', 'Ar_RoC_RoC']
	d = np.arange(18)
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.yticks(fontsize=20)
	plt.grid()
	plt.show()

def DT(train, train_label, test):
	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(train, train_label)
	result = clf.predict(test)
	print(result.sum())
	feature_importance(clf)
	return result

def visualizeAnom(cname, dtree_result, anom):
	center = retail[cname].reset_index(drop=True)
	center = (center - center.mean())/center.std()
	center.plot()
	j = 0
	#plot Training labels
	for i in range(0,2557):
		if j >= len(anom):
			break
		if anom[j] == i:
			plt.plot([i], center.ix[i], 'o', color='k')
		if anom[j] <= i:
			j = j + 1
	for i in range(0, len(dtree_result)):
		if dtree_result[i] == 1:
			plt.plot([i+2557], center.ix[i+2557], 'o', color='r')
	labels = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(0,11)]
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=15)
	plt.yticks(fontsize=20)
	plt.grid()
	plt.show()



'''
Precision and recall.
Note this can only be calculated for Delhi, Lucknow and Mumbai. Specify anom before calling this function. i.e anom = anoms_delhi
'''
def precision_recall(result, wndw, anom, days):
	#list of newspaper articles reported for test set period.
	test_labels = []
	for i in anom:
		if i >= (2557 + days - 1):
			test_labels.append(i)
	positively_classified = 0
	negatively_classified = 0
	covered_news_articles = set([])
	for i in range(0, len(result)):
		if result[i] == 1:
			flag = 1
			for j in test_labels:
				if (j - (i+2557+days-1)) < wndw and (j - (i+2557+days-1)) >= 0:
					flag = 0
					covered_news_articles.add(j)
			if flag == 1:
				negatively_classified = negatively_classified + 1
			else:
				positively_classified = positively_classified + 1
	print("positively_classified "+str(positively_classified)+"		negatively_classified: "+str(negatively_classified))
	print("Precision: "+str(float(positively_classified)/(positively_classified+negatively_classified)))
	print("Recall: "+str(float(len(covered_news_articles))/len(test_labels)))
	return

def OnlineAnomalyPrediction():
	days = 15
	cname = "DELHI"
	wndw = 21
	anom = anom_delhi
	#Delhi, Lucknow: Arrival_india
	#Mumbai: arrival_mumbai
	arrivals = arrival_india
	#feature set created
	X, Y = featureSetCreation(cname, arrivals, days, anom)
	#cropping till December 2014 since only this much Newspaper database is present
	X = X.ix[0:3272,:]
	Y = Y.ix[0:3272]
	#Train Test Division
	train = X.ix[0:2556,:]
	train_label = Y.ix[0:2556,:]
	test = X.ix[2557:,:].reset_index(drop=True)
	#train your classifier, and test
	dtree_result = DT(train, train_label, test)
	#Visualize your prediction.
	visualizeAnom(cname, dtree_result, anom)
	#calculate precision and recall
	precision_recall(dtree_result, wndw, anom, days)



#Visualize Decision tree using
#tree.export_graphviz(clf, out_file='tree.dot') 