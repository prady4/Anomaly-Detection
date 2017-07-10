'''
4 Fold Cross Validation of Online Anomaly prediction
'''

from OnlineAnomalyPrediction import *


def plot(cname, arrival, result, index, anom, days):
	#retail["MUMBAI"].reset_index(drop=True).plot()
	center = retail["MUMBAI"].reset_index(drop=True)
	center = (center - center.mean())/center.std()
	center.plot()
	arrival = (arrival - arrival.mean())/arrival.std()
	arrival.reset_index(drop=True).plot()
	for i in anom:
		plt.plot([i], center.ix[i], 'o', color='k')
	plt.plot([index+days-1, index+days-1], [-3,5], 'r')
	plt.plot([index+818+days-1, index+818+days-1], [-3,5], 'r')
	for i in range(0, len(result)):
		if result[i] == 1:
			plt.plot([i+index+days-1], center.ix[i+index+days-1], 'o', color='r')
	labels = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(0,11)]
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=15)
	plt.yticks(fontsize=20)
	plt.grid()
	plt.show()#block=False


#precision and recall
def precision_recall(start, end, result, wndw, anom, days):
	s = []
	news = set([])
	pos = 0
	neg = 0
	j = 0
	for i in anom:
		if i >= (start + days - 1) and i < (end + days - 1):
			s.append(i)
	print("news "+ str(len(s)))
	for i in range(0, len(result)):
		if result[i] == 1:
			b = 1
			for j in s:
				if abs(j - (i+start+days-1)) < wndw:
					b = 0
					news.add(j)
			if b == 1:
				neg = neg + 1
			if b == 0:
				pos = pos + 1
	print("POS: "+str(pos) + "		NEG: "+str(neg) + "		Covered News articles: " + str(len(news)))
	print("precision: " + str(float(pos)/(pos+neg)))
	print("recall: " + str(float(len(news))/len(s)))
	#print("recall: " + str(float(pos)/(pos+neg)))

def kfold():
	days = 15
	cname = "DELHI"
	wndw = 21
	anom = anom_mumbai
	arrivals = arrival_mumbai
	X, Y = featureSetCreation(cname, arrivals, days, anom)
	X = X.ix[0:3272,:]
	Y = Y.ix[0:3272]
	fold = [0, 818, 1636, 2454, len(X)]
	for i in range(0, 4):
		train = X.drop(X.index[[np.arange(fold[i], fold[i+1])]]).reset_index(drop=True)
		train_label = Y.drop(Y.index[[np.arange(fold[i], fold[i+1])]]).reset_index(drop=True)
		test = X.ix[np.arange(fold[i],fold[i+1])].reset_index(drop=True)
		test_label = Y.ix[np.arange(fold[i],fold[i+1])].reset_index(drop=True)
		#print(train_label.shape, test_label.shape)
		result = DT(train, train_label, test)
		precision_recall(fold[i], fold[i+1], result, wndw, anom, days)
		plot("DELHI", arrivals.reset_index(drop=True), result, fold[i], anom, days)



