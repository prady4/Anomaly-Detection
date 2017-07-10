from OnlineAnomalyPrediction import *
	'''
	if priceSlope == 0:
		if arrivalSlope > 0:
			anomalyList.append(i)
	'''
def Rule(cname, arrival, days):
	center = retail[cname].reset_index(drop=True)
	center = (center - center.mean())/center.std()
	arrival = (arrival - arrival.mean())/arrival.std()
	anomalyList = []
	regr = linear_model.LinearRegression()
	for i in range(days-1, 3461):
		x = center.ix[i-days+1:i].reset_index(drop=True)
		y = arrival.ix[i-days+1:i].reset_index(drop=True)
		#Slope of Price time series
		priceTrend = regr.fit(np.arange(days).reshape(-1,1), np.array(x)).predict(np.arange(days).reshape(-1,1)).tolist()
		priceSlope = (priceTrend[days-1] - priceTrend[0])/days
		#Slope of Arrival TS
		arrivalTrend = regr.fit(np.arange(days).reshape(-1,1), np.array(y)).predict(np.arange(days).reshape(-1,1)).tolist()
		arrivalSlope = (arrivalTrend[days-1] - arrivalTrend[0])/days
		if priceSlope > 0.015:
			if arrivalSlope >= 0:
				anomalyList.append(i)
	return anomalyList

def precision_recall(anomalyList, anom):
	positive_labels = anomalyList
	positively_classified = 0
	negatively_classified = 0
	covered_news_articles = set([])
	for i in positive_labels:
		b = 1
		for j in anom:
			if (j - i) < 21 and (j - i) >= 0:
				b = 0
				covered_news_articles.add(j)
		if b == 1:
			negatively_classified = negatively_classified + 1
		else:
			positively_classified = positively_classified + 1
	print("positively_classified "+str(positively_classified)+"		negatively_classified: "+str(negatively_classified)+"		Total: "+str(len(positive_labels)))
	print("Precision: "+str(float(positively_classified)/(positively_classified+negatively_classified)))
	print("Recall: "+str(float(len(covered_news_articles))/len(anom)))

def main():
	days = 15
	cname = "LUCKNOW"
	wndw = 21
	anom = anom_lucknow
	#Delhi, Lucknow: Arrival_india
	#Mumbai: arrival_mumbai
	arrival = arrival_india.reset_index(drop=True)
	anomalyList = Rule(cname, arrival, days)
	center = retail[cname].reset_index(drop=True)
	center.plot()
	for i in anomalyList:
		plt.plot([i], center.ix[i], 'o', color='r')
	plt.plot(anomalyList[0], center.ix[anomalyList[0]], 'o', color='r', label='Detected Anomalies')
	plt.xlabel("Days", fontsize=20)
	plt.ylabel("Prices", fontsize=20)
	plt.legend(loc='best')
	plt.grid()
	plt.show()
	precision_recall(anomalyList, anom)

'''
threshold = 0.85*(abs(center - center.median()).median())
center.reset_index(drop=True).plot()
plt.plot([0, 3461], [threshold, threshold])
plt.show()

'''