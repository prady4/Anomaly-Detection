'''
Seasonal model: This module create clusters of year which has  similar arrival pattern. Also, seasonal time series is generated in this module from the clusters with highest similarity in arrival pattern.

Note in this approach we use Business years which starts in APRIL and ends in MARCH. 
'''

from OnlineAnomalyPrediction import *

#Detrend the time series first
regr = linear_model.LinearRegression()
#Removing Trend from time series
y = retail["MUMBAI"].reset_index(drop=True)
x = np.arange(3461).reshape(-1,1)
regr.fit(x, y)
trend = regr.predict(np.arange(3461).reshape(-1,1))
detrendedTS = y -trend

#preprocessing
tmp = retail["MUMBAI"]
tmp = tmp.reset_index()
tmp[0] = detrendedTS
tmp.set_index('date', inplace=True)
detrendedTS = tmp[0]


#Cluster of years that look similar exempting anomalous year 2010 and 2013 to learn average signals.
list1 = [2006,2008,2012]
list2 = [2007,2011]
list3 = [2009]

def clusterSingal(x, l):
    X = pd.DataFrame()
    for i in l:
        tmp = x.loc[str(i)+'-04-01':str(i+1)+'-03-31']
        tmp = tmp.reset_index(drop=True)
        X[i] = tmp
    X = (X.sum(axis=1)/len(l))
    return X

#Cluster 1, 2, 3 average signal generation
cluster1 = clusterSingal(detrendedTS, list1)
cluster2 = clusterSingal(detrendedTS, list2)
cluster3 = clusterSingal(detrendedTS, list3)


#CLUBBED years based on similar arrival singals including anomalous years to evaluate seasonal TS.
list1 = [2006,2008,2012]
list2 = [2007,2011,2013, 2014]
list3 = [2009,2010]

def seasonalTS():
    row = []
    for i in range(2006, 2015):
        if i in set(list1):
            row = row + cluster1.values.flatten().tolist()
        elif i in set(list2):
        	if i == 2007 or i == 2011:
        		row = row + cluster2.values.flatten().tolist()
        	else:
        		row = row + cluster2[:-1].values.flatten().tolist()
        else:
            row = row + cluster3.values.flatten().tolist()
    return pd.DataFrame(row)

seasonalComponent = seasonalTS()

#########################################
### Equal weighted Linear combination ###
#########################################
#trend needs to be accounted since april 2006 and till march 2015(BUSINESS YEARS). Thus, slicing trend.
prediction = seasonalComponent[0] + trend[90:-84]
plt.plot(seasonalComponent)
plt.plot(y[90:].reset_index(drop=True))
plt.plot(prediction)
plt.show()


############################################
### Learm weights for Linear combination ###
############################################
#data structure
df = pd.DataFrame()
rt = retail["MUMBAI"]
df[0] = rt.loc[str(2006)+'-04-01':str(2015)+'-03-31'].reset_index(drop=True)
df[1] = seasonalTS()
df[2] = trend[90:-84]
#learn weights
tmp = df[:]
reg = linear_model.LinearRegression()
reg.fit(tmp[[1, 2]], tmp[0])
x = df[1]*reg.coef_[0]+df[2]*reg.coef_[1]
#plots
plt.plot(x, label='Predicted')
plt.plot(df[0], label='Actual TS')
labels = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
d = [i*365 for i in range(0,11)]
plt.legend(loc='best')
plt.xticks(d, labels, rotation='vertical')
plt.yticks()
plt.grid()
plt.show()
reg.coef_
