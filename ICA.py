import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

#Anomalies list from newspaper database. Each point repesent the index in the time series
anom_delhi = [70, 102, 160, 306, 390, 405, 406, 409, 518, 521, 551, 597, 637, 819, 826, 887, 915, 1000, 1014, 1102, 1103, 1104, 1106, 1107, 1280, 1317, 1362, 1376, 1406, 1413, 1428, 1434, 1441, 1445, 1448, 1462, 1483, 1560, 1581, 1658, 1711, 1724, 1758, 1816, 1817, 1818, 1819, 1820, 1822, 1824, 1826, 1828, 1829, 1833, 1834, 1837, 1838, 1839, 1841, 1854, 1861, 1875, 1896, 1945, 2050, 2064, 2071, 2134, 2190, 2204, 2299, 2394, 2553, 2589, 2594, 2757, 2781, 2782, 2783, 2786, 2789, 2791, 2792, 2794, 2798, 2814, 2817, 2818, 2819, 2820, 2821, 2845, 2852, 2853, 2854, 2855, 2856, 2861, 2867, 2874, 2876, 2894, 2895, 2909, 3030, 3070, 3084, 3091, 3093, 3097, 3100, 3103, 3104, 3106, 3108, 3110, 3129, 3176]
anom_mumbai = [70, 96,	100, 102, 103, 167, 180, 191, 316, 406, 410, 412, 423, 543, 597, 601, 630, 632, 638, 639, 641, 643, 645, 647, 652, 826, 828, 887, 1057, 1074, 1101,	1102, 1279, 1376, 1406, 1413, 1427, 1434, 1439, 1441, 1448, 1459, 1462, 1476, 1550, 1581, 1588, 1599, 1602, 1637, 1638, 1759, 1781, 1784, 1804, 1805, 1810, 1816, 1817, 1818, 1819, 1820, 1824, 1826, 1829, 1830, 1832, 1833, 1834, 1835, 1837, 1838, 1839, 1840, 1841, 1847, 1848, 1852, 1854, 1856, 1861, 1867, 1869, 1873, 1888, 1938, 1945, 2022, 2023, 2043, 2050, 2064, 2074, 2092, 2099, 2129, 2134, 2185, 2201, 2219, 2569, 2738, 2745, 2748, 2754, 2756, 2780, 2782, 2783, 2784, 2786, 2793, 2808, 2814, 2817, 2818, 2820, 2821, 2832, 2845, 2847, 2854, 2855, 2860, 2864, 2895, 2906, 2909, 2962, 3030, 3073, 3074, 3084, 3085, 3093, 3098, 3099, 3103, 3104, 3106, 3107, 3108, 3110, 3125, 3167, 3200]
anom_lucknow = [102, 171, 176, 191, 308, 364, 406, 409, 600, 601, 628, 642, 822, 825, 835, 848, 892, 929, 1102, 1293, 1296, 1406, 1410, 1441, 1445, 1448, 1449, 1462, 1475, 1476, 1487, 1525, 1539, 1581, 1623, 1637, 1651, 1658, 1708, 1721, 1729, 1777, 1816, 1817, 1818, 1819, 1826, 1833, 1834, 1835, 1836, 1837, 1839, 1840, 1842, 1846, 1847, 1854, 1861, 1875, 1957, 1963, 2023, 2050, 2064, 2090, 2298, 2328, 2401, 2450, 2591, 2782, 2784, 2786, 2789, 2794, 2814, 2817, 2818, 2819, 2820, 2849, 2852, 2853, 2854, 2855, 2863, 2874, 2876, 2894, 2909, 2933, 2988, 3030, 3070]

'''
index for all the centres are as Follows:
1: BHUBANESHWAR
2: DELHI
3: LUCKNOW
4: MUMBAI
5: PATNA
'''

'''
VARIBALES Description.

S_: Independent components
A_: Mixing matrix
X : Whitened data
index: Described above
anom: anom is the list of anomalous point reported in the newspaper archive for a particular center. Set it to anom_delhi, anom_lucknow, anom_mumbai accordingly.

'''

#This method plot two ICs. Reconfigure this function incase you generate more number of ICs. 
def Plot_ICs(S_):
	labels = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(1,10)]
	plt.figure(1)
	plt.subplot(211)
	plt.plot(S_.T[0], label="IC1")
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.grid()
	plt.subplot(212)
	plt.plot(S_.T[1], label="IC2")
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.grid()
	plt.show()
	return

#reconstruction plot generation.
def rc_signal(index, S_, A_, X):
	plt.plot(X.ix[:,index-1], 'b', label="DELHI")
	plt.plot(np.dot(S_,A_[index-1].T), 'r', label="Reconstructed Singal")
	labels = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(1,10)]
	plt.xlabel("Days", fontsize=20)
	plt.ylabel("Prices", fontsize=20)
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.yticks(fontsize=20)
	plt.title("Reconstructed Signals", fontsize=20)
	plt.grid()
	plt.show()
	return

#Method geneate the residual, with static window around it. Median absolute deviation is used for setting threshold.
def residual(index, S_, A_, X):
	#plt.plot(X.ix[:,index-1], 'b', label="DELHI")
	#plt.plot(np.dot(S_,A_[index-1].T), 'r', label="Reconstructed Singal")
	r = X.ix[:, index-1] - pd.DataFrame(np.dot(S_,A_[index-1].T))[0]
	threshold = 2.7*(abs(r - r.median()).median())
	r.plot()
	plt.plot([0, 3461], [threshold, threshold], 'r')
	plt.plot([0, 3461], [-threshold,-threshold], 'r')
	labels = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(1,10)]
	plt.xlabel("Days", fontsize=20)
	plt.ylabel("Prices", fontsize=20)
	#plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.yticks(fontsize=20)
	plt.title("Reconstructed Signals", fontsize=20)
	plt.grid()
	plt.show()
	return


'''
Precision and recall.
Note this can only be calculated for Delhi, Lucknow and Mumbai. Specify anom before calling this function. i.e anom = anoms_delhi
'''
def precision_recall(index, S_, A_, X, anom):
	#plt.plot(X.ix[:,index-1], 'b', label="DELHI")
	#plt.plot(np.dot(S_,A_[index-1].T), 'r', label="Reconstructed Singal")
	residual = X.ix[:, index-1] - pd.DataFrame(np.dot(S_,A_[index-1].T))[0]
	threshold = 2.7*(abs(residual - residual.median()).median())
	X.ix[:, index-1].plot()
	positive_labels = residual[residual.ix[:] > threshold].index.tolist()
	positive_labels = positive_labels + residual[residual.ix[:] < -threshold].index.tolist()
	positively_classified = 0
	negatively_classified = 0
	covered_news_articles = set([])
	for i in positive_labels:
		b = 1
		for j in anom:
			if abs(j - i) < 25:
				b = 0
				covered_news_articles.add(j)
		if b == 1:
			negatively_classified = negatively_classified + 1
			plt.plot([i], X.ix[i, index-1], 'o', color='r')
		else:
			positively_classified = positively_classified + 1
			plt.plot([i], X.ix[i, index-1], 'o', color='k')
	print("positively_classified "+str(positively_classified)+"		negatively_classified: "+str(negatively_classified)+"		Total: "+str(len(positive_labels)))
	print("Precision: "+str(float(positively_classified)/(positively_classified+negatively_classified)))
	print("Recall: "+str(float(len(covered_news_articles))/len(anom)))
	labels = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
	d = [i*365 for i in range(1,10)]
	plt.xlabel("Days", fontsize=20)
	plt.ylabel("Prices", fontsize=20)
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.yticks(fontsize=20)
	plt.title("Reconstructed Signals", fontsize=20)
	plt.grid()
	plt.show()
	return

#Method generate the contribution of each ICs in reconstruction.
def IC_contribution(index, S_, A_, X):
	centre_name = ["BHUBANESHWAR", "DELHI", "LUCKNOW", "MUMBAI", "PATNA"]
	plt.clf()
	plt.figure(figsize=(1900/100, 1000/100), dpi=100)
	#plt.plot(X.ix[:,index-1], 'k', label=centre_name[index-1])
	plt.plot(np.dot(S_,A_[index-1].T), 'k', label="Reconstructed Singal")
	plt.plot(S_.T[0]*A_[index-1][0], 'r', label='IC1')
	plt.plot(S_.T[1]*A_[index-1][1], 'c', label='IC2')
	plt.title('Contribution of ICs for '+str(centre_name[index-1]))
	plt.xlabel("Days", fontsize=20)
	plt.ylabel("Prices", fontsize=20)
	plt.legend(loc='best')
	plt.xticks(d, labels, rotation='vertical', fontsize=20)
	plt.yticks(fontsize=20)
	plt.grid()
	#plt.show()
	plt.savefig('Contribution of ICs for '+str(centre_name[index-1]))
	return


#CALL this function to start ICA
def ICA():
	retail = pd.read_csv("processed_retail.csv")
	X = retail.ix[:,0:5].reset_index(drop=True)
	ica = FastICA(n_components=2)
	S_ = ica.fit_transform(X)
	A_ = ica.mixing_
	Plot_ICs(S_)
	#Run it for Delhi, Mumbai, Lucknow
	precision_recall(index, S_, A_, X, anom_delhi)
	for index in np.arange(1,6):
		rc_signal(index, S_, A_, X)
		residual(index, S_, A_, X)
