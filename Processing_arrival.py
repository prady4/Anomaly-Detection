import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#Generate Arrival data
mandi = pd.read_csv("mandis.csv") #, header=None
mandi.columns = ["mid", "mname", "scode", "lati", "long", "cid"]
ws = pd.read_csv("WS.csv", header=None)
ws.columns = ["date", "mid", "arrival", "origin", "var", "min", "max", "price"]

#Mandis with possible data Entry problems. Unreseasonal spikes notices.
Mandi_IDs_With_Possible_Data_Entry_Problem = set([698, 193, 583, 329, 298, 9, 591, 174, 227, 627, 741])

#generate India Produce
def SumAllArrivals():
	India_produce = pd.DataFrame()
	i = 0
	for x in range(1, 1600):
		if x not in Mandi_IDs_With_Possible_Data_Entry_Problem:
			f = ws[ws["mid"] == x]
			f = f[f["date"] >= "2006-01-01"]
			f = f[f["date"] <= "2015-06-23"]
			#f.drop(f.columns[[3,4,5,6]], axis=1, inplace=True)
			f = f.sort(["date"], ascending = True)
			f = f.drop_duplicates(cols='date', take_last=True)
			f['date'] =  pd.to_datetime(f['date'], format='%Y-%m-%d')
			f.set_index('date', inplace=True)
			idx = pd.date_range('2006-01-01', '2015-06-23')
			f = f.reindex(idx)
			#f = f.interpolate(method='pchip')
			f = f.fillna(method='bfill')
			f = f["arrival"]
			India_produce[x] = f
	India_produce = India_produce.sum(axis=1)
	return India_produce


#Generate Statewise produce
def stateWiseArrivals(scode):
	stateArrivals = pd.DataFrame()
	mandis_List = mandi[mandi["scode"] == scode]
	mandi_IDs = mandis_List.ix[:,0]
	for x in mandi_IDs:
		if x not in Mandi_IDs_With_Possible_Data_Entry_Problem:
			f = ws[ws["mid"] == x]
			f = f[f["date"] >= "2006-01-01"]
			f = f[f["date"] <= "2015-06-23"]
			#f.drop(f.columns[[3,4,5,6]], axis=1, inplace=True)
			f = f.sort(["date"], ascending = True)
			f = f.drop_duplicates(cols='date', take_last=True)
			f['date'] =  pd.to_datetime(f['date'], format='%Y-%m-%d')
			f.set_index('date', inplace=True)
			idx = pd.date_range('2006-01-01', '2015-06-23')
			f = f.reindex(idx)
			#f = f.interpolate(method='pchip')
			f = f.fillna(method='bfill')
			#f = f.resample('M')
			f = f["arrival"]
			stateArrivals[x] = f
	stateArrivals = stateArrivals.sum(axis=1)
	return stateArrivals