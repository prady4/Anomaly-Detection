############################
### Description on CSVs  ###
############################


=======================================================================================================================================================
=======================================================================================================================================================
|	FILES					|	DESCRIPTION
=======================================================================================================================================================
=======================================================================================================================================================
|	mandis.csv				|	This CSV contains info about all mandis present across India.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	Retail.csv				|	This CSV contains retail time series of all centers present across India.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	WS.csv					|	This CSV contains arrivals and mandi prices time series for all mandis present across India.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	processed_retail.csv			|	This CSV contains interpolated retail time series of selected 5 centers.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	MaharastraArrivals.csv			|	Quantity of arrivals time series for the state of Maharastra.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	NationalArrivals.csv			|	Quantity of arrivals time series for entire Nation.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	export.csv				|	Monthly export price time series for entire India.
-------------------------------------------------------------------------------------------------------------------------------------------------------
|	oil.csv					|	Oil prices time series logged on bi-monthly basis. Data is obtained from "https://www.iocl.com/Product_PreviousPrice/DieselPreviousPrice.aspx"
-------------------------------------------------------------------------------------------------------------------------------------------------------



##########################
##  Structure of CSVs:  ##
##########################

1. public   | centres.csv            |

Column      |          Type          |                   
------------+------------------------+------------------------------------------------------------
 centreid   | integer                |
 statecode  | integer                | 
 centrename | character varying(100) | 
 longitude  | numeric                | 
 latitude   | numeric                | 



2. public | Retail.csv       | table    | postgres

   Column   |  Type   | 
------------+---------+-----------
 dateofdata | date    | 
 centreid   | integer | 
 price      | numeric | 

 * Database data since 1997 for some centres. But we are considering frm 1st Jaunuary 2006
 * Last date of data : 2016-01-16



3. public | mandis.csv                | table    | postgres


  Column   |          Type          |
-----------+------------------------+------------------------------------------------------------
 mandicode | integer                |
 mandiname | character varying(200) | 
 statecode | integer                |
 latitude  | numeric                | 
 longitude | numeric                | 
 centreid  | integer  


4.   public	 |	    WS.csv    	  |

     Column      |          Type          | 
-----------------+------------------------+-----------
 dateofdata      | date                   | 
 mandicode       | integer                | 
 arrivalsintons  | numeric                | 
 origin          | character varying(50)  | 
 variety         | character varying(100) | 
 minpricersqtl   | numeric                | 
 maxpricersqtl   | numeric                | 
 modalpricersqtl | numeric                | 

 * Start: 2006-01-01
 * End:


5.   public	 |	MaharastraArrivals.csv    	  |

     Column      |          Type          | 
-----------------+------------------------+-----------
 dateofdata      | date                   | 
 arrivalsintons  | integer                |

 * Start: 2006-01-01
 * End:

6.   public	 |	NationalArrivals.csv    	  |

     Column      |          Type          | 
-----------------+------------------------+-----------
 dateofdata      | date                   | 
 arrivalsintons  | integer                | 



Preprocess arrivals data. Along with that, MaharastraArrival.csv, NationalArrival.csv contains data generated by this file.


