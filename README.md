# PRBundling
Analysis on Purchase Requisition (PR) bundling an ML based recommendation service

Prerequisite:
  * Install Python 3.6 https://www.python.org/downloads/release/python-368/ 
	* Install needed python libraries
		- python -m pip install --upgrade pip
		- pip install hdbcli
		- pip install pyhdb
		- pip install numpy
		- pip install sklearn
		- pip install pandas		
		- pip install kmodes
		- pip install matplotlib
		- pip install opencv-python
		
Step 1: Clone git
  * $SRC_DIR>git clone https://github.wdf.sap.corp/I050385/PRBundling

Step 2:
  * Add following OS environment variables pointing to HANA DB system
	- HOST - HANA DB instance name
	- PORT - HANA DB port number
	- USER - HANA DB user name
	- PASSWORD - HANA DB password
	- SCHEMA - HANA DB schema name
  
Step 3: Run application from CLI
  * This is 2 step process, first to get cluster 'K' through elbow graph by enabling 'getBestK' in main method 
  * Second step to get list of PR numbers for all 'K' clusters by enabling 'computePRBundlingRecommendation' in main method
	- $SRC_DIR>python PRBundlingElbow.py
	- $SRC_DIR>python PRBundlingTest.py (Some other set of features for better clusters formation)
  * In future will enable via command line parameters/automation

* Highlights:
	- Performing fields level statistical analysis (mean, mode, distinct values, e.t.c..)
	- Techniques to isolate in generic way numerical and categorical features (Feature engineering)
	- Get best 'K' cluster number using elbow method
	- Identify to run K-Means, K-Mode or K-Prototype clustering algorithms
	- Cluster data in to 'K' cluster; each cluster contains PR numbers close to cluster centroids/mediods
	- Perform cosine similarity to select PR numbers which are more closer to its centroid/mediods with having 95% confidence level
	- Group PR's within cluster with certain threshold level (say atleast 5 PR's in each cluster).

