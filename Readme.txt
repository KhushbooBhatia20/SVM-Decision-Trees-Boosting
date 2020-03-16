-------------------------------------------------
Implementation of Support Vector Machines, Decision Trees & Boosting:
-------------------------------------------------

In this project, I have implemented Support Vector Machine, Decision Trees and Boosting on two data sets. My first data set is 'SGEMM GPU Kernel Performance Prediction' and my second data set is 'Rain in Australia'.

-----------------
Dataset Source:
-----------------

We have used the SGEMM GPU kernel performance Data Set available for download at -

https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance 

Second Data set can be obtained from Kaggle, link to the dataset is given below –

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

---------------
Prerequisites: 
---------------

Below libraries to be installed before running code -

!pip install xgboost
!pip install graphviz
!pip install pydotplus
!pip install chart_studio

Below Packages are prerequisites to run SVM, Decision Trees and Boosting -

import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing, model_selection, metrics
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import preprocessing  
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import datasets, utils, tree
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from IPython import display
from graphviz import Source
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import learning_curve

---------------
Important Note: 
---------------

In this project, I have implemented algorithm by using subset of dataset i.e. 10,000 due to system constraints.
