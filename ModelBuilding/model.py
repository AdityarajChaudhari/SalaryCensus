from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from DataSplitting.Splitter import SplitData
import pickle
import bz2
from sklearn.ensemble import GradientBoostingClassifier
import _pickle as cPickle
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


class Model:

    def __init__(self):
        self.data = SplitData()

    def extratress(self):
        try:
            x_train, x_test, y_train, y_test = self.data.split()
            etc = ExtraTreesClassifier()
            etc.fit(x_train, y_train)
            return etc
        except Exception as e:
            raise e

    def randomforest(self):
        try:
            x_train, x_test, y_train, y_test = self.data.split()
            rfc = RandomForestClassifier()
            rfc.fit(x_train, y_train)
            return rfc
        except Exception as e:
            raise e

    def bagging(self):
        try:
            x_train, x_test, y_train, y_test = self.data.split()
            bgc = BaggingClassifier()
            bgc.fit(x_train, y_train)
            return bgc
        except Exception as e:
            raise e

    def decisiontree(self):
        try:
            x_train, x_test, y_train, y_test = self.data.split()
            dtc = DecisionTreeClassifier()
            dtc.fit(x_train, y_train)
            return dtc
        except Exception as e:
            raise e

    def xgboost1(self):
        try:
            x_train, x_test, y_train, y_test = self.data.split()
            xgbc = XGBClassifier()

            xgbc.fit(x_train, y_train)
            return xgbc
        except Exception as e:
            raise e














