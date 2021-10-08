import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ModelBuilding.model import Model
from DataSplitting.Splitter import SplitData


class HyperParameterTuner:

    """

    Class_Name : ParameterTuning
    Description: This Class is used to perform hyperparamter tuning on the Random Forest model.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = SplitData()
        self.model = Model()

    def etctuner(self):

        """

        Method Name : etc_tuner
        Description : This method is used to choose the best parameters for ExtraTreesClassifier and assigning it with set of values which can help our model to give better results.
        output      : generalized model
        On_failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            etc = self.model.extratress()
            params = {
                 'n_estimators': [int(i) for i in range(100, 2000, 100)],
                 'criterion': ['gini', 'entropy'],
                 'min_samples_leaf': [int(x) for x in range(1, 50, 1)],
                 'min_samples_split': [int(x) for x in range(2, 50, 1)],
                 'max_features': ['sqrt', 'log2', None]
            }
            etc_grid = RandomizedSearchCV(estimator=etc, param_distributions=params, n_iter=100, cv=3, n_jobs=-1, verbose=True, random_state=100)
            etc_grid.fit(x_train, y_train)
            etc_best = etc_grid.best_params_
            print(etc_best)
            # print(xgbc_best.get_booster().feature_names)
            etc_clf = ExtraTreesClassifier(criterion=etc_best['criterion'], n_estimators=etc_best['n_estimators'],min_samples_leaf=etc_best['min_samples_leaf'],min_samples_split=etc_best['min_samples_split'],max_features=etc_best['max_features'])
            etc_clf.fit(x_train, y_train)
            return etc_clf
        except Exception as e:
            raise e

    def rfctuner(self):

        """

        Method Name : rfc_tuner
        Description : This method is used to choose the best parameters for RandomForestClassifier and assigning it with set of values which can help our model to give better results.
        output      : generalized model
        On_failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            rfc = self.model.randomforest()
            params = {
                'n_estimators': [int(x) for x in np.linspace(start=100,stop=3000,num=30)],
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [int(x) for x in range(1, 25, 1)],
                'min_samples_split': [int(x) for x in range(2, 50, 1)],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [int(x) for x in range(1,30,5)]
            }

            rfc_grid = RandomizedSearchCV(estimator=rfc, param_distributions=params, n_iter=50, cv=5, n_jobs=-1, verbose=2,
                                          random_state=100)
            rfc_grid.fit(x_train, y_train)
            rfc_best = rfc_grid.best_params_
            print(rfc_best)
            # print(xgbc_best.get_booster().feature_names)
            rfc_clf = RandomForestClassifier(criterion=rfc_best['criterion'], n_estimators=rfc_best['n_estimators'],
                                           min_samples_leaf=rfc_best['min_samples_leaf'],
                                           min_samples_split=rfc_best['min_samples_split'],
                                           max_features=rfc_best['max_features'], max_depth=rfc_best['max_depth'])
            rfc_clf.fit(x_train, y_train)
            return rfc_clf
        except Exception as e:
            raise e


    def bgctuner(self):

        """

        Method Name : bgc_tuner
        Description : This method is used to choose the best parameters for BaggingClassifier and assigning it with set of values which can help our model to give better results.
        output      : generalized model
        On_failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            bgc = self.model.bagging()
            knc1 = KNeighborsClassifier()
            knc1.fit(x_train, y_train)
            # svc1 = SVC()
            # svc1.fit(x_train, y_train)
            dtc1 = DecisionTreeClassifier()
            dtc1.fit(x_train, y_train)
            params = {
                'base_estimator': [knc1, dtc1],
                'n_estimators': [i for i in range(100, 500, 10)]
            }
            bgc_rndm = RandomizedSearchCV(estimator=bgc, param_distributions=params, n_iter=100, cv=2, n_jobs=-1, random_state=75, verbose=True)
            bgc_rndm.fit(x_train, y_train)
            bgc_best = bgc_rndm.best_estimator_
            return bgc_best
        except Exception as e:
            raise e

    def dtctuner(self):

        """

        Method Name : dtc_tuner
        Description : This method is used to choose the best parameters for DecisionTreeClassifier and assigning it with set of values which can help our model to give better results.
        output      : generalized model
        On_failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            dtc = self.model.decisiontree()
            print(dtc)
            params = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [i for i in range(1, 20, 1)],
                'min_samples_leaf': [i for i in range(1, 20, 1)],
                'min_samples_split': [i for i in range(2, 20, 1)],
                'max_features': ['sqrt', 'log2', None],
                'ccp_alpha': [float(i) for i in np.linspace(0, 1, 10)]
            }
            dtc_rndm = RandomizedSearchCV(estimator=dtc, param_distributions=params, cv=10, n_iter=100, n_jobs=6, random_state=100, verbose=True)
            dtc_rndm.fit(x_train, y_train)
            dtc_best = dtc_rndm.best_params_
            print(dtc_best)
            # print(xgbc_best.get_booster().feature_names)
            dtc_clf = DecisionTreeClassifier(criterion=dtc_best['criterion'], max_depth = dtc_best['max_depth'], min_samples_leaf = dtc_best['min_samples_leaf'], min_samples_split = dtc_best['min_samples_split'], max_features = dtc_best['max_features'], ccp_alpha = dtc_best['ccp_alpha'])
            dtc_clf.fit(x_train, y_train)
            return dtc_clf
        except Exception as e:
            raise e

    def xgbtuner(self):

        """

        Method Name : xgbtuner
        Description : This method is used to choose the best parameters for XGBoost and assigning it with set of values which can help our model to give better results.
        output      : generalized model
        On_failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            xgbc = self.model.xgboost1()

            params = {
                "learning_rate": [float(i) for i in np.linspace(0.001,0.2,20)],
                #"max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [float(i) for i in np.linspace(0.1,1,10)],
                "colsample_bytree": [0.2,0.3, 0.4, 0.5, 0.7,0.8,0.9]
            }
            xgbc_rndm = RandomizedSearchCV(estimator=xgbc, param_distributions=params, cv=10, n_iter=50, n_jobs=6, verbose=True, random_state=75)
            xgbc_rndm.fit(x_train, y_train)
            # xgbc_rndm.fit(x_train, y_train)
            xgbc_best = xgbc_rndm.best_params_
            print(xgbc_best)
            #print(xgbc_best.get_booster().feature_names)
            final_clf = XGBClassifier(learning_rate=xgbc_best['learning_rate'],min_child_weight=xgbc_best['min_child_weight'],gamma=xgbc_best['gamma'],colsample_bytree=xgbc_best['colsample_bytree'])
            final_clf.fit(x_train, y_train)
            return final_clf

        except Exception as e:
            raise e


