from sklearn import metrics
from ModelBuilding.model import Model
from DataSplitting.Splitter import SplitData


class Eval:

    def __init__(self):
        self.data = SplitData()
        self.model = Model()

    def etcmodeleval(self):

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            etc_eval = self.model.extratress()
            y_pred = etc_eval.predict(x_test)
            #print("ExtrasTreeClassifier")
            test_acc = metrics.accuracy_score(y_test,y_pred)
            x_pred = etc_eval.predict(x_train)
            train_acc =  metrics.accuracy_score(y_train, x_pred)
            return test_acc, train_acc
        except Exception as e:
            raise e

    def rfcmodeleval(self):

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            rfc_eval = self.model.randomforest()
            y_pred = rfc_eval.predict(x_test)
            #print("RandomForestClassifier")
            test_acc =  metrics.accuracy_score(y_test,y_pred)
            x_pred = rfc_eval.predict(x_train)
            train_acc =  metrics.accuracy_score(y_train, x_pred)
            return test_acc, train_acc
        except Exception as e:
            raise e

    def bgcmodeleval(self):

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            bgc_eval = self.model.bagging()
            y_pred = bgc_eval.predict(x_test)
            #print("BaggingClassifier")
            test_acc =  metrics.accuracy_score(y_test,y_pred)
            x_pred = bgc_eval.predict(x_train)
            train_acc = metrics.accuracy_score(y_train, x_pred)
            return test_acc, train_acc
        except Exception as e:
            raise e

    def dtcmodeleval(self):

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            dtc_eval = self.model.decisiontree()
            y_pred = dtc_eval.predict(x_test)
            #print("DecisionTreeClassifier")
            test_acc =  metrics.accuracy_score(y_test, y_pred)
            x_pred = dtc_eval.predict(x_train)
            train_acc = metrics.accuracy_score(y_train, x_pred)
            return test_acc, train_acc
        except Exception as e:
            raise e


    def xgbcmodel(self):

        try:
            x_train, x_test, y_train, y_test = self.data.split()
            xgbc_eval = self.model.xgboost1()
            y_pred = xgbc_eval.predict(x_test)
            #print("XGBClassifier")
            test_acc =  metrics.accuracy_score(y_test,y_pred)
            #print(test_acc)
            x_pred = xgbc_eval.predict(x_train)
            train_acc =  metrics.accuracy_score(y_train, x_pred)
            #print(train_acc)
            return test_acc, train_acc
        except Exception as e:
            raise e


