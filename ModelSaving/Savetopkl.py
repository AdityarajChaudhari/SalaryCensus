import pickle
import blosc
import bz2
import _pickle as cPickle
from HyperparameterTuning.Tuner import HyperParameterTuner


class SaveModel:

    """

    Class Name : SaveModel
    Description: This class is used to save to finalized model in pickle file in serialized manner.
    Written By : Adityaraj Hemant Chaudhari
    Revisions  : None
    Version    : 0.1

    """

    def __init__(self):
        self.model = HyperParameterTuner()
        self.path = "model.pkl"
        self.mode = "wb"

    def save(self):

        """

        Method Name : save
        Description : This method is used to save the model to pickle file.
        Output      : Pickle file
        On_failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Revisions   : None
        Version     : 0.1

        """

        try:
            xgbc_model = self.model.xgbtuner()

            path = self.path
            mode = self.mode
            pickle.dump(xgbc_model, open(path, mode))


        except Exception as e:
            raise e



