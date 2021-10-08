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
            #xgbc_model = self.model.xgbtuner()
            #dtc_model = self.model.dtctuner()
            #etc_model = self.model.etctuner()
            rfc_model = self.model.rfctuner()

            #path = self.path
            mode = self.mode
            #pickle.dump(xgbc_model, open(path, mode))
            #pickle.dump(dtc_model,open('dtc.pkl',mode))
            #pickle.dump(etc_model, open('etc.pkl', mode))
            pickle.dump(rfc_model, open('rfc.pkl', mode))
            #sfile = bz2.BZ2File('etc1.pkl', 'wb')
            #pickle.dump(etc_model, sfile)

        except Exception as e:
            raise e


s = SaveModel()
s.save()