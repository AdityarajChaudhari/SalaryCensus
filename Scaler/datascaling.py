import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from DataPreprocessing.Sampler import DataSampling


class DataScaler:

    """

    ClassName  : DataScaler
    Description: This class is used to scale down the data between +1 to -1 using StandardScaler.
    Written By : Adityaraj Hemant Chaudhari.
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = DataSampling()

    def scale(self):

        """

        Method Name : scale
        Description : This method is used to scale down the independent data.
        Output      : scaled data(x)
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            x, y = self.data.sampling()
            scalar = StandardScaler()
            x = pd.DataFrame(scalar.fit_transform(x), columns=x.columns)
            print(x)
        except Exception as e:
            raise e

    def serializescalar(self):

        """

        Method Name : serializescalar
        Description : This method is used to save the scalar in the serialized format in the pickle file.
        Output      : Pickle file
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            scalar, x, y = self.scale()
            with open('Scalar.pkl', 'wb') as file:
                pickle.dump(scalar, file)
        except Exception as e:
            raise e


s = DataScaler()
s.scale()