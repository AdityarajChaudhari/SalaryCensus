from imblearn.over_sampling import RandomOverSampler
from DataPreprocessing.FeatEngg import FeatureEngineering


class DataSampling:

    """
    ClassName  : DataSampling
    Description: This class is used to separate dependent and independent features and then perform over sampling to counter the problem of imbalanced data.
    Written By : Adityaraj Hemant Chaudhari.
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = FeatureEngineering()

    def separatedepfeatures(self):

        """

        Method Name : seperatedepfeatures
        Description : This method is used to separate dependent and independent features.
        Output      : x, y
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            data = self.data.removeoutliers()
            x = data.drop('salary_>50K', axis=1)
            y = data['salary_>50K']
            return x, y
        except Exception as e:
            raise e

    def sampling(self):

        """

        Method Name : sampling
        Description : This method is used to perform random over sampling to tackle the problem of imbalanced dataset.
        Output      : x, y
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            x, y = self.separatedepfeatures()
            ros = RandomOverSampler()
            x, y = ros.fit_resample(x, y)
            return x, y
        except Exception as e:
            raise e


