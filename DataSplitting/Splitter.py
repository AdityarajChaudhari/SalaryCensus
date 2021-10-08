from sklearn.model_selection import train_test_split
from Scaler.datascaling import DataScaler


class SplitData:

    """

    ClassName  : SplitData
    Description: This class is used to split the dependent and independent features into training and testing set.
    Written By : Adityaraj Hemant Chaudhari.
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = DataScaler()

    def split(self):

        """

        Method_Name : split
        Description : Splitting the dependent and independent dataset into training and testing set.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            scalar, x, y = self.data.scale()
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise e

