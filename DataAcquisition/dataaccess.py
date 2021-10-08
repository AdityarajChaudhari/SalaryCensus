import pandas as pd


class DataAccess:

    """

    ClassName  : DataAccess
    Description: This class is used to acquire/access data that is stored in the text file
    Written by : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : 0

    """

    def __init__(self):
        self.data_src = r'../Data/adult.csv'

    def access(self):

        """

        Method_Name : acquire_data
        Description : This method is used to acquire the data from the data source
        Output      : Pandas DataFrame
        On_Failure  : Raise Exceptions

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : 0

        """

        try:
            data = pd.read_csv(self.data_src, skipinitialspace=True)
            return data
        except Exception as e:
            raise e


