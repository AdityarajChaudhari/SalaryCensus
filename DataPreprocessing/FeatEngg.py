import pandas as pd
import numpy as np
from DataAcquisition.dataaccess import DataAccess
pd.set_option('display.max_columns', None)


class FeatureEngineering:

    def __init__(self):
        self.data = DataAccess()

    def replace(self):
        data = self.data.access()

        def replacevalues(col, to_replace, val):
            data[col] = data[col].replace(to_replace, val)

        replacevalues('workclass', '?', 'Private')
        replacevalues('workclass', ['Local-gov', 'State-gov', 'Federal-gov'], 'Government')
        replacevalues('workclass', ['Self-emp-not-inc', 'Self-emp-inc'], 'Self Employeed')
        replacevalues('workclass', ['Without-pay', 'Never-worked'], 'No Income')

        replacevalues('education', ['Some-college', 'Bachelors'], 'Bachelors')
        replacevalues('education', ['Assoc-voc', 'Assoc-acdm'], 'Associate')
        replacevalues('education', ['HS-grad'], 'Diploma')
        replacevalues('education', ["11th", "9th", "7th-8th", "5th-6th", "10th", "1st-4th", "12th", "Preschool"], "School")
        replacevalues('education', ['Masters', 'Doctorate'], 'Higher Studies')

        replacevalues('marital-status', ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
        replacevalues('marital-status', ['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Not Married')

        replacevalues('race', ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        replacevalues('race', ['Black'], 'Not White')

        return data

    def transform(self):
        data = self.replace()
        data['capital-gain'] = np.where(data['capital-gain']<=0, 0, 1)    # 0-No-Gain      1-Gain
        data['capital-loss'] = np.where(data['capital-loss']<=0, 1, 0)  # 0-Loss      1-No-Loss
        data['country'] = np.where(data['country'] == 'United-States', 'US', 'Non-US')
        return data

    def setrange(self):
        data = self.transform()

        def range(col, bins, labels):
            data[col] = pd.cut(data[col], bins=bins, labels=labels)

        range('hours-per-week', (1,25, 41, 100), ['Part-Time', 'Ideal-Time','Over-Time'])
        return data

    def encode(self):
        data = self.setrange()
        x = pd.get_dummies(data[['workclass', 'marital-status', 'race', 'sex', 'hours-per-week', 'country', 'salary']], drop_first=True , prefix=None)
        data = pd.concat([data, x],axis=1)
        return data

    def ordinalencoder(self):
        data = self.encode()

        def enc(col, cat_val, num_val):
            data[col] = data[col].replace(cat_val, num_val)
        enc('education', 'Higher Studies', 0)
        enc('education', 'Bachelors', 1)
        enc('education', 'Associate', 2)
        enc('education', 'Prof-school', 3)
        enc('education', 'Diploma', 4)
        enc('education', 'School', 5)

        return data

    def drop(self):

        data = self.ordinalencoder()

        def dropfeat(col):
            data.drop(col, axis=1, inplace=True)

        dropfeat(['education-num', 'occupation', 'relationship', 'workclass', 'marital-status', 'race', 'sex',
                  'hours-per-week', 'country', 'salary'])
        return data

    def removeoutliers(self):
        data = self.drop()
        data['fnlwgt'] = data['fnlwgt'].mask(data['fnlwgt'] > data['fnlwgt'].quantile(0.90), data['fnlwgt'].mean())
        return data




