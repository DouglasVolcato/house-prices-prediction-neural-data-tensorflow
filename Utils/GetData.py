import pandas as pd

class GetData:
    def __init__(self):
        self.__csv_file_path = 'data/housing.csv'

    def execute(self):
        data = pd.read_csv(self.__csv_file_path)
        data = pd.get_dummies(data, columns=['ocean_proximity'])
        data = data.drop(['longitude', 'latitude', 'population', 'households'], axis=1)
        return data