import pandas as pd

class GetData:
    def __init__(self):
        self.__csv_file_path = 'data/housing.csv'

    def execute(self):
        data = pd.read_csv(self.__csv_file_path)
        data = pd.get_dummies(data, prefix='ocean_proximity', columns=['ocean_proximity'])
        data.dropna()
        return data[['housing_median_age','total_rooms','total_bedrooms','median_income','ocean_proximity_<1H OCEAN','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN','median_house_value']]