import pandas as pd

class GetData:
    def __init__(self):
        self.__csv_file_path = 'data/housing.csv'

    # housing_median_age
    # total_rooms
    # total_bedrooms
    # median_income
    # median_house_value
    # ocean_proximity_<1H OCEAN
    # ocean_proximity_INLAND	
    # ocean_proximity_ISLAND	
    # ocean_proximity_NEAR BAY	
    # ocean_proximity_NEAR OCEAN
    def execute(self):
        data = pd.read_csv(self.__csv_file_path)
        data = pd.get_dummies(data, columns=['ocean_proximity'])
        data = data.drop(['longitude', 'latitude', 'population', 'households'], axis=1)
        return data