import pandas
import numpy as np

restaurant_info = pandas.read_csv("restaurant_info.csv")

# Add property 'food quality' to the database for each restaurant, and filling value 'good food' for some
# of the restaurants

restaurant_info = restaurant_info.assign(food_quality=np.where(restaurant_info['food'].index % 2 == 0, 'good food', ''))
restaurant_info.to_csv("data/restaurant_info.csv")