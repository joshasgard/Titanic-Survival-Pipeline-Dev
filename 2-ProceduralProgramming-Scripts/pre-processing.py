
## Import all necessary libraries

import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


#Load the data from source
def load_data(path_to_data):
    data = pd.read_csv(path_to_data)
    return data


def 