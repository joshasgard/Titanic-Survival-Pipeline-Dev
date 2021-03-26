
# =================IMPORT NECESSARY LIBRARIES========================

#for processing data
import pandas as pd
import numpy as np 

#for splitting, scaling and training model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#for memoization into caches 
import joblib


# ================== PRE-PROCESSING =========================

#Load the data from source for processing and training
def load_data(path_to_data):
    return pd.read_csv(path_to_data)


#function replaces all '?' with standard numpy 'nan'
def replace_question_marks(df):
    
    return df = df.replace('?', np.nan)


# retain only the first cabin if more than 1 are available per passenger
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan


# extracts the title (Mr, Ms, etc) from the passenger name feature
def get_title(passenger):
    if re.search('Mrs', passenger):
        return 'Mrs'
    elif re.search('Mr', passenger):
        return 'Mr'
    elif re.search('Miss', passenger):
        return 'Miss'
    elif re.search('Master', passenger):
        return 'Master'
    else:
        return 'Other'


# Cast numerical variables as floats
def num_to_flts(df, var):
    return df[var].astype('float')


# Drop features
def drop_unnecessary_features(df,unnece_fts):
    df.drop(unnece_fts, axis=1, inplace = True)
    return df



# =============== FEATURE ENGINEERING =====================

#Function divides data set in train and test
def split_data(df,target_feature):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target_feature,axis=1),
                                                        df[target_feature],
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


# Extracts only the first letter and drop number
def extract_letter(df):
    df['cabin'] = df['cabin'].str[0]
    return df


#function replaces NA by value entered by user
    # or by string Missing (default behaviour)
def impute_na(df, var, replacement='Missing'):
    return df[var].fillna(replacement)


    # groups labels that are not in the frequent list into the umbrella
    # group Rare
def remove_rare_labels(df, var, frequent_labels):

    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')


#One-Hot encoding of categorical variables
def encode_categorical(df,cate_var):
    df = pd.get_dummies(df, columns = cate_var, drop_first=True)
    return df


def add_missing_dummy(df,var, value = 0):
    df[var] = value
    return df


def train_scaler(df,rearranged_columns, output_path):
    scaler = StandardScaler()
    scaler.fit(df[rearranged_columns])
    joblib.dump(scaler, output_path)
    return scaler_
  

def scale_features(df,rearranged_columns, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df[rearranged_columns])


# ====================== TRAIN MODEL ================================

def train_model(df, target, output_path):
    # initialise the model
    log_model = LogisticRegression(C=0.0005, random_state=0)
    
    # train the model
    log_model.fit(df,target)
    
    # save the model
    joblib.dump(log_model, output_path)
    
    return None


def predict(df, model):
    model = joblib.load(log_model)
    return model.predict(df)

