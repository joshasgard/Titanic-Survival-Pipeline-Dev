import numpy as np 

import pre_processing as pf
import config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.split_data(data, config.TARGET_FEATURE)

#Replace interrogation marks by NaN
X_train = pf.replace_question_marks(X_train)

# extracts the title (Mr, Ms, etc) from the name variable
X_train['title'] = X_train['name'].apply(pf.get_title)

#numerical to floats
for var in config.NUMERICAL_TO_FLOATS:
    X_train[var] = pf.num_to_flts(X_train,var)

#drop unneeded features
X_train = pf.drop_unnecessary_features(X_train,config.UNNECESSARY_VAR)

#Extract first letter of cabin
X_train = pf.extract_letter(X_train)

# impute numerical variables
for var in config.NUMERICAL_MISSING_TO_FILL:
    
    # add missing indicator first
    X_train[var + '_na'] = pf.add_missing_indicator(X_train, var)

    X_train[var] = pf.impute_na(X_train, var,replacement = X_train[var].median())


# impute categorical variables
for var in config.CATEGORICAL_MISSING_TO_FILL:
    X_train[var] = pf.impute_na(X_train, var, replacement='Missing')

# Group rare labels
for var in config.CATEGORICAL_ENCODE:
    X_train[var] = pf.remove_rare_labels(X_train, var, config.FREQUENT_LABELS[var])

##########confirm if third argument in function above works ####

#One-hot encoding of categoricals
X_train = pf.encode_categorical(X_train,config.CATEGORICAL_ENCODE)

# check all dummies were added
X_train = pf.check_dummy_variables(X_train, config.FEATURES)

# train scaler and save
scaler = pf.train_scaler(X_train, config.FEATURES,
                         config.OUTPUT_SCALER_PATH)

# scale training data set
X_train = scaler.transform(X_train[config.FEATURES])

# train model and save
pf.train_model(X_train,
               y_train,
               config.OUTPUT_MODEL_PATH)

print('Finished training')


