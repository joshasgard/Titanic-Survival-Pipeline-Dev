import pre_processing as pf
import config

# =========== scoring pipeline =========

def predict(data):

    #Replace interrogation marks by NaN
    data = pf.replace_question_marks(data)

    # extracts the title (Mr, Ms, etc) from the name variable
    data['title'] = data['name'].apply(pf.get_title)

    #numerical to floats
    for var in config.NUMERICAL_TO_FLOATS:
        data[var] = pf.num_to_flts(data,var)

    #drop unneeded features
    data = pf.drop_unnecessary_features(data,config.UNNECESSARY_VAR)

    #Extract first letter of cabin
    data = pf.extract_letter(data)

    # impute numerical variables
    for var in config.NUMERICAL_MISSING_TO_FILL:
        
        # add missing indicator first
        data[var + '_na'] = pf.add_missing_indicator(data, var)

        data[var] = pf.impute_na(data, var,replacement = data[var].median())


    # impute categorical variables
    for var in config.CATEGORICAL_MISSING_TO_FILL:
        data[var] = pf.impute_na(data, var, replacement='Missing')

    # Group rare labels
    for var in config.CATEGORICAL_ENCODE:
        data[var] = pf.remove_rare_labels(data, var, config.FREQUENT_LABELS[var]) 
        ##confirm if third argument in function above works ##

    #One-hot encoding of categoricals
    data = pf.encode_categorical(data,config.CATEGORICAL_ENCODE)

    # check all dummies were added
    data = pf.check_dummy_variables(data, config.FEATURES)

    # scale test data set
    data = pf.scale_features(data,config.FEATURES,config.OUTPUT_SCALER_PATH)

    # Predict off the data
    predictions = pf.predict(data,config.OUTPUT_MODEL_PATH)
    
    return predictions


# check the model performance

if __name__ == '__main__':

    from sklearn.metrics import accuracy_score
    import warnings
    warnings.simplefilter(action='ignore')

    #Load data

    data = pf.load_data(config.PATH_TO_DATASET)
    X_train, X_test, y_train, y_test = pf.split_data(data, config.TARGET_FEATURE)

    pred = predict(X_test)

    # evaluate
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()