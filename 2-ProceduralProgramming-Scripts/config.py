# ====   PATHS ===================

PATH_TO_DATASET = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
#LOCAL_PATH = 'titanic.csv'
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===========



FREQUENT_LABELS = {
    'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']}





# ===============  FEATURE GROUPS ==============

# variable groups for preprocessing steps
NUMERICAL_TO_FLOATS = ['fare', 'age']

UNNECESSARY_VAR = ['name','ticket', 'boat', 'body','home.dest']

# variable groups for engineering
TARGET_FEATURE = 'survived'

NUMERICAL_MISSING_TO_FILL = ['age', 'fare']

CATEGORICAL_MISSING_TO_FILL = ['cabin', 'embarked']

# variable groups for one-hot encoding
CATEGORICAL_ENCODE = [['sex', 'cabin', 'embarked', 'title']

#Final Re-ordered Features for training
FEATURES = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'age_na', 'fare_na',
       'sex_male', 'cabin_Missing', 'cabin_Rare', 'embarked_Q',
       'embarked_Rare', 'embarked_S', 'title_Mr', 'title_Mrs', 'title_Rare']


