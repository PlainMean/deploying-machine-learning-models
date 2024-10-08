
from columns import CABIN, CATEGORICAL_VARIABLES, NUMERICAL_VARIABLES
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (AddMissingIndicator, CategoricalImputer,
                                       MeanMedianImputer)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=CATEGORICAL_VARIABLES)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARIABLES)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=NUMERICAL_VARIABLES)),


    # Extract letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=CABIN)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, n_categories=1, variables=CATEGORICAL_VARIABLES)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=CATEGORICAL_VARIABLES)),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C=0.0005, random_state=0)),
])
