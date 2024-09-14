import numpy as np
import pandas as pd
from funcs import get_first_cabin, get_title
from pipeline import titanic_pipe
# to divide train and test set
from sklearn.model_selection import train_test_split

# TODO turn into load data
data = pd.read_csv("datasets/train.csv")

data = data.replace('?', np.nan)

data['cabin'] = data['cabin'].apply(get_first_cabin)
data['title'] = data['name'].apply(get_title)

data['fare'] = data['fare'].astype('float')
data['age'] = data['age'].astype('float')

data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility
