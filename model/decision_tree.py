from connector.pg_connector import get_data
from util.util import save_model, load_model
from conf.conf import logging, settings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def split(df):

    logging.info('Defining X and Y')
    # Variables
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info('Splitting dataset')

    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                        y,  # dependent variable
                                                        random_state=3
                                                        )
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train):
    # Initialize the model
    clf = DecisionTreeClassifier(max_depth=3,
                                 random_state=3
                                 )

    # Train the model
    clf.fit(X_train, y_train)

    save_model(dir='model/conf/decision_tree.pkl', model=clf)

    return clf


def predict(values, path_to_model):
    clf = load_model(path_to_model)
    return clf.predict(values)


print(settings.DATA)
print(f'parameter {settings.DATA.data_set}')
df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/blob/master/supplements/data/heart.csv')
X_train,X_test, y_train, y_test = split(df)
clf = train_decision_tree(X_train, X_test)
logging.info(f'Accuracy is {clf.score(X_test, y_test)}')

responce = predict(X_test, 'model/conf/decision_tree.pkl')
logging.info(f'Prediction is {responce}')