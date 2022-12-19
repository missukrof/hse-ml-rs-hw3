# from connector.pg_connector import get_data
from util.util import save_models, load_models
from conf.conf import logging #, settings
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.neighbors import KNeighborsClassifier


def split_data(df, x_test_dir, y_test_dir, **kwargs) -> tuple:
    logging.info('Defining X (feature variables) and Y (target variable)...')
    # Variables
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info('X (feature variables) and Y (target variable) are defined.')
    logging.info('Splitting dataset begins...')
    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                        y,  # dependent variable
                                                        **kwargs
                                                        )
    logging.info('Splitting dataset is over.')

    logging.info(f'Saving X_test set (test X-features) begins...')
    X_test.to_pickle(x_test_dir)
    logging.info(f'Saving X_test set (test X-features) is over.')

    logging.info(f'Saving y_test set (test target values) begins ...')
    y_test.to_pickle(y_test_dir)
    logging.info(f'Saving y_test set (test target values) is over.')

    return X_train, X_test, y_train, y_test


def make_preprocessor(categorical, numeric, ScalerClass, EncoderClass, save_prep_dir) -> 'sklearn.compose._column_transformer.ColumnTransformer':
    logging.info(f'Building {type(ScalerClass).__name__} Pipeline begins...')
    numeric_transformer = Pipeline(
        steps=[
            ('scaler', ScalerClass)
        ])
    logging.info(f'Building {type(ScalerClass).__name__} Pipeline is over.')

    logging.info(f'Building {type(EncoderClass).__name__} Pipeline begins...')
    categorical_transformer = Pipeline(
        steps=[
            ('encoder', EncoderClass)
        ])
    logging.info(f'Building {type(EncoderClass).__name__} Pipeline is over.')

    logging.info(f'Building {type(ScalerClass).__name__}, {type(EncoderClass).__name__} preprocessor begins...')
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric),
            ('categorical', categorical_transformer, categorical),
        ])
    logging.info(f'Building {type(ScalerClass).__name__}, {type(EncoderClass).__name__} preprocessor is over.')

    logging.info(f'Saving {type(preprocessor).__name__}'
                 f'(categorical: {type(EncoderClass).__name__}, numeric: {type(ScalerClass).__name__}) begins ...')
    save_models(dir=save_prep_dir, model=preprocessor)
    logging.info(f'Saving {type(preprocessor).__name__}'
                 f'(categorical: {type(EncoderClass).__name__}, numeric: {type(ScalerClass).__name__}) is over.')

    return preprocessor


def make_pipeline(PreprocessorClass, ModelClass) -> 'sklearn.pipeline.Pipeline':
    logging.info(f'Building Pipeline with {type(PreprocessorClass).__name__}, {type(ModelClass).__name__} begins...')
    pipeline = Pipeline(
        steps=[
            ('preprocessor', load_models(PreprocessorClass)),
            ('regressor', ModelClass)
        ])
    logging.info(f'Building Pipeline with {type(PreprocessorClass).__name__}, {type(ModelClass).__name__} is over.')

    return pipeline


def cross_validation(CrossValClass, cv_params) -> 'sklearn.model_selection._split.KFold':
    kf = CrossValClass(**cv_params)
    return kf


def search_parameters(pipeline,
                      X_train, y_train,
                      CrossValClass=KFold, cv_params={},
                      SearchParamsClass=GridSearchCV, sp_params={}) -> 'sklearn.pipeline.Pipeline':
    kf = cross_validation(CrossValClass, cv_params)
    logging.info(f'Searching for best hyperparameters {SearchParamsClass.__name__} begins...')
    grid = SearchParamsClass(pipeline, cv=kf, **sp_params)
    grid.fit(X_train, y_train)
    logging.info(f'Searching for best hyperparameters {SearchParamsClass.__name__} is over.')

    return grid.best_estimator_


def fit_pipeline(pipeline, X_train, y_train, save_model_dir) -> tuple:
    # Train the pipeline
    logging.info(f'Training {type(pipeline).__name__} begins...')
    model = pipeline.fit(X_train, y_train)
    logging.info(f'Training {type(model).__name__} is over.')

    logging.info(f'Saving {type(model).__name__}...')
    save_models(dir=save_model_dir, model=model)
    logging.info(f'{type(model).__name__} is saved.')

    accuracy_score = round(model.score(X_train, y_train) * 100, 2)
    logging.info(f'Train accuracy of {type(model).__name__} (model: {type(model[1]).__name__}) is {accuracy_score}')

    return model, accuracy_score


def predict_values(path_to_x_test, path_to_model, path_to_y_preds) -> 'numpy.ndarray':
    pipeline = load_models(dir=path_to_model)
    X_test = load_models(dir=path_to_x_test)

    logging.info(
        f'Predicting target values with {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    y_preds = pipeline.predict(X_test)
    logging.info(
        f'Predicting target values with {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')

    logging.info(
        f'Saving predicted target values of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    save_models(dir=path_to_y_preds, model=y_preds)
    logging.info(
        f'Saving predicted target values of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')
    return y_preds


def results_evaluation(path_to_y_preds, path_to_y_test, path_to_model):
    y_preds = load_models(dir=path_to_y_preds)
    y_test = load_models(dir=path_to_y_test)
    pipeline = load_models(dir=path_to_model)

    logging.info(
        f'Building classification report of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    conf_matrix = confusion_matrix(y_test, y_preds)
    acc_score = accuracy_score(y_test, y_preds)

    clf_report = 'Confusion matrix\n{conf_matrix}\n' \
                 'Accuracy of {pipeline} (model: {model}): {accuracy}\n' \
                 'Classification report:\n{classification_report}'.format(conf_matrix=conf_matrix,
                                                                          pipeline=type(pipeline).__name__,
                                                                          model=type(pipeline[1]).__name__,
                                                                          accuracy=round(acc_score * 100, 2),
                                                                          classification_report=classification_report(
                                                                              y_test,
                                                                              y_preds
                                                                          )
                                                                          )
    logging.info(
        f'Building classification report of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')
    return print(clf_report)