from util.util import save_to_pickle, load_from_pickle, save_report
from conf.conf import logging
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def split_data(df, x_test_dir, y_test_dir, target_variable, **kwargs) -> tuple:
    """
    This function divides the dataset into training and test data, returns them in tuple,
    and also saves text data in a separate pickle file
    :param df: Dataframe for splitting
    :param x_test_dir: The path to save the test X-features
    :param y_test_dir: The path to save the test target values
    :param target_variable: Name of the target variable
    :param kwargs: Parameters for the train_test_split
    :return: Split X_train (X-train features), X_test (X-test features),
    y_train (y-train target variables), y_test (y-test target variables)
    """
    logging.info('Defining X (feature variables) and Y (target variable)...')
    # Variables
    X = df.iloc[:, :-1]
    y = df[target_variable]
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


def make_preprocessor(categorical: list, numeric: list, ScalerClass, EncoderClass, save_prep_dir: str) -> 'sklearn.compose._column_transformer.ColumnTransformer':
    """
    This function forms a preprocessor for categorical and numeric features
    :param categorical: List of the categorical columns (features)
    :param numeric: List of the numeric columns (features)
    :param ScalerClass: Scaler class
    :param EncoderClass: Encoder class
    :param save_prep_dir: The path to preprocessor for opening/saving/updating
    :return: Preprocessor with the Scaler (for numeric) & Encoder (for categorical) classes
    """
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

    logging.info(f'Checking if {type(ScalerClass).__name__}, {type(EncoderClass).__name__} preprocessor exists...')
    try:
        previous_preprocessor = load_from_pickle(dir=save_prep_dir)
        if previous_preprocessor == preprocessor:
            logging.info(
                f'Preprocessor with {type(ScalerClass).__name__}, {type(EncoderClass).__name__} exists.')
            preprocessor = previous_preprocessor
        else:
            logging.info(
                f'Preprocessor with {type(ScalerClass).__name__}, {type(EncoderClass).__name__} exists.')
            preprocessor = preprocessor
    except FileNotFoundError:
        preprocessor = preprocessor

    logging.info(f'Saving {type(preprocessor).__name__}'
                 f'(categorical: {type(EncoderClass).__name__}, numeric: {type(ScalerClass).__name__}) begins ...')
    save_to_pickle(dir=save_prep_dir, model=preprocessor)
    logging.info(f'Saving {type(preprocessor).__name__}'
                 f'(categorical: {type(EncoderClass).__name__}, numeric: {type(ScalerClass).__name__}) is over.')

    return preprocessor


def make_pipeline(PreprocessorClass, ModelClass) -> 'sklearn.pipeline.Pipeline':
    """
    This function forms a pipeline from a preprocessor and a model.
    :param PreprocessorClass: Preprocessor class
    :param ModelClass: Model class
    :return: Pipeline with defined Preprocessor & Model classes
    """
    logging.info(f'Building Pipeline with {type(PreprocessorClass).__name__}, {type(ModelClass).__name__} begins...')
    pipeline = Pipeline(
        steps=[
            ('preprocessor', load_from_pickle(PreprocessorClass)),
            ('regressor', ModelClass)
        ])
    logging.info(f'Building Pipeline with {type(PreprocessorClass).__name__}, {type(ModelClass).__name__} is over.')

    return pipeline


def cross_validation(CrossValClass, cv_params: dict) -> 'sklearn.model_selection._split.KFold':
    """
    This function defines cross-validator.
    :param CrossValClass: Cross-validator class
    :param cv_params: Parameters for cross-validator class
    :return: Cross-validator class with the specified parameters
    """
    kf = CrossValClass(**cv_params)
    return kf


def search_parameters(pipeline,
                      X_train, y_train,
                      CrossValClass=KFold, cv_params={},
                      SearchParamsClass=GridSearchCV, sp_params={}) -> 'sklearn.pipeline.Pipeline':
    """
    This function selects the optimal hyperparameters of the model
    :param pipeline: Pipeline with column transformer and model for hyperparameters tuning
    :param X_train: DataFrame of X-train features
    :param y_train: Array of y-train (target) values
    :param CrossValClass: Cross-validator class
    :param cv_params: Parameters for cross-validator class
    :param SearchParamsClass: Search parameters class for searching over specified parameter values for an estimator
    :param sp_params: Specified parameter values for searching for an estimator
    :return: The best estimator (pipeline) with the best tuned parameters
    """
    kf = cross_validation(CrossValClass, cv_params)
    logging.info(f'Searching for best hyperparameters {SearchParamsClass.__name__} begins...')
    grid = SearchParamsClass(pipeline, cv=kf, **sp_params)
    grid.fit(X_train, y_train)
    logging.info(f'Searching for best hyperparameters {SearchParamsClass.__name__} is over.')

    return grid.best_estimator_


def fit_pipeline(pipeline, X_train, y_train, save_model_dir: str) -> tuple:
    """
    This function fits the pipeline using training data
    :param pipeline: Pipeline with column transformer and model
    :param X_train: DataFrame of X-train features
    :param y_train: Array of y-train (target) values
    :param save_model_dir: The path to model (pipeline) for opening/saving/updating
    :return: Formed pipeline, train accuracy score
    """
    try:
        previous_model = load_from_pickle(dir=save_model_dir)
        if previous_model != pipeline:
            model = previous_model
        else:
            model = pipeline
    except FileNotFoundError:
        model = pipeline

    # Train the pipeline
    logging.info(f'Training {type(pipeline).__name__} begins...')
    model.fit(X_train, y_train)
    logging.info(f'Training {type(model).__name__} is over.')

    logging.info(f'Saving {type(model).__name__}...')
    save_to_pickle(dir=save_model_dir, model=model)
    logging.info(f'{type(model).__name__} is saved.')

    accuracy_score = round(model.score(X_train, y_train) * 100, 2)
    logging.info(f'Train accuracy of {type(model).__name__} (model: {type(model[1]).__name__}) is {accuracy_score}')

    return model, accuracy_score


def predict_values(path_to_x_test: str, path_to_model: str, path_to_y_preds: str) -> 'numpy.ndarray':
    """
    This function predicts target values using a ready-made pipeline
    :param path_to_x_test: The path to open the test X-features for the pipeline prediction
    :param path_to_model: The path to model (pipeline) for opening
    :param path_to_y_preds: The path to save the values predicted by the pipeline
    :return: Numpy array of predicted values
    """
    pipeline = load_from_pickle(dir=path_to_model)
    X_test = load_from_pickle(dir=path_to_x_test)

    logging.info(
        f'Predicting target values with {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    y_preds = pipeline.predict(X_test)
    logging.info(
        f'Predicting target values with {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')

    logging.info(
        f'Saving predicted target values of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    save_to_pickle(dir=path_to_y_preds, model=y_preds)
    logging.info(
        f'Saving predicted target values of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')
    return y_preds


def results_evaluation(path_to_y_preds: str, path_to_y_test: str, path_to_model: str, model_report_dir: str) -> str:
    """
    This function generates results evaluation report, including: confusion matrix, accuracy score,
    classification report (precision, recall, f1-score, support, accuracy, macro avg, weighted avg)
    :param path_to_y_preds: The path to open predicted by the pipeline & saved target values
    :param path_to_y_test: The path to open the test target values
    :param path_to_model: The path to model (pipeline) for opening
    :param model_report_dir: The path to the model (pipeline) execution report
    :return: Pipeline perform report
    """
    y_preds = load_from_pickle(dir=path_to_y_preds)
    y_test = load_from_pickle(dir=path_to_y_test)
    pipeline = load_from_pickle(dir=path_to_model)

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

    logging.info(
        f'Saving classification report of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) begins...')
    save_report(dir=model_report_dir, report=clf_report)
    logging.info(
        f'Saving classification report of {type(pipeline).__name__} (model: {type(pipeline[1]).__name__}) is over.')

    return clf_report