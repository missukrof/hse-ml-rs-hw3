from constructor.model_constructor import split_data, make_preprocessor, make_pipeline, search_parameters, fit_pipeline
from connector.pg_connector import get_data
from conf.conf import settings

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

print(f"Parameter: {settings.DATA.data_set}")
df = get_data(settings.DATA.data_set)

X_train, X_test, y_train, y_test = split_data(df=df,
                                              x_test_dir=settings.DATA.x_test,
                                              y_test_dir=settings.DATA.y_test,
                                              **settings.SPLITTING.parameters_splitting)

preprocessors = make_preprocessor(categorical=settings.CATEGORICAL.cat_features,
                                  numeric=settings.NUMERIC.num_features,
                                  ScalerClass=StandardScaler(),
                                  EncoderClass=OneHotEncoder(handle_unknown='ignore'),
                                  save_prep_dir=settings.PREPROCESSOR.prep_conf)
# KNC
kn_pipeline = make_pipeline(PreprocessorClass=settings.PREPROCESSOR.prep_conf,
                            ModelClass=KNeighborsClassifier())

kn_best_estimator = search_parameters(pipeline=kn_pipeline,
                                      X_train=X_train,
                                      y_train=y_train,
                                      CrossValClass=KFold,
                                      cv_params=settings.CROSS_VALIDATION_PARAMS.parameters_cv,
                                      SearchParamsClass=GridSearchCV,
                                      sp_params=settings.GRID_SEARCH_PARAMS.parameters_kn)

kn_fitted_pipeline = fit_pipeline(pipeline=kn_best_estimator,
                                  X_train=X_train, y_train=y_train,
                                  save_model_dir=settings.MODEL.kn_conf)

#XGBC
xgb_pipeline = make_pipeline(PreprocessorClass=settings.PREPROCESSOR.prep_conf,
                             ModelClass=XGBClassifier())

xgb_best_estimator = search_parameters(pipeline=xgb_pipeline,
                                       X_train=X_train,
                                       y_train=y_train,
                                       CrossValClass=KFold,
                                       cv_params=settings.CROSS_VALIDATION_PARAMS.parameters_cv,
                                       SearchParamsClass=GridSearchCV,
                                       sp_params=settings.GRID_SEARCH_PARAMS.parameters_xgb)

xgb_fitted_pipeline = fit_pipeline(pipeline=xgb_best_estimator,
                                   X_train=X_train, y_train=y_train,
                                   save_model_dir=settings.MODEL.xgb_conf)
