from model.constructor.model_constructor import predict_values, results_evaluation
from conf.conf import settings
from conf.conf import logging

# KNC
kn_responce = predict_values(path_to_x_test=settings.DATA.x_test,
                             path_to_model=settings.MODEL.kn_conf,
                             path_to_y_preds=settings.DATA.kn_y_preds)

logging.info(f'Prediction (model: {type(settings.MODEL.kn_conf[1]).__name__}) is {kn_responce}')

results_evaluation(path_to_y_preds=settings.DATA.kn_y_preds,
                   path_to_y_test=settings.DATA.y_test,
                   path_to_model=settings.MODEL.kn_conf,
                   model_report_dir=settings.REPORTS.kn_report)

#XGBC
xgb_responce = predict_values(path_to_x_test=settings.DATA.x_test,
                              path_to_model=settings.MODEL.xgb_conf,
                              path_to_y_preds=settings.DATA.xgb_y_preds)

logging.info(f'Prediction is {xgb_responce}')

results_evaluation(path_to_y_preds=settings.DATA.xgb_y_preds,
                   path_to_y_test=settings.DATA.y_test,
                   path_to_model=settings.MODEL.xgb_conf,
                   model_report_dir=settings.REPORTS.xgb_report)