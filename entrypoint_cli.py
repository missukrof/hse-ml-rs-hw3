import argparse
from conf.conf import logging
from model.constructor.model_constructor import predict_values

parser = argparse.ArgumentParser(description='Pipeline predictions')

parser.add_argument('path_to_x_test', type=str, help='The path to open the test X-features for the pipeline prediction')
parser.add_argument('path_to_model', type=str, help='The path to model (pipeline) for opening')
parser.add_argument('path_to_y_preds', type=str, help='The path to save the values predicted by the pipeline')
args = parser.parse_args()
print(args.path_to_x_test)
responce = predict_values(path_to_x_test=args.path_to_x_test,
                          path_to_model=args.path_to_model,
                          path_to_y_preds=args.path_to_y_preds)

logging.info(f'Prediction (model: {type(args.path_to_model[1]).__name__}) is {responce}')