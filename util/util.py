import pickle


def save_model(dir: str, model) -> None:
    """
    This function save model results (predicted values) to ...
    """
    pickle.dump(model, open(dir, 'wb'))


def load_model(dir: str) -> None:
    return pickle.load(open(dir, 'rb'))