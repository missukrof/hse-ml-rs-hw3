import pickle


def save_models(dir: str, model) -> None:
    """
    This function save model to ...
    """
    pickle.dump(model, open(dir, 'wb'))


def load_models(dir: str) -> None:
    """
    This function load model from ...
    """
    return pickle.load(open(dir, 'rb'))