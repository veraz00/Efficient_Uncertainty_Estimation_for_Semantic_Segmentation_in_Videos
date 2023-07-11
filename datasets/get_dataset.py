from .camvid_datasets import Camvid

def dataset_dict(name):
    return {'camvid': Camvid}[name]