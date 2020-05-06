import pandas as pd
import json

def create_set_copy(dset, paths):
    
    dset_tru = []
    
    for img in dset:
        for path in paths:
            if img in path:
                dset_tru.append(path)
    
    return dset_tru

def create_img_names(data_json, test_csv, train_csv, val_csv):

    with open(data_json) as json_file:
        data_dict = json.load(json_file)

    all_names = list(data_dict.keys())

    test = pd.read_csv(test_csv)
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    img_names_train = create_set_copy(list(train), all_names)
    img_names_val = create_set_copy(list(val), all_names)
    img_names_test = create_set_copy(list(test), all_names)

    return all_names, img_names_test, img_names_train, img_names_val
