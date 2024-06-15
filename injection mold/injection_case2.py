import numpy as np
import pandas as pd
from preprocess_data import *
import unsupervise_detect
from sklearn.model_selection import train_test_split
from autoML_clf_model import *
from automl_clf_make import *
import argparse

parser = argparse.ArgumentParser(description="Argparse Tutorial")
parser.add_argument("--setting")
args = parser.parse_args()

db = dict(
    server="localhost",
    port="xx",
    user="postgres",
    password="xx",
    database="xx",
    raw_data_table="xx",
)

def main(setting):
    
    df_raw = Db_connect('db info')
    df_base  = prepare_data_classify(df_raw.dbConnect())
    dataset_rule_base = df_base.groupby(["MC_CODE","ITEM_CODE","WORKORDERNO"]).describe()

    df_restruct = pd.DataFrame()
    for idx, df_a in enumerate(dataset_rule_base.index):
        tmp = df_base[(df_base.MC_CODE == df_a[0]) & (df_base.ITEM_CODE == df_a[1]) & (df_base.WORKORDERNO == df_a[2])]
        infer_df = unsupervise_detect(tmp, idx, dataset_rule_base, 'select_name')
        df_restruct = pd.concat([df_restruct, infer_df])
        del infer_df
        
    model_dict = dict()
    model_dict['test_model'] ={}
    model_dict['pv_best'] ={}
    model_data = df_restruct.groupby(["MC_CODE", "ITEM_CODE"]).get_group(('호기명', '제품명'))
    model = ML_Model((model_data,model_dict,False,1)).models
    
    # 평가
    X_train, X_test, y_train, y_test = prepare_data(model_data)
    y_pred = modeling_ml(model, X_train, X_test, y_train, y_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == "__main__":
    print('시작')
    main(args.setting)
