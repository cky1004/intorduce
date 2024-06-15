import pandas as pd
import numpy as np
from autoML_custom import *
from joblib import dump
import json
import warnings
import gc
warnings.filterwarnings("ignore")

db = dict(password="xxxx", user="xxx", dsn="xxx.xxx.xx.xx:xxxx/oracle")


def main():
    
    ################ Data Loding #########################
    df = 'get_dataframe(db)'
    
    print("Size of Dataset: ", len(df))
    ################ Data common Preprocess #########################
    re_df = df
    
    sv3 = ['SV1_PR', 'SV2_PR', 'SV3_PR', 'SV4_PR',
        'SV5_PR', 'SV6_PR', 'SV7_PR', 'SV8_PR',
        'SV9_PR', 'SV10_PR', 'SV11_PR', 'SV12_PR',
        'SV13_PR', 'SV14_PR', 'SV15_PR', 'SV16_PR',
        'SV17_PR', 'SV18_PR', 'SV19_PR', 'SV20_PR',
        'SV21_PR','SV22_PR','SV23_PR','SV24_PR',
        'SV25_PR','SV26_PR']
    
    sv4 = ['SV1_delta', 'SV2_delta', 'SV3_delta', 'SV4_delta',
        'SV5_delta', 'SV6_delta', 'SV7_delta', 'SV8_delta',
        'SV9_delta', 'SV10_delta', 'SV11_delta', 'SV12_delta',
        'SV13_delta', 'SV14_delta', 'SV15_delta', 'SV16_delta',
        'SV17_delta', 'SV18_delta', 'SV19_delta', 'SV20_delta',
        'SV21_delta','SV22_delta','SV23_delta','SV24_delta',
        'SV25_delta','SV26_delta']
    
    pv3 = ['PV1_PR', 'PV2_PR', 'PV3_PR', 'PV4_PR', 'PV5_PR', 
           'PV6_PR', 'PV7_PR', 'PV8_PR', 'PV9_PR', 'PV10_PR']
    
    pv4 = ['PV1_AF', 'PV2_AF', 'PV3_AF', 'PV4_AF','PV5_AF', 
           'PV6_AF', 'PV7_AF', 'PV8_AF', 'PV9_AF', 'PV10_AF']

    with open('information.json', 'r') as file:
        json_content = file.read()
    data = json.loads(json_content)
    
    # common set value
    selected_models = data["select_models"]
    pn = data["data_filter"]['pn']
    coating_type = data["data_filter"]['coating_type']
    coating_mode = data["data_filter"]['coating_mode']
    
    model_dict = data.copy()
    model_dict['test_model'] ={}
    model_dict['pv_best'] ={}
    
    ################ train #########################

    results_train = pd.DataFrame()
    result_test = pd.DataFrame()

    
    for (col1, col2) in zip(pv3, pv4):
        sel_col = [col1] + sv3 + sv4 + [col2]
        c_model = Coating_Model((False, 1, selected_models))
        df_train, df_test, model = c_model.compare_model(re_df, sel_col, model_dict.copy()) 
        model_dict = model
        results_train = pd.concat([results_train, df_train])
        result_test = pd.concat([result_test, df_test])
        del sel_col ,df_train, df_test, model ,c_model
        model_dict['test_model'] ={}
        gc.collect()
         
    results_train_f = results_train.loc[results_train['RANK']==1]
    result_test_f = result_test.loc[result_test['RANK']==1]

    prefix1 = "train"
    train_metric = {
        f"{prefix1}_r2_score": results_train_f['r2'].max(),
        f"{prefix1}_mse": results_train_f['mse'].max(),
        f"{prefix1}_mae": results_train_f['mae'].max(),
    }
    prefix2 = "val"
    val_metric = {
        f"{prefix2}_r2_score": result_test_f['r2'].max(),
        f"{prefix2}_mse": result_test_f['mse'].max(),
        f"{prefix2}_mae": result_test_f['mae'].max(),
    }
    
    metric = {**train_metric, **val_metric}
    print(metric)

    dump(model_dict, f"Product_{pn}_coating_mode_{coating_type}_coating_step_{coating_mode}.joblib")
    
    
if __name__ == "__main__":
    print('시작')
    main()
   