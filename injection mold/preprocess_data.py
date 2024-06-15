import pandas as pd
from urllib.parse import quote
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score
import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Db_connect:
    def __init__(self, db):
        self.db = db

    def dbConnect(self):
        print("<------------DB Connection Start-------------------->")
        def connect_db():
            return create_engine(
                f"postgresql://{self.db['user']}:%s@{self.db['server']}:{self.db['port']}/{self.db['database']}"
                % quote(self.db["password"])
            )
        def etl():
            return pd.read_sql_table(f"{self.db['raw_data_table']}", con=connect_db())
        
        df = etl()
        df.columns = [col.upper() for col in df.columns]
        print("<------------DB Connection End-------------------->")
        return df

def prepare_data_classify(df):
    base1 = list(df.columns)
    base_c = [x for x in base1 if "S_" not in x]
    df_base = df[base_c].drop(['OK_NG_FLAG', 'SHOT_NUM'], axis=1)
    df_base['DATA_ID'] = df_base['DATA_ID'].astype('str')
    df_base['WORKORDERNO'] = df_base['WORKORDERNO'].astype('str')
    return df_base 

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def scale_df(df_base):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df_base[df_base.columns[6:]])
    df_base_scaled = scaler.transform(df_base[df_base.columns[6:]])
    
    df_base_df_scaled = pd.DataFrame(data=df_base_scaled, columns=df_base.columns[6:])
    df_base_df_scaled['idx'] = range(len(df_base_df_scaled))
    
    df_base_idx_df = df_base[df_base.columns[:6]]
    df_base_idx_df['idx'] = range(len(df_base_idx_df))
    
    df_base_total = pd.merge(df_base_idx_df, df_base_df_scaled, how='outer', on='idx')
    df_base_total['class'] = np.where(df_base_total['BAD_CNT'] > 0, 1, 0)
    
    return df_base_total

def augment_df(df_base_total):
    dataset_rule_base_groups_scaled = df_base_total.groupby(["MC_CODE", "ITEM_CODE", "WORKORDERNO"])
    seed_num = [123, 234, 456, 789, 2023, 1004, 5050, 7070, 8080, 9090]

    result_df = pd.DataFrame()
    for i, seed in enumerate(seed_num, start=1):
        df_name = f"df_{i}"
        globals()[df_name] = pd.DataFrame()
        for name, group in dataset_rule_base_groups_scaled:
            np.random.seed(seed)
            group = group.sample(frac=1)
            group["WORKORDERNO"] = group["WORKORDERNO"].apply(lambda x: str(x) + f"_{i}")
            globals()[df_name] = pd.concat([globals()[df_name], group])
        result_df = pd.concat([result_df, globals()[df_name]], axis=0)

    result_df = pd.concat([df_base_total, result_df], axis=0)
    return result_df

def reshape_df(result_df):
    result_df_groups = result_df.groupby(["MC_CODE", "ITEM_CODE", "WORKORDERNO"])
    x_df = pd.DataFrame()

    mc_tmp, ic_tmp, wo_tmp = [], [], []
    for group_name in result_df_groups.groups.keys():
        mc_tmp.append(group_name[0])
        ic_tmp.append(group_name[1])
        wo_tmp.append(group_name[2])
    
    x_df['MC_CODE'] = mc_tmp
    x_df['ITEM_CODE'] = ic_tmp
    x_df['WORKORDERNO'] = wo_tmp
    x_df['BAD_CNT'] = result_df_groups['BAD_CNT'].mean().values
    x_df['class'] = result_df_groups['class'].mean().values

    for col in result_df_groups.mean().columns[4:-1]:
        col_values = []
        for name, group in result_df_groups:
            g_c_list = group[col].to_list()
            while len(g_c_list) < 1024:
                g_c_list.append(0)
            col_values.append(g_c_list)
        x_df[col] = col_values

    x_df = x_df[[
        'select_cols'
    ]]
    
    return x_df

def prepare_data(df):
    
    def flatten_column(col):
        return list(col.apply(lambda x: np.array(x)))

    df_x = np.array([flatten_column(df[df.columns[5:]][col]) for col in df.columns[5:]]).T
    df_y = df['class'].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        df_x.reshape((len(df_y), 1024, -1, 1)), df_y, test_size=0.2, random_state=0
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train.reshape((len(Y_train), 1024, -1, 1)), Y_train, test_size=0.2, random_state=0
    )

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

#평가 지표
def metrics_reg(y_test,pred):
    mse = mean_squared_error(y_test,pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test,pred)
    print('mse: {}'.format(mse))
    print('rmse: {}'.format(rmse))
    print('mae: {}'.format(mae))

def metrics_cls(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred, average='micro')
    print(f'Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}')
    print(f'F1-score: {f1:.5f}, AUC: {roc_score:.5f}')

def modeling_ml(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics_cls(y_test, pred)
    return pred

def modeling_cnn(model,x_test,y_test):
    import itertools
    pred = list(itertools.chain(*model.predict(x_test).tolist()))
    for i in range(len(pred)):
        if pred[i] < 0:
            pred[i] = 0
        else:
            pred[i] = round(pred[i])
    metrics_reg(y_test,pred)
    return np.array(pred)