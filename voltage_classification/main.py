import pandas as pd
import prepare_df as pdfp
import util as pdfu
import model as pdfm
import information as pdfi
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import warnings
import oracledb

warnings.filterwarnings("ignore")

db = dict(password="xxxx", user="xxxx", dsn="xxx.xxx.xx.xx:xxxx/oracle")


def get_dataframe(train_data_from,train_data_to,table_name):

    params = dict(
        target=table_name,
        train_data_from=train_data_from,
        train_data_to=train_data_to,
    )
    connection = oracledb.connect(
        user=db.get("user"), password=db.get("password"), dsn=db.get("dsn")
    )
    cursor = connection.cursor()
    query = f"""
    SELECT *
    FROM {params['target']}
    WHERE EVENTTIME >= :train_data_from AND EVENTTIME <= :train_data_to
    """
    params.pop("target")
    cursor.execute(query, params)
    columns = [col[0] for col in cursor.description]
    cursor.rowfactory = lambda *args: dict(zip(columns, args))
    result = cursor.fetchall()
    return pd.DataFrame.from_records(result)


def main():

    ################ Data Loding #########################
    df_base = get_dataframe('train_data_from','train_data_to','table_name')

    ################ Data common Preprocess #########################
    m_group = df_base.groupby(['EQUIPMENTID', 'GROUPNUM'])
    para_sep_zone = pdfi.seperate_zone()
    result = pdfu.applyParallel_init(m_group,pdfp.gmake_df,para_sep_zone)
    print(result.train_group.value_counts())
    ################ Data Preprocess_train #########################
    focus_group = tuple([key for (key, val) in dict(result.train_group.value_counts()).items() if val > 5000])
    merge_group = None
    focus_group_split = None
    abnomal_grm_list = None
    abnomal_grm_limit_value = None
    _beta_value = None
    _iter_num = 100
    ################ Data Preprocess_train #########################
    df ,_base_model = pdfp.preprocess_df(result,'train',
                                        _bpr=list(para_sep_zone),
                                        _add_list=focus_group,
                                        _merge_gr=merge_group,
                                        _detail_value=focus_group_split)
    print(_base_model)
    ################ setting common #########################

    bo_setting = pdfi.bo_param(l_limit_b=0,
                                u_limit_b=30,
                                iteration=_iter_num, 
                                _li_b= abnomal_grm_limit_value, 
                                _beta_b=_beta_value)
    _f_list = pdfp.filter_grm(df, _f_list=(abnomal_grm_list))
    
    ################ Training ########################################
    clf = pdfm.PDFclassifier(bo_setting, _f_list, para_sep_zone)
    
    model_pdf = clf.train(df,_base_model)
    print(model_pdf)

    re_df = clf.inference(df, model_pdf, cpu_multi_use=True)
    results_pdf = pdfu.get_clf_eval("pdf", re_df["pred"], re_df["pdf"])
    results_pdf_df =pd.DataFrame(data=[results_pdf],columns=['pdf_Model',
                                                'accuracy_score',
                                                'precision_score',
                                                'recall_score',
                                                'f1_score',
                                                'AUC_score',
                                                'tp','fn','fp','tn'])
    return_df = pd.concat([return_df,results_pdf_df])
    
    with mlflow.start_run():
        # Logging metrics
        mlflow.log_metric('pdf_acc', return_df['accuracy_score'])
        mlflow.log_metric('pdf_precision', return_df['precision_score'])
        mlflow.log_metric('pdf_recall', return_df['recall_score']) 
        mlflow.log_metric('pdf_F1_sc', return_df['f1_score']) 
        # Saving the model
        mlflow.sklearn.log_model(sk_model=model_pdf, 
                                 artifact_path="Shortage_model") 

if __name__ == "__main__":
    print('시작')
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")  
    mlflow.set_experiment("MLflow Shortage1")
    main()