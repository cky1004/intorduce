import os
import pickle
import gzip
import csv
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

def pdfm_save(path, result_save):
    try:
        os.makedirs(path, exist_ok=True)
        date_str = datetime.today().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(path, "{}.csv".format(date_str))
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            for k, v in result_save.items():
                writer.writerow([k, v])

        pickle_path = path + "{}".format(date_str) + ".pickle"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(result_save, f)

        print("File saved successfully at:", file_path)
        return pickle_path
    except Exception as e:
        print("Error occurred while saving the file:", str(e))

def pdfm_get(path):
    files = os.listdir(path)
    file_list = [file for file in files if file.endswith(".pickle")]

    print(file_list)
    model_num = 0
    result_get = {}

    with gzip.open("{}".format(os.path.join(path,file_list[model_num])),'rb') as f:
        result_get = pickle.load(f)

    print(result_get)
    return result_get


def cm_get_val(y_real, y_predict):
    from sklearn.metrics import confusion_matrix
    cms = confusion_matrix(y_real, y_predict)
    tp = cms[0][0]
    fn = cms[0][1]
    fp = cms[1][0]
    tn = cms[1][1]
    return tp,fn,fp,tn


def get_clf_eval(name,y_real, y_predict):
    # 정확도, 정밀도, 재현율, f1_score, auc
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y_real, y_predict)
    precision = precision_score(y_real, y_predict)
    recall = recall_score(y_real, y_predict)
    F1_sc = f1_score(y_real, y_predict)
    AUC_sc = roc_auc_score(y_real, y_predict)
    tp,fn,fp,tn = cm_get_val(y_real, y_predict)
    return [name,accuracy,precision,recall,F1_sc,AUC_sc,tp,fn,fp,tn]


def applyParallel(dfGrouped, func, n_jobs_ = cpu_count()):
    import pandas as pd
    from joblib import Parallel, delayed
    import tqdm
    re_df = Parallel(n_jobs=n_jobs_)(delayed(func)(group) for name, group in tqdm.tqdm(dfGrouped, total=len(dfGrouped)))  
    df = pd.concat(re_df)
    return df


def applyParallel_init(dfGrouped, func, args, n_jobs_=cpu_count()):
    import pandas as pd
    from joblib import Parallel, delayed
    import tqdm
    
    def apply_func(group, args):
        return func(group,args)
    
    re_df = Parallel(n_jobs=n_jobs_)(delayed(apply_func)(group, args) for name, group in tqdm.tqdm(dfGrouped, total=len(dfGrouped)))  
    df = pd.concat(re_df)
    return df