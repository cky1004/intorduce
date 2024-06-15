# 히스토그램
import numpy as np
from sklearn.metrics import fbeta_score , f1_score ,recall_score
import pandas as pd
from skopt import gp_minimize
import util as pdfu
import prepare_df as pdfp
from multiprocessing import Process, Manager, cpu_count
from functools import partial
import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class PDFclassifier :
    def __init__(self, bo_set, _f_list ,sep_zone):
        self.boset = bo_set
        self.flist = _f_list
        self.sep_zone = sep_zone

    def train(self,df,_base_model):
        # Get the number of available CPU cores
        num_cores = cpu_count()
        start = time.time()
        results = {}
        total_dict = _base_model
        key_group = df.groupby(['train_group_new'])
        # 각 키에 대한 서브 키를 리스트로 담을 빈 리스트 생성
        sub_keys_list = []

        default_value = [12.0, 12.0]     # 학습시 없는 그룹 고정 시그마 값
        
        results_dict = Manager().dict()
        processes = []
        
        # CPU 병렬 처리
        for id, (name,group) in tqdm.tqdm(enumerate(key_group),total=len(key_group),desc='processing_groups'):
            # Distribute the tasks across available CPU cores
            core_id = id % num_cores
            grm = name[0]
            p = Process(target=self.train_parallel, args=(group, grm, results ), kwargs={'results_dict': results_dict})
            processes.append((p,core_id))
            p.start()
            
        for p , core_id in processes:
            p.join()
            
        final_results = dict(results_dict)


        # 딕셔너리에서 각 키의 서브 키를 추출하여 리스트에 추가
        for key, value in total_dict['interval'].items():
            sub_keys = list(value.keys())
            sub_keys_list.extend(sub_keys)

        # 딕셔너리에 키가 없으면 default_value를 할당
        for items_ in sub_keys_list:
            if items_  not in final_results.keys():
                final_results[items_] = default_value
        
        # Use final_results as needed
        print("Final Results:", final_results)
        print(f"End Time : {time.time() - start}s")
        total_dict['ul'] = final_results
        total_dict['boset'] = self.boset
        total_dict['liv'] = self.boset[2]
        total_dict['beta'] = self.boset[3]
        total_dict['fg'] = self.flist
        total_dict["zone"] = self.sep_zone
        total_dict["sel_col"] = ['VALUE','Std_v','Regression','train_group_value','diff']
        return total_dict
    
    def train_parallel(self,df_run, grm, results , results_dict=None):
        params = [df_run, grm, results]
        train_partial = partial(self.training_BO, results_dict=results_dict)
        train_partial(params)

    def training_BO(self,params,results_dict=None):

        df_tmp  =  params[0]
        tgm     =  params[1]
        result  =  params[2]
        print(tgm)
        tmp = gp_minimize(
                        lambda para : 
                        self.objective(df_tmp,para,self.flist,self.boset[2],self.boset[3],tgm) 
                        , self.boset[0]
                        , n_calls=self.boset[1]
                        , acq_func='EI' 
                        , n_random_starts=20
                        , random_state=30
                        )
        result[tgm] = tmp.x
        print('{} 결과 : '.format(tgm),result[tgm])

        # Store the result in the shared dictionary
        if results_dict is not None:
            results_dict[tgm] = result[tgm]

    def objective(self,df,param,_f_li, _lm_v,_beta,grm):
        import tqdm
        groups = df.groupby(['EQUIPMENTID','GROUPNUM'])
        restruct_df = pd.DataFrame()
        for id, (name,group) in tqdm.tqdm(enumerate(groups),total=len(groups),desc='processing_groups'):
            re_df = classifier(group,param,'train',_f_li,_lm_v)
            restruct_df = pd.concat([restruct_df,re_df])
        f1 = f1_score(restruct_df['pred'], restruct_df['pdf'])
        print(f'{grm}_group_F1 score : ' ,f1)
        return -f1  # maximize the F1 score

    def inference(self,df,_model,cpu_multi_use = False):
        import tqdm
        _f_li= _model['fg']
        _lm_v= _model['liv']
        groups = df.groupby(['EQUIPMENTID','GROUPNUM'])  
        if cpu_multi_use == True:
            def infer_pdf(df_base):
                re_df = classifier(df_base,_model['ul'],'infer',_f_li,_lm_v)
                return re_df
            restruct_df = pdfu.applyParallel(groups,infer_pdf)
        else:
            restruct_df = pd.DataFrame()
            for id, (name,group) in tqdm.tqdm(enumerate(groups),total=len(groups),desc='processing_groups'):
                re_df = classifier(group,_model['ul'],'infer',_f_li,_lm_v)
                restruct_df = pd.concat([restruct_df,re_df])
        return restruct_df

np.vectorize
def classifier(df,param:dict,_str,_f_li,_limit_val):
    group = df.copy()
    (d1,p1) , (d2,p2) , (d3,p3) = reg_make(group)
    std_g = group.VALUE.describe().loc['std']
    if np.isnan(std_g) or std_g==0 or std_g < 0.01 : # 0516 변경
        std_g = 0.01
    
    if _str == 'infer':
        # print(group.train_group_new.unique()[0])
        if pd.isna(group.train_group_new.unique()[0]):
            print(group)
        param2 = param[group.train_group_new.unique()[0]]
        if type(param2) == str:
            param2 =list(map(float,param2.replace('[','').replace(']', '').split(',')))
            
        arr_result  = process_group_data(group, param2, std_g, p3,_f_li, _limit_val)

        x3      = [d3[0] for i in range(len(group.ROWNUM))]    
        bias    = [d3[3] for i in range(len(group.ROWNUM))]
        std_v   = [std_g for i in range(len(group.ROWNUM))]
        up_v    = [param2[0] for i in range(len(group.ROWNUM))]
        down_v  = [param2[1] for i in range(len(group.ROWNUM))]
        regre_3 = [p3(ir) for ir in group.ROWNUM.values.tolist()]
        
        group['X3_coeffient']= x3
        group['Bias_coeffient']= bias
        group['Regression']= regre_3
        group['diff']  = group['Regression']  - group['VALUE']
        group['Std_v']= std_v
        group['Up_v']= up_v
        group['Down_v']= down_v
        group['pdf']= arr_result

    elif _str == 'train':
        param2 = param
        arr_result  = process_group_data(group, param2, std_g, p3, _f_li,_limit_val)
        group['pdf'] = arr_result
        
    return group

def process_group_data(group, param2, std_g, p, _f_li, _limit_val):
    result_arr, result_regre = [], []
    lower_value = std_g * param2[1]
    upper_value = std_g * param2[0]
    if _f_li==None:
        _f_li = ()
    
    if group.train_group_new.unique()[0] in list(_f_li):
        lower_threshold = min(lower_value, _limit_val)
        upper_threshold = min(upper_value, _limit_val)
    else:
        lower_threshold = lower_value
        upper_threshold = upper_value
    
    for rownum, value in zip(group.ROWNUM, group.VALUE):
        regression_val = p(rownum)
        condition = ((regression_val - lower_threshold > value) or (value > regression_val + upper_threshold))
        result_arr.append(np.where(condition, 1, 0).tolist())

    return result_arr

def reg_make(group):
    d1 = np.polyfit(group.ROWNUM.values.tolist(),group.VALUE.values.tolist(),1)
    d2 = np.polyfit(group.ROWNUM.values.tolist(),group.VALUE.values.tolist(),2)
    d3 = np.polyfit(group.ROWNUM.values.tolist(),group.VALUE.values.tolist(),3)
    p1 = np.poly1d(d1)
    p2 = np.poly1d(d2)
    p3 = np.poly1d(d3)
    return (d1,p1) , (d2,p2) , (d3,p3)

