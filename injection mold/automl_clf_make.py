import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from glob import glob
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

def prepare_data(df):
    # 전처리 함수
    df_x = df[[x for x in list(df.columns) if "S_" in x or "C_" in x][1:]]
    df_y = df['y_pred'].replace(1, 0).replace(-1, 1)
    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=30,shuffle=True)

    return (X_train, Y_train), (X_test,Y_test)

def blenders_(model, weight_,para_b):
  weight_tune = []
  models_tune = []
  model_name  = []
  sel_para ={}
  for i ,(m,x) in enumerate(zip(model,weight_)):
    if x >  0:
        weight_tune.append(x)
        models_tune.append(m)
        model_name.append(m[0])
  print(model_name)
  for idx , m2 in enumerate([*para_b.keys()]):
    if m2.split('_')[0] in model_name:
      sel_para[m2] = para_b[m2]
              
  return weight_tune, models_tune, sel_para ,model_name

def para_sel(para_b,model_name):
  sel_para ={}
  for idx , m2 in enumerate([*para_b.keys()]):
    if m2.split('_')[0] in model_name:
        print(m2.split('__')[1])
        sel_para[m2] = para_b[m2]
  return sel_para

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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_real, y_predict)
    precision = precision_score(y_real, y_predict)
    recall = recall_score(y_real, y_predict)
    F1_sc = f1_score(y_real, y_predict)
    return [name,accuracy,precision,recall,F1_sc]


def tune_model(models_list,param_space_base):
    from skopt import BayesSearchCV
    from sklearn.ensemble import VotingClassifier
    models = dict(models_list)
    tuned_models = {}  # 베이지안 튜닝된 모델을 담을 딕셔너리 생성
    m_n = [x for x in models]

    # 각 모델에 대해 베이지안 튜닝 수행
    for model_name, model in models.items():
        tuned_model = f"{model_name}_tuned"
        select_para = para_sel(param_space_base,model_name)
        voting_clf = VotingClassifier(estimators=[(model_name,model)], voting='soft', weights=[1,] )
        
        globals()[tuned_model] = BayesSearchCV(estimator = voting_clf, 
                                            search_spaces=select_para, 
                                            scoring='f1',
                                            n_iter=30,
                                            cv=3, 
                                            n_jobs=-1
                                            )
        
        tuned_models[tuned_model] = globals()[tuned_model]
    tuned_models = list(tuned_models.items())
    return tuned_models


def generate_lists(model_s,select_num):
    import itertools
    numbers = [0] * (len(model_s)-select_num) + [1] * select_num
    lists = set(itertools.permutations(numbers, len(model_s)))
    return lists

def mix_model_select(model_s,select_num):
    lists = generate_lists(model_s,select_num)
    weights = []
    for lst in lists:
        weights.append(list(lst))
        
    return weights

def vote_model_mix(model_s,select_num,para):
    #import information as info
    from sklearn.ensemble import VotingClassifier
    weight_select = mix_model_select(model_s,select_num)
    vote_models ={}
    
    for weight in weight_select:
        weight_tune1 , models_tune_b , sel_para_b , m_name1 = blenders_(model_s,weight,para)
        vote_ = ','.join(m_name1)
        globals()[vote_] = VotingClassifier(estimators=models_tune_b, voting='soft', weights=weight_tune1 )
        
        vote_models[vote_] = globals()[vote_]
    vote_models = list(vote_models.items())
    return vote_models

def model_compare(model_data,model_base,model_dict):
    from sklearn.pipeline import Pipeline
    import tqdm
    import gc

    (x_train, y_train), (x_test,y_test) = prepare_data(model_data)

    results_train , results_test =[] , [] 
    _result_col = ['model','acc','precision','recall','f1']
    for idx , (name ,test_model) in enumerate(tqdm.tqdm(model_base[:])):
        print(idx,name)
        # 분류기 학습 및 튜닝
        pipe = [(name ,test_model)]
        pipeline = Pipeline(pipe, verbose=True)
        pipeline.fit(x_train, y_train)
        
        # 최적의 모델로 예측
        y_pred_train = pipeline.predict(x_train)
        y_pred_test = pipeline.predict(x_test)

        # 성능 평가
        result_train = get_clf_eval(name,y_train, y_pred_train)
        result_test = get_clf_eval(name,y_test, y_pred_test)
        
        results_train.append(result_train)
        results_test.append(result_test)
        # 메모리 정리
        del pipeline, y_pred_train, y_pred_test, result_train, result_test
        gc.collect()

    result_train_pd = pd.DataFrame(results_train, 
                                   columns= _result_col)
    result_test_pd = pd.DataFrame(results_test, 
                                  columns= _result_col)
    
    result_train_pd.sort_values('f1',ascending=False,inplace=True)
    result_test_pd.sort_values('f1',ascending=False,inplace=True)
    result_test_pd['RANK'] = [x for x in range(len(result_test_pd))]
    f1_1st = result_test_pd['model'].loc[result_test_pd['RANK']==0].values[0]
    f1_val = result_test_pd['f1'].loc[result_test_pd['RANK']==0].values[0]
    final_model =[]
    m_name= f1_1st
    for x in model_base:
        if m_name == x[0]:
            final_model.append((x[0],x[1]))
    print('최종 모델 : {} _ f1: {}'.format(m_name,f1_val))
    print(final_model[0])
    pipeline_final = Pipeline(final_model, verbose=True)
    pipeline_final.fit(x_train, y_train) 
    model_dict["ml_model"] = pipeline_final
    return model_dict