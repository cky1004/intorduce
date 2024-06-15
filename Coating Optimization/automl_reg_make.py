import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from glob import glob
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

def prepare_data(df,SN_):
    # 전처리 함수
    target = SN_[-1] # PV_AF(1) 
    SN1= SN_[:-1] # PV_PR(1) + SV_PR(26) + SV_delta(26)

    df_x = df[SN1]
    df_y = df[target]
    # Data Split
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=30,shuffle=True)

    return (X_train, y_train), (X_test,y_test)

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

def my_custon_func(y_real, y_predict, th):
    correct_predictions = 0
    total_predictions = len(y_real)
    
    for yr, yd in zip(y_real, y_predict):
        error = abs(yr - yd) / abs(yr)
        if 0 < error <= th:
            correct_predictions += 1
    return correct_predictions / total_predictions

def get_reg_eval(name, y_real, y_predict):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Calculate evaluation metrics
    mape = 100 * (abs(y_real - y_predict) / abs(y_real)).mean()  # Mean Absolute Percentage Error
    mse = mean_squared_error(y_real, y_predict)  # Mean Squared Error
    mae = mean_absolute_error(y_real, y_predict)  # Mean Absolute Error
    r2 = r2_score(y_real, y_predict)  # R-squared
    one = my_custon_func(y_real, y_predict,0.01)
    two =  my_custon_func(y_real, y_predict,0.02)
    three =  my_custon_func(y_real, y_predict,0.03)
    five =  my_custon_func(y_real, y_predict,0.05)
    ten =  my_custon_func(y_real, y_predict,0.1)
    return [name, mse, mae, r2, mape, one, two, three, five, ten]


def tune_model(models_list, param_space_base):
    from skopt import BayesSearchCV
    from sklearn.ensemble import VotingRegressor
    
    models = dict(models_list)
    tuned_models = {}  # 튜닝된 모델을 저장할 딕셔너리
    
    # 각 모델에 대해 베이지안 튜닝 수행
    for model_name, model in models.items():
        tuned_model_name = f"{model_name}_tuned"
        select_para = para_sel(param_space_base, model_name)
        
        # 모델별로 정의된 하이퍼파라미터 검색 공간을 가져옵니다.
        if not select_para:  # 검색 공간이 비어있다면 해당 모델에 대한 기본 설정을 사용
            tuned_models[tuned_model_name] = model
            continue
        
        # 현재 모델을 사용하여 VotingRegressor 생성
        voting_reg = VotingRegressor(estimators=[(model_name, model)], weights=[1,])
        tuned_model = BayesSearchCV(estimator=voting_reg, 
                                     search_spaces=select_para, 
                                     scoring='neg_mean_absolute_error',
                                     n_iter=30,
                                     cv=3, 
                                     n_jobs=-1
                                     )
        
        tuned_models[tuned_model_name] = tuned_model
    
    # tuned_models 딕셔너리를 (model_name, tuned_model) 튜플의 리스트로 변환
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
    from sklearn.ensemble import  VotingRegressor
    weight_select = mix_model_select(model_s,select_num)
    vote_models ={}
    
    for weight in weight_select:
        weight_tune1 , models_tune_b , _ , m_name1 = blenders_(model_s,weight,para)
        vote_ = ','.join(m_name1)
        globals()[vote_] = VotingRegressor(estimators=models_tune_b, weights=weight_tune1 )
        vote_models[vote_] = globals()[vote_]
    vote_models = list(vote_models.items())
    return vote_models

def model_compare_make(model_data,select_col,model_list,model_dict):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
    from sklearn.pipeline import Pipeline
    import tqdm
    
    pipe_base = []
    col1= select_col[0].split('_PR')
    
    (x_train, y_train), (x_test,y_test) = prepare_data(model_data,select_col)
    
    results_train , results_test =[] , [] 
    _result_col = ['model','mse','mae','r2','mape','1%오차','2%오차','3%오차','5%오차','10%오차']
    
    for idx , (name ,test_model) in enumerate(tqdm.tqdm(model_list[:])):
        print(idx,name)
        # 분류기 학습 및 튜닝
        print(test_model)
        pipe = pipe_base + [(name ,test_model)]
        pipeline = Pipeline(pipe, verbose=True)
        pipeline.fit(x_train, y_train)
        model_dict['test_model'][col1][name] = pipeline
        
        # 최적의 모델로 예측
        y_pred_train = pipeline.predict(x_train)
        y_pred_test  = pipeline.predict(x_test)

        # 성능 평가
        result_train = get_reg_eval(name,y_train, y_pred_train)
        result_test = get_reg_eval(name,y_test, y_pred_test)
        
        results_train.append(result_train)
        results_test.append(result_test)

    result_train_pd = pd.DataFrame(results_train, columns= _result_col)
    result_test_pd = pd.DataFrame(results_test, columns= _result_col)
    
    result_train_pd.sort_values('mape',ascending=True,inplace=True)
    result_test_pd.sort_values('mape',ascending=True,inplace=True)
    result_train_pd['RANK'] = [x for x in range(len(result_train_pd))]
    result_test_pd['RANK'] = [x for x in range(len(result_test_pd))]
    
    _1st_mape = result_test_pd['model'].loc[result_test_pd['RANK']==0].values[0]
    model_dict['pv_best'][col1] = _1st_mape

    return result_train_pd,result_test_pd,model_dict