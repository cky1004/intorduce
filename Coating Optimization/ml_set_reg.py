def model_for_regressor(m_name=None):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    
    # 회귀 모델로 사용할 모델들을 선택 또는 추가.
    lin_reg = LinearRegression()
    ridge_reg = Ridge(alpha=0.1)
    lasso_reg = Lasso(alpha=0.1)
    et = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
    dec_tree_reg = DecisionTreeRegressor()
    rf_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    gb_reg = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)
    xgb_reg = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5, n_jobs=-1)
    lgbm_reg = LGBMRegressor(learning_rate=0.1, n_estimators=100, max_depth=5, n_jobs=-1)
    catboost_reg = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5)
    svr_reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    estim_base = [
        ('lin', lin_reg),
        ('ridge', ridge_reg),
        ('lasso', lasso_reg),
        ('et', et),
        ('dt', dec_tree_reg),
        ('rf', rf_reg),
        ('gbc', gb_reg),
        ('xgb', xgb_reg),
        ('lgbm', lgbm_reg),
        ('catboost', catboost_reg),
        ('svr', svr_reg),
    ]
    
    reorganized_base = [estim_base[i] for i in select_model(estim_base,m_name)]
    return reorganized_base

def select_model(estima , m_names):
    r_value =[]
    if m_names:
        for m_name in m_names:
            tmp = [i[0] for i in estima].index(m_name)
            r_value.append(tmp)
    else:
        r_value = [i for i in range(len(estima))]
    return r_value

def hyperparameter_total():
    param_space_base = {
        'gbc__n_estimators': (50, 200),
        'gbc__learning_rate': (0.01, 0.3, 'log-uniform'),
        'gbc__max_depth': (1, 10),
        'gbc__min_samples_split': (2, 10),  
        
        'dt__max_depth': (1, 20),
        'dt__min_samples_split': (2, 10),  
        
        'xgb__learning_rate': (0.01, 0.3, 'log-uniform'),
        'xgb__max_depth': (1, 20),
        'xgb__n_estimators': (50, 200),
        'xgb__gamma': (0, 5.0),  
        
        'lgbm__learning_rate': (0.01, 0.3, 'log-uniform'),
        'lgbm__max_depth': (1, 20),
        'lgbm__n_estimators': (50, 200),
        'lgbm__num_leaves': (10, 200),
        'lgbm__min_child_samples': (2, 20),  
        
        'catboost__learning_rate': (0.01, 0.3, 'log-uniform'),
        'catboost__max_depth': (1, 15),
        'catboost__n_estimators': (50, 200),
        'catboost__min_data_in_leaf': (1, 20),  
        
        'rf__n_estimators': (50, 200),
        'rf__max_depth': (1, 15),
        'rf__min_samples_split': (2, 10),  
        
        'et__n_estimators': (50, 200),
        'et__max_depth': (1, 15),
        'et__min_samples_split': (2, 10), 
        
        'svr__C': (1.0, 10),
        'svr__gamma': (0.01, 1.0, 'log-uniform'),
        'svr__epsilon': (0.1, 0.3),  
    }
    
    return param_space_base