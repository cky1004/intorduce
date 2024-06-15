def model_for_classifier():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    import xgboost as xgb
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB

    # Voting 분류기에 사용할 개별 모델들
    gbc_model = GradientBoostingClassifier(random_state=2020,subsample=0.1, max_depth=2)
    dec_tree = DecisionTreeClassifier(random_state=2020, max_depth=5, min_samples_split=2, min_samples_leaf=1)
    knn = KNeighborsClassifier(n_neighbors=7)
    xgb_model = xgb.XGBClassifier(random_state=2020, n_estimators=100, learning_rate=0.1, max_depth=5, tree_method='gpu_hist')
    lgbm_model = LGBMClassifier(random_state=2020, n_estimators=100, learning_rate=0.1, max_depth=5, device='gpu')
    catboost_model = CatBoostClassifier(random_state=2020,verbose=False, learning_rate=0.1, max_depth=5, task_type='GPU')
    ada_model = AdaBoostClassifier(random_state=2020, n_estimators=100, learning_rate=0.1)
    rf_model = RandomForestClassifier(random_state=2020, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None,n_jobs=-1)
    et_model = ExtraTreesClassifier(random_state=2020, n_estimators=100, max_depth=5, n_jobs=-1)
    svm_model = SVC(gamma=0.1, kernel="rbf", probability=True)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu')
    gnb_model = GaussianNB()


    estim_base = [
        ('gbc', gbc_model),
        ('dt', dec_tree),
        ('knn', knn),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('catboost', catboost_model),
        ('ada', ada_model),
        ('rf', rf_model),
        ('et', et_model),
        ('svm', svm_model),  # 추가: SVM 모델
        ('mlp', mlp_model),  # 추가: 다층 퍼셉트론 분류기
        ('gnb', gnb_model),  # 추가: 가우시안 나이브 베이즈 모델
    ]
    
    return estim_base

def _hyperparameter_total():
    param_space_base = {
    'gbc__n_estimators': (50, 200),
    'gbc__learning_rate': (0.01, 0.3, 'log-uniform'),
    'gbc__max_depth': (1, 10),

    'dt__max_depth': (1, 20),

    'knn__n_neighbors': (1, 20),

    'xgb__learning_rate': (0.01, 0.3, 'log-uniform'),
    'xgb__max_depth': (1, 20),
    'xgb__n_estimators': (50, 200),

    'lgbm__learning_rate': (0.01, 0.3, 'log-uniform'),
    'lgbm__max_depth': (1, 20),
    'lgbm__n_estimators': (50, 200),
    'lgbm__num_leaves': (10, 200),

    'catboost__learning_rate': (0.01, 0.3, 'log-uniform'),
    'catboost__max_depth': (1, 15),
    'catboost__n_estimators': (50, 200),

    'ada__learning_rate': (0.01, 1.0, 'log-uniform'),
    'ada__n_estimators': (50, 200),

    'rf__n_estimators': (50, 200),
    'rf__max_depth': (1, 15),

    'et__n_estimators': (50, 200),
    'et__max_depth': (1, 15),

    'svm__C': (1.0, 10),  # 추가: SVM의 C 파라미터
    'svm__gamma': (0.01, 1.0, 'log-uniform'),  # 추가: SVM의 gamma 파라미터

    'mlp__hidden_layer_sizes': ((50,), (100,), (50, 50)),  # 추가: 다층 퍼셉트론의 은닉층 크기 파라미터
    'mlp__activation': ('relu', 'tanh'),  # 추가: 다층 퍼셉트론의 활성화 함수 파라미터

    'gnb__var_smoothing': (1e-9, 1e-6, 'log-uniform'),  # 추가: 가우시안 나이브 베이즈의 분산 평활 파라미터
    }
    
    return param_space_base