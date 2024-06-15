from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import time
import numpy as np

def model_for_anomaly(X, num, dataset_rule_base, select_name):
    n_samples = X.shape[0]
    rule_base_index = dataset_rule_base.index[num]
    bad_count = X.groupby(["MC_CODE", "ITEM_CODE", "WORKORDERNO"]).get_group((rule_base_index[0], rule_base_index[1], rule_base_index[2]))['BAD_CNT'].mean()
    bad_count = max(bad_count, 0.01)
    outliers_fraction = round(bad_count / n_samples, 5)

    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="poly", gamma=0.1)),
        ("One-Class SVM (SGD)", make_pipeline(Nystroem(gamma=0.1, random_state=42, n_components=150), SGDOneClassSVM(nu=outliers_fraction, shuffle=True, fit_intercept=True, random_state=42, tol=1e-6))),
        ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)),
    ]
    
    x_tmp = X[select_name]
    y_pred = np.ones(x_tmp.shape[0])

    for name, algorithm in anomaly_algorithms:
        if name == "Local Outlier Factor":
            t0 = time.time()
            try:
                if bad_count > 0:
                    y_pred = algorithm.fit_predict(x_tmp)
                else:
                    y_pred = np.ones(x_tmp.shape[0])
            except:
                y_pred = algorithm.fit_predict(x_tmp)
            t1 = time.time()
            print('Execution time: ', t1 - t0)
    
    X['y_pred'] = y_pred
    return X