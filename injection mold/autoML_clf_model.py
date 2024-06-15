from pyexpat import model
import numpy as np
import pandas as pd
from urllib.parse import quote
import pandas as pd
import argparse
from datetime import datetime
from automl_clf_make import *
import ml_clf_set as mlset
from sklearn.ensemble import VotingClassifier

class ML_Model:
    def __init__(self, params):
        self.optimize_use = params[2]
        self.blen_model = params[3]
        self.models = self.build_model_clf(params[0],params[1])
    
    def build_model_clf(self,model_data,result : dict):
        estim_base= mlset.model_for_classifier()
        param_space_base = mlset._hyperparameter_total()
        tuned_m = tune_model(estim_base,param_space_base)
        select_col = result["sel_col"]
        total_estimators = self.build_total_estimators(estim_base, tuned_m, param_space_base)
        clf = model_compare(model_data, select_col, total_estimators,result)

        return clf    

    def build_total_estimators(self, base_estimators, tuned_estimators, param_space):
        if self.optimize_use == False:
            if self.blen_model in [0,1]:
                return base_estimators
            else: 
                return base_estimators + vote_model_mix(base_estimators, self.blen_model , param_space)
        elif self.optimize_use == True:
            if self.blen_model in [0,1]:
                return base_estimators + tuned_estimators
            else: 
                return base_estimators + tuned_estimators + vote_model_mix(tuned_estimators, self.blen_model , param_space)


        