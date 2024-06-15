from automl_reg_make import *
from ml_set_reg import *

class Coating_Model:
    def __init__(self, params):
        self.optimize_use = params[0]
        self.blen_model = params[1]
        self.sel_model = params[2]
        self.estimator = self.initialize_estimator()
    
    def initialize_estimator(self):
        estim_base = model_for_regressor(self.sel_model)
        param_space_base = hyperparameter_total()
        tuned_m = tune_model(estim_base, param_space_base)
        total_estimators = self.build_total_estimators(estim_base, tuned_m, param_space_base)
        return total_estimators
    
    def build_total_estimators(self, base_estimators, tuned_estimators, param_space):
        if not self.optimize_use or self.blen_model < 2:
            return base_estimators if self.blen_model in [0, 1] else base_estimators + vote_model_mix(base_estimators, self.blen_model, param_space)
        else:
            return base_estimators + tuned_estimators + vote_model_mix(tuned_estimators, self.blen_model, param_space)
    
    def compare_model(self, model_data, sv_cols, result):
        train_pd, test_pd, clf = model_compare_make(model_data, sv_cols, self.estimator, result)
        return train_pd, test_pd, clf