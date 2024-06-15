import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import json
from joblib import load

with open('information.json', 'r') as file:
    data = json.load(file)

with open('init_data.json', 'r') as file:
    data_init = json.load(file)

# initial values
sv_pr = data_init['sv_pr']
pv_pr = data_init['pv_pr']

# 예시 target
target_pvs = {
    'PV1_PR': 0.5,
    'PV2_PR': 0.3,
    'PV3_PR': 0.7,
    'PV4_PR': 0.6,
    'PV5_PR': 0.8,
    'PV6_PR': 0.2,
    'PV7_PR': 0.4,
    'PV8_PR': 0.9,
    'PV9_PR': 0.1,
    'PV10_PR': 0.3
}

# Load trained model
model_dict = load(f"Product_{data['data_filter']['pn']}_coating_mode_{data['data_filter']['coating_type']}_coating_step_{data['data_filter']['coating_mode']}.joblib")


def predict_pvs(svs, model_dict, sv_pr, pv_col):
    x_test = np.array([pv_pr[pv_col]]+[sv_pr[sv] for sv in sv_pr.keys()] + list(svs)).reshape(1, -1)
    model = model_dict['pv_best'][pv_col]
    y_pred = model.predict(x_test)
    return y_pred

def objective_function(svs, model_dict, sv_pr, target_pvs):
    total_error = 0
    for pv_col in target_pvs.keys():
        predicted_pv = predict_pvs(svs, model_dict, sv_pr, pv_col)[0]
        error = abs(predicted_pv - target_pvs[pv_col])
        total_error += error
    return total_error

def constraint(svs, model_dict, sv_pr, target_pvs):
    constraints = []
    for pv_col in target_pvs.keys():
        predicted_pv = predict_pvs(svs, model_dict, sv_pr, pv_col)[0]
        target_pv = target_pvs[pv_col]
        if predicted_pv < target_pv:
            constraints.append(predicted_pv - target_pv)  # Positive 조정
        else:
            constraints.append(target_pv - predicted_pv)  # Negative 조정
    return constraints


space = [Real(-5.0, 5.0, name=f'sv{i}') for i in range(26)]

result = gp_minimize(
    lambda svs: objective_function(svs, model_dict, sv_pr=sv_pr, target_pvs=target_pvs),
    space,
    n_calls=20,  
    constraints=[lambda svs: constraint(svs, model_dict, sv_pr=sv_pr, target_pvs=target_pvs)],  # Pass the dynamic constraint
)

print("Best SV parameters:", result.x)
print("Best objective value:", result.fun)
