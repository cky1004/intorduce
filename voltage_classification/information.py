
def bo_param(l_limit_b=None, u_limit_b=None, iteration=None,_li_b = None,_beta_b = None):
    
    from skopt.space import Real
    default_l_limit, default_u_limit, default_iteration ,default_limit_v ,default_beta_v  = 1, 50, 20 , 5.0 ,1.5

    if l_limit_b is None:
        l_limit = default_l_limit
    else:
        l_limit = l_limit_b

    if u_limit_b is None:
        u_limit = default_u_limit
    else:
        u_limit = u_limit_b

    if iteration is None:
        n_calls = default_iteration
    else:
        n_calls = iteration

    if _li_b is None:
        _li_v = default_limit_v
    else:
        _li_v = _li_b
        
    if _beta_b is None:
        _beta_v = default_beta_v
    else:
        _beta_v = _beta_b        
    
    
    space = [
        Real(l_limit, u_limit, name='up_multi'),
        Real(l_limit, u_limit, name='down_multi'),
    ]

    return (space, n_calls, _li_v , _beta_v)


def seperate_zone(min_val=None, max_val=None, split_Val=None):
    import pandas as pd
    default_min_val, default_max_val, default_split_Val =  0.5,1.5,0.1

    if min_val is None:
        min_val = default_min_val

    if max_val is None:
        max_val = default_max_val

    if split_Val is None:
        split_Val = default_split_Val

    zone_intervals = {}
    current_val = min_val
    while current_val < max_val:
        interval_label = chr(ord('A') + len(zone_intervals))
        interval = pd.Interval(current_val, current_val + split_Val, closed='right')
        zone_intervals[interval_label] =dict(zip(interval_label,[interval]))
        current_val += split_Val
    rounded_intervals = {group: {subgroup: round_interval(interval) for subgroup, interval in subgroups.items()} for group, subgroups in zone_intervals.items()}
    return (rounded_intervals,min_val,max_val,split_Val)

def round_interval(interval, ndigits=3):
    from math import isclose
    import pandas as pd
    left = round(interval.left, ndigits)
    right = round(interval.right, ndigits)
    closed = 'right' if isclose(right, interval.right) else interval.closed
    return pd.Interval(left, right, closed=closed)