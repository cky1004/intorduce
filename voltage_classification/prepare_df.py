import pandas as pd
import numpy as np
import math
import util as pdfu
import information as pdfi
import warnings
warnings.filterwarnings('ignore')

np.vectorize
def gmake_df(df,sep_para):
    sep_val = sep_para
    mean_value = round(df.VALUE.mean(), 3)
    _,key_group = find_matching_subgroup(mean_value, sep_val[0])
    if sep_val[1] > mean_value or sep_val[2] < mean_value:
        key_group = g_maker(mean_value, sep_val[1:])
    df['train_group'] = key_group
    df['train_group_value'] = mean_value
    return df

def g_maker(x, sep_val):
    min_val, max_val, split_val = sep_val
    num_intervals = (max_val - min_val) / split_val
    interval_length = (max_val - min_val) / num_intervals
    interval = math.floor((x - min_val) // interval_length)
    if num_intervals != 1:
        if interval < 0:
            return 'A'
        elif (min_val <= x <= max_val):
            return chr(ord('A') + interval)
        else:
            interval = math.floor((max_val - min_val) // interval_length)
            return chr(ord('A') + interval)
    else:
        return 'A'

def detail_grm(df,_add_list=None):
    base_list = []  # 정보 파일 또는 외부에서 값 받도록 설정 필요
    if _add_list is None:
        _list_grm = base_list
    else:
        def test(t):
            assert t in df.train_group.unique().tolist(), '포함 그룹'
            return t
        _list_grm = list(filter(test, _add_list))
    return _list_grm

def filter_grm(df, _f_list=None):
    # _list = []  # 정보 파일 또는 외부에서 값 받도록 설정 필요
    if _f_list is None:
        _list_grm = None
    else:
        def test(t):
            if t not in df.train_group_new.unique():
                raise ValueError(f"미포함 그룹: {t}")
        _list_grm = []
        for i in _f_list:
            try:
                test(i)
                _list_grm.append(i)
            except ValueError as e:
                print(e)  # 에러 메시지를 출력하지만 다른 값 처리 계속
    return _list_grm

def preprocess_df(df, _str, _bpr, _add_list=None, _merge_gr=None,_detail_value=3):
    import copy
    g_list = detail_grm(df,_add_list)
    restruct_df_b_last = pd.DataFrame()
    f_li_total = dict()
    
    if _str == 'train':
        f_li = copy.deepcopy(_bpr[0])
        for grm in df.train_group.unique():
            tmp_df3 = df[df.train_group == grm]
            if grm in g_list or g_list is None:
                qcut_values, bins = pd.qcut(tmp_df3.train_group_value, q=_detail_value, duplicates='drop', retbins=True)
                list_grm = [f"{grm}_{i}" for i in range(len(bins) - 1)]
                min_v = f_li[f'{grm}'][f'{grm}'].left
                max_v = f_li[f'{grm}'][f'{grm}'].right
                bins[0] = min_v
                bins[-1] = max_v
                intervals = [pd.Interval(bins[i], bins[i + 1], closed='right') for i in range(len(list_grm))]

                for i in range(1, len(intervals)):
                    if intervals[i].left > intervals[i-1].right:
                        intervals[i] = pd.Interval(intervals[i-1].right, intervals[i].right, closed='right')
                tmp_df3['train_group_new'] = pd.cut(tmp_df3.train_group_value, bins=bins, labels=list_grm, include_lowest=True)
                f_li[grm] = {list_grm[i]: intervals[i] for i in range(len(list_grm))}
                
            else:
                selected_group = grm if (_merge_gr is None) or (ord(str(grm)) < ord(str(_merge_gr))) else _merge_gr
                tmp_df3['train_group_new'] = selected_group
            
            restruct_df_b_last = pd.concat([restruct_df_b_last, tmp_df3])
        
        if _merge_gr is not None:
            merge_g = _merge_gr
            merge_tmp = chr(ord(str(_merge_gr)) - 1)
            merge_tmp_del = chr(ord(str(_merge_gr)) + 1)
            t = [x for x in f_li.keys() if ord(str(x)) >= ord(str(merge_tmp))]
            t2 = [x for x in f_li.keys() if ord(str(x)) >= ord(str(merge_tmp_del))]
            left_val = [list(f_li[x1].values())[-1].right for x1 in t]
            right_val = [list(f_li[x1].values())[-1].right for x1 in t]

            print(f_li)
            if len(f_li.keys()) == 1:
                _inter = f_li['A']['A']
            else:
                _inter = pd.Interval(min(left_val), max(right_val), closed='right')
            f_li[merge_g] = {merge_g: _inter}

            def del_key(_dict, keys):
                for key in keys:
                    if key in _dict:
                        del _dict[key]
                return _dict

            x_tmp = lambda w: del_key(f_li, w)
            f_li = x_tmp(t2)

        f_li_total['interval'] = f_li
        
        
        sep_zone = _bpr.copy()
        new_intervals = {}
        for key, value in f_li_total['interval'].items():
            new_intervals.update(value)
        
        # 가장 작은 영역을 나타내는 서브키값 찾기
        min_key = min(new_intervals, key=lambda x: new_intervals[x].left)

        # 가장 큰 영역을 나타내는 서브키값 찾기
        max_key = max(new_intervals, key=lambda x: new_intervals[x].right)
        
        # 조건에 따라 값 대체
        restruct_df_b_last.loc[restruct_df_b_last['train_group_new'].isna() & (restruct_df_b_last['train_group_value'] < sep_zone[1]), 'train_group_new'] = min_key
        restruct_df_b_last.loc[restruct_df_b_last['train_group_new'].isna() & (restruct_df_b_last['train_group_value'] > sep_zone[2]), 'train_group_new'] = max_key       
            
    elif _str == 'infer':
        f_li = _bpr.copy()
        m_group = df.groupby(['EQUIPMENTID', 'GROUPNUM'])
        if len(m_group) ==1:
            df2 = gmake_df(df, f_li['zone'])
        else:
            df2 =pdfu.applyParallel_init(m_group,gmake_df,f_li['zone'])
        for grm in df2.train_group.unique():
            tmp_df3 = df2[df2.train_group == grm]
            matches = []
            for value in tmp_df3.train_group_value.tolist():
                if f_li['zone'][1] <= value and  value <= f_li['zone'][2]:  
                    group_key, subgroup_key = find_matching_subgroup(value, f_li, step=_str)
                    matches.append(subgroup_key)
                elif f_li['zone'][1] > value:
                    group_k = g_maker(value, f_li['zone'][1:])
                    intervals_list = list(f_li['interval'][group_k].items())
                    matches.append(intervals_list[0][0])
                elif f_li['zone'][2] < value:
                    group_k2 = g_maker(value, f_li['zone'][1:])
                    intervals_list = list(f_li['interval'][group_k2].items())
                    matches.append(intervals_list[-1][0])
            tmp_df3['train_group_new'] = matches
            restruct_df_b_last = pd.concat([restruct_df_b_last, tmp_df3])
        f_li_total = f_li.copy()

    return restruct_df_b_last, f_li_total

def find_matching_subgroup(value, f_li,step='train'):
    if step == 'infer':
        try:
            if 'interval' in f_li:
                for key, subgroups in f_li['interval'].items():
                    for subgroup_key, subgroup_value in subgroups.items():
                        if isinstance(subgroup_value, pd.Interval):
                            if subgroup_value.left <= value <= subgroup_value.right:
                                return key, subgroup_key
            else:
                print("'interval' key not found in f_li dictionary.")
                return None, None
        except KeyError as e:
            print(f"KeyError: {e}")
            return None, None
    
    else:
        for key, subgroups in f_li.items():            
            for subgroup_key, subgroup_value in subgroups.items():
                if isinstance(subgroup_value, pd.Interval):
                    if subgroup_value.left <= value <= subgroup_value.right:
                        return key, subgroup_key
        return None, None
