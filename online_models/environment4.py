#encoding: utf-8
import os, gc, time, random
import numpy as np
import pandas as pd
import torch

seed = 223991

gcc='gcc-4.4'
N_PROGRAMS = 10
csmith_home = '/csmith_recorder/csmith_record_2.3.0_t1/'
csmith= csmith_home + '../csmith'
model_path = './xgb.pkl'
library = csmith_home+'runtime'
configFile = 'config'
crashres='./crash'
wrongcoderes='./wrongcode'

check_time = True
extra_feature = False

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

ACTION_BOUND = range(-5,6)
DISTANCE_BASELINE_TYPE = 'ase' ## 'default' 'no' 'ase'


categories_dir={
    'correct':0, # pass 0
    'crash':1, # fail 1
    'wrongcode':1,
    'generation':-1, # invalid
    'compile':-1,
    'execute':-1,
}

single_keys = ["more_struct_union_type_prob",
        "bitfields_creation_prob",
        "bitfields_signed_prob",
        "bitfield_in_normal_struct_prob",
        "scalar_field_in_full_bitfields_struct_prob",
        "exhaustive_bitfield_prob",
        "safe_ops_signed_prob",
        "select_deref_pointer_prob",
        "regular_volatile_prob",
        "regular_const_prob",
        "stricter_const_prob",
        "looser_const_prob",
        "field_volatile_prob",
        "field_const_prob",
        "std_unary_func_prob",
        "shift_by_non_constant_prob",
        "pointer_as_ltype_prob",
        "struct_as_ltype_prob",
        "union_as_ltype_prob",
        "float_as_ltype_prob",
        "new_array_var_prob",
        "access_once_var_prob",
        "inline_function_prob",
        "builtin_function_prob"]
group_keys = ["statement_prob", "assign_unary_ops_prob", "assign_binary_ops_prob", "simple_types_prob", "safe_ops_size_prob"]
group_items_keys = {
        "statement_prob": ["statement_assign_prob", "statement_block_prob",
                           "statement_for_prob", "statement_ifelse_prob",
                           "statement_return_prob", "statement_continue_prob",
                           "statement_break_prob", "statement_goto_prob",
                           "statement_arrayop_prob"],
        "assign_unary_ops_prob": ["unary_plus_prob", "unary_minus_prob",
                                  "unary_not_prob", "unary_bit_not_prob"],
        "assign_binary_ops_prob": ["binary_add_prob", "binary_sub_prob",
                                   "binary_mul_prob", "binary_div_prob",
                                   "binary_mod_prob", "binary_gt_prob",
                                   "binary_lt_prob", "binary_ge_prob",
                                   "binary_le_prob", "binary_eq_prob",
                                   "binary_ne_prob", "binary_and_prob",
                                   "binary_or_prob", "binary_bit_xor_prob",
                                   "binary_bit_and_prob", "binary_bit_or_prob",
                                   "binary_bit_rshift_prob", "binary_bit_lshift_prob"],
        "simple_types_prob": ["void_prob", "char_prob", "int_prob",
                              "short_prob", "long_prob", "long_long_prob",
                              "uchar_prob", "uint_prob", "ushort_prob",
                              "ulong_prob", "ulong_long_prob", "float_prob"],
        "safe_ops_size_prob": ["safe_ops_size_int8", "safe_ops_size_int16",
                               "safe_ops_size_int32", "safe_ops_size_int64"]
    }


'''from Probabilities.cpp'''
def defaultConfig():
    base_itm_single = {
        "more_struct_union_type_prob": 50,
        "bitfields_creation_prob": 50,
        "bitfields_signed_prob": 50,
        "bitfield_in_normal_struct_prob": 10,
        "scalar_field_in_full_bitfields_struct_prob": 10,
        "exhaustive_bitfield_prob": 10,
        "safe_ops_signed_prob": 50,
        "select_deref_pointer_prob": 80, # 0 pointer_as_ltype_prob
        "regular_volatile_prob": 50, # 0
        "regular_const_prob": 10,# 0
        "stricter_const_prob": 50,# 0
        "looser_const_prob": 50, # 0
        "field_volatile_prob": 30, # 0
        "field_const_prob": 20, # 0
        "std_unary_func_prob": 5,
        "shift_by_non_constant_prob": 50,
        "pointer_as_ltype_prob": 50, # 0 select_deref_pointer_prob
        "struct_as_ltype_prob": 30,
        "union_as_ltype_prob": 25,
        "float_as_ltype_prob": 40,# 0
        "new_array_var_prob": 20, # 0
        "access_once_var_prob": 20,
        "inline_function_prob": 50,
        "builtin_function_prob": 50
    }
    base_itm_group = {
        "statement_prob": {"statement_assign_prob": 100, "statement_block_prob": 0,
                           "statement_for_prob": 30, "statement_ifelse_prob": 15,
                           "statement_return_prob": 35, "statement_continue_prob": 40,
                           "statement_break_prob": 45, "statement_goto_prob": 50,
                           "statement_arrayop_prob": 60},
        "assign_unary_ops_prob": {"unary_plus_prob": 25, "unary_minus_prob": 25,
                                  "unary_not_prob": 75, "unary_bit_not_prob": 100},
        "assign_binary_ops_prob": {"binary_add_prob": 100.0/18, "binary_sub_prob": 200.0/18,
                                   "binary_mul_prob": 300.0/18, "binary_div_prob": 400.0/18,
                                   "binary_mod_prob": 500.0/18, "binary_gt_prob": 600.0/18,
                                   "binary_lt_prob": 700.0/18, "binary_ge_prob": 800.0/18,
                                   "binary_le_prob": 900.0/18, "binary_eq_prob": 1000.0/18,
                                   "binary_ne_prob": 1100.0/18, "binary_and_prob": 1200.0/18,
                                   "binary_or_prob": 1300.0/18, "binary_bit_xor_prob": 1400.0/18,
                                   "binary_bit_and_prob": 1500.0/18, "binary_bit_or_prob": 1600.0/18,
                                   "binary_bit_rshift_prob": 1700.0/18, "binary_bit_lshift_prob": 1800.0/18},
        "simple_types_prob": {"void_prob": 0, "char_prob": 10, "int_prob": 20,
                              "short_prob": 30, "long_prob": 40, "long_long_prob": 90,
                              "uchar_prob": 60, "uint_prob": 70, "ushort_prob": 80,
                              "ulong_prob": 50, "ulong_long_prob": 100, "float_prob": 0},
        "safe_ops_size_prob": {"safe_ops_size_int8": 25, "safe_ops_size_int16": 50,
                               "safe_ops_size_int32": 75, "safe_ops_size_int64": 100}
    }
    return base_itm_single, base_itm_group

def ASEConfig():
    base_itm_single = {
        "more_struct_union_type_prob": 50.0,
        "bitfields_creation_prob": 50.0,
        "bitfields_signed_prob": 56.0540511,
        "bitfield_in_normal_struct_prob": 10.0,
        "scalar_field_in_full_bitfields_struct_prob": 10.0,
        "exhaustive_bitfield_prob": 10.0,
        "safe_ops_signed_prob": 50.0,
        "select_deref_pointer_prob": 80.0,
        "regular_volatile_prob": 50.0,
        "regular_const_prob": 10.0,
        "stricter_const_prob": 50.0,
        "looser_const_prob": 49.89045548,
        "field_volatile_prob": 30.0,
        "field_const_prob": 20.0,
        "std_unary_func_prob": 5.0,
        "shift_by_non_constant_prob": 50.07385808,
        "pointer_as_ltype_prob": 50.0,
        "struct_as_ltype_prob": 30.0,
        "union_as_ltype_prob": 25.0,
        "float_as_ltype_prob": 0.0,
        "new_array_var_prob": 20.0,
        "access_once_var_prob": 20.0,
        "inline_function_prob": 50.0,
        "builtin_function_prob": 50.0
    }
    base_itm_group = {
        "statement_prob": {"statement_assign_prob": 40.30477293, "statement_block_prob": 0,
                           "statement_for_prob": 55.08220903, "statement_ifelse_prob": 70.05480602,
                           "statement_return_prob": 75.04567168, "statement_continue_prob": 80.03653734,
                           "statement_break_prob": 85.027403, "statement_goto_prob": 90.01826866,
                           "statement_arrayop_prob": 100.0},
        "assign_unary_ops_prob": {"unary_plus_prob": 24.98363383, "unary_minus_prob": 49.96726766,
                                  "unary_not_prob": 74.95090149, "unary_bit_not_prob": 100.0},
        "assign_binary_ops_prob": {"binary_add_prob": 5.55537201, "binary_sub_prob": 11.11074402,
                                   "binary_mul_prob": 16.66611603, "binary_div_prob": 22.20038084,
                                   "binary_mod_prob": 27.75575285, "binary_gt_prob": 33.31849759,
                                   "binary_lt_prob": 38.8738696, "binary_ge_prob": 44.44456409,
                                   "binary_le_prob": 49.9999361, "binary_eq_prob": 55.55530811,
                                   "binary_ne_prob": 61.12273779, "binary_and_prob": 66.6781098,
                                   "binary_or_prob": 72.23348181, "binary_bit_xor_prob": 77.78885382,
                                   "binary_bit_and_prob": 83.33959289, "binary_bit_or_prob": 88.8949649,
                                   "binary_bit_rshift_prob": 94.4375762, "binary_bit_lshift_prob": 100.0},
        "simple_types_prob": {"void_prob": 0, "char_prob": 9.96410344, "int_prob": 19.95662017,
                              "short_prob": 29.9491369, "long_prob": 39.94165363, "long_long_prob": 49.93417036,
                              "uchar_prob": 60.04619209, "uint_prob": 70.03870882, "ushort_prob": 80.06652526,
                              "ulong_prob": 90.03796406, "ulong_long_prob": 100, "float_prob": 0},
        "safe_ops_size_prob": {"safe_ops_size_int8": 24.99500907, "safe_ops_size_int16": 50.00998187,
                               "safe_ops_size_int32": 75.00499094, "safe_ops_size_int64": 100.0}
    }
    return base_itm_single,  base_itm_group

def exccmd2arr(cmd):
    p = os.popen(cmd, "r")
    rs = []
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    gc.collect() # free mem
    return rs

def exccmd2string(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    ss = ''
    for item in rs:
        ss += item
    gc.collect() # free mem
    return ss

def exccmd(cmd):
    return os.system(cmd)


class Environment:
    def __init__(self):
        self.fv_history = []
        self.base_itm_single, self.base_itm_group = defaultConfig()
        self.vals = self.dic2vals(self.base_itm_single, self.base_itm_group)
        self.s_dim = len(self.vals)


    def vals2dic(self, vals):
        if len(vals) == 0: # if data is not complete, return empty dic
            return {},{}
        # else change
        base_itm_single = {}
        base_itm_group = {}
        index = 0
        for key in single_keys:
            base_itm_single[key] = vals[index]
            index +=1
        for group in group_keys:
            group_dir = {}
            for key in group_items_keys[group]:
                group_dir[key] = vals[index]
                index += 1
            base_itm_group[group] = group_dir
        return base_itm_single, base_itm_group

    def dic2vals(self, base_itm_single, base_itm_group):
        singlevals = []
        for key in single_keys:
            singlevals.append(base_itm_single[key])
        groupvals = []
        for group in group_keys:
            group_dir = base_itm_group[group]
            for key in group_items_keys[group]:
                groupvals.append(group_dir[key])

        vals = singlevals + groupvals
        return vals

    """input@param is direct output arr from csmith
        return@param feature vector"""
    def feature2vector(self, feature):
        feature_vector = []
        for index in range(len(feature) - 1):
            # print(feature[index].split(" ")[-1])
            feature_vector.append(int(feature[index].split(" ")[-1]))
        return feature_vector

    def get_extra_features(self, prog_name):
        def get_val(string):
            res = string.split(': ')[-1]
            res = res.split('\n')[0]
            if type(eval(res)) == float:
                return float(res)
            return int(res)

        prog = exccmd2arr('cat ' + prog_name)
        begin_key = '/************************ statistics *************************\n'
        begin_index = prog.index(begin_key)
        feature_zone = prog[begin_index:]
        extra_features = {
            'XXX max struct depth': 0, 'XXX max struct depth occurrence': [],  # 0-
            'total union variables': 0,
            'non-zero bitfields defined in structs': 0, 'zero bitfields defined in structs': 0,
            'const bitfields defined in structs': 0, 'volatile bitfields defined in structs': 0,
            'structs with bitfields in the program': 0, 'full-bitfields structs in the program': 0,
            'times a bitfields struct\'s address is taken': 0, 'times a bitfields struct on LHS': 0,
            'times a bitfields struct on RHS': 0, 'times a single bitfield on LHS': 0,
            'times a single bitfield on RHS': 0,
            'XXX max expression depth': 0, 'XXX max expression depth occurrence': [],  # 1-
            'total number of pointers': 0, 'times a variable address is taken': 0,
            'XXX times a pointer is dereferenced on RHS': 0,
            'XXX times a pointer is dereferenced on RHS occurrence': [],  # 1-
            'XXX times a pointer is dereferenced on LHS': 0,
            'XXX times a pointer is dereferenced on LHS occurrence': [],  # 1-
            'times a pointer is compared with null': 0,
            'times a pointer is compared with address of another variable': 0,
            'times a pointer is compared with another pointer': 0, 'times a pointer is qualified to be dereferenced': 0,
            'XXX max dereference level': 0, 'XXX max dereference level occurrence': [],  # 0-
            'number of pointers point to pointers': 0, 'number of pointers point to scalars': 0,
            'number of pointers point to structs': 0, 'percent of pointers has null in alias set': 0,
            'average alias set size': 0, 'times a non-volatile is read': 0, 'times a non-volatile is write': 0,
            'times a volatile is read': 0,
            'times read thru a pointer': 0,
            'times a volatile is write': 0,
            'times written thru a pointer': 0,
            'times a volatile is available for access': 0, 'percentage of non-volatile access': 0,
            'forward jumps': 0, 'backward jumps': 0,
            'stmts': 0,
            'XXX max block depth': 0, 'XXX max block depth occurrence': [],  # 0-
            'percentage a fresh-made variable is used': 0, 'percentage an existing variable is used': 0

        }

        for key in extra_features.keys():
            # print(key)
            # print(key.find(" occurrence")!= -1)
            if key.find(" occurrence") != -1:
                continue
                # key_name = key.replace(' occurrence', '')
                # depth_count_begin = 0 # begin with index 1, so add 0
                # if key_name=='XXX max struct depth' or key_name=='XXX max dereference level':
                #     depth_count_begin = 1 # begin with index 0, so add 1
                # key_begin_index = -1
                #
                #
                # for feature in feature_zone:
                #     if key_name in feature:
                #         key_begin_index = feature_zone.index(feature)+2 # +1 is breakdown:, +2 is depth: 0/1
                # key_len = depth_count_begin + extra_features[key_name]
                # key_feature_zone = feature_zone[key_begin_index:key_begin_index+key_len]
                # for key_feature in key_feature_zone:
                #     # print("s "+key_feature.split('depth: ')[-1])
                #     if key_feature.startswith('   level: '):
                #         extra_features[key].append(get_val(key_feature.split('level: ')[-1]))
                #     elif key_feature.startswith('   depth: '):
                #         extra_features[key].append(get_val(key_feature.split('depth: ')[-1]))
                #     else:
                #         break

            else:
                for feature in feature_zone:
                    if key in feature:
                        # print(key)
                        extra_features[key] = get_val(feature)
                        # print(extra_features[key])

        extra_features_vals = []
        for item in extra_features.values():
            if isinstance(item, list):
                for i in item:
                    extra_features_vals.append(i)
            else:
                extra_features_vals.append(item)

        return extra_features_vals

    """
    L1 distances of a list of vector
    """

    def manhattan_distance(self, feature_vector, feature_vectors):
        def norm(v, vs):
            vs = np.array(vs)
            # print(vs,'\n',v)
            # print(len(vs[0]),len(v))
            m = np.append(vs, [v], axis=0)
            m_normed = m / m.max(axis=0)
            where_are_nan = np.isnan(m_normed)  # xmax=0, div by 0
            m_normed[where_are_nan] = 0
            return m_normed.tolist()
        # print('feature_vector',feature_vector)
        # print('feature_vectors',feature_vectors)
        m = norm(feature_vector, feature_vectors)
        # print('norm',m)
        fv_normed = m[-1]
        fvs_normed = m[:-1]
        # print('norm feature_vector', fv_normed)
        # print('norm feature_vectors', fvs_normed)

        distance = 0
        for fv in fvs_normed:
            d = np.sum(np.abs(np.array(fv_normed) - np.array(fv)),
                              axis=0)  # add distances of every options in two config
            distance += d / len(feature_vector)  # mean distance for every option
        # print('distance',distance)
        return distance / len(feature_vectors)  # mean distance, between feature_vector and items in feature_vectors

    def get_offline_prediction(self, model_path, X, y):

        from sklearn.externals import joblib
        clf = joblib.load(model_path)

        if 'rf' in model_path:
            clf.n_jobs = 1
        cat = clf.predict(X)
        # print(cat)
        res = clf.score(X, y)
        return cat, res #[0,0,1,1] 0.86

    def distance_baseline(self, fv, baseline_type):
        if baseline_type == 'default':
            baseline = 0.2
        elif baseline_type == 'ase':
            baseline = 0.3
        else:
            return 0
        return baseline

    def constrainCheck(self):
        import copy
        def csmithFormat2proportionValue(csmith_values):
            csmith_value_sorted = copy.deepcopy(csmith_values)
            csmith_value_sorted.sort()
            proportion_values = []
            proportion_add_info = [] # 在 csmith_value 中的位置
            for v in csmith_values:
                index = csmith_value_sorted.index(v)  # v is the (index+1)^th small item in its group
                while (proportion_add_info.count(index) != 0):
                    index += 1
                proportion_add_info.append(index)
                proportion_v = csmith_value_sorted[index]
                if index != 0:
                    proportion_v -= csmith_value_sorted[index - 1]
                proportion_values.append(proportion_v)

            while(proportion_values.count(0) != 0):
                pos0 = proportion_values.index(0)
                proportion_values[pos0] = 1

            # for v in proportion_values: # 等比例缩放 使sum=100
            #     v *= 100/sum(proportion_values)
            bias = 0
            overflow = sum(proportion_values)-100
            while (overflow != 0):
                pos = (overflow + bias) % len(proportion_values)
                if proportion_values[pos] > 1:
                    proportion_values[pos] -= 1
                    overflow -= 1
                else:
                    bias += 1
            return proportion_values, proportion_add_info

        def proportionValue2csmithFormat(proportion_values, proportion_add_info):
            csmith_values = []
            for index in range(len(proportion_values)):
                csmith_v = proportion_values[index]
                for i in range(proportion_add_info[index]):
                    add_pos = proportion_add_info.index(i)
                    csmith_v += proportion_values[add_pos]
                csmith_values.append(csmith_v)
            return csmith_values

        print("checking Constrain")
        for group in group_keys:
            group_dic = self.base_itm_group[group]
            ## constrain4 ##
            to100key = max(group_dic, key=group_dic.get)
            if self.base_itm_group[group][to100key] != 100.0:
                print("constrain4")
                print(self.base_itm_group[group][to100key])
                self.base_itm_group[group][to100key] = 100.0
                print(self.base_itm_group[group])

            ## constrain3 ##
            keys = group_items_keys[group]
            values = [int(group_dic[key]) for key in keys]
            valuesset = list(set(values))
            if len(valuesset) != len(values):  # dumplicates elem exist
                print("constrain3")
                ifcons3 = True
                proportion, addinfo = csmithFormat2proportionValue(values)
                csmith_vals = proportionValue2csmithFormat(proportion, addinfo)
                print(group)
                print("proportion", end=": ")
                print(proportion)
                print("addinfo", end=": ")
                print(addinfo)
                print("csmith_vals", end=": ")
                print(csmith_vals)
                self.base_itm_group[group] = dict(zip(keys, csmith_vals))
                print(self.base_itm_group[group])
            ## constrain2 ##
            if self.base_itm_group['simple_types_prob']['void_prob'] != 0:
                print("constrain2")
                self.base_itm_group['simple_types_prob']['void_prob'] = 0
                print(self.base_itm_group['simple_types_prob'])

    def reset(self):
        self.base_itm_single, self.base_itm_group = defaultConfig()
        self.vals = self.dic2vals(self.base_itm_single, self.base_itm_group)
        return self.vals

    """return@param crash/N_PROGRAMS 
    return@param intra diversity of current configration
    return@paramaverage feature vectors of N_PROGRAMS generated programs
    """
    def execute_config(self, n):
        exccmd('mkdir ' + crashres)
        exccmd('mkdir ' + wrongcoderes)

        exccmd('rm trainprogram*')
        res = open('status.txt', 'a')
        res.flush()
        config = self.dic2vals(self.base_itm_single, self.base_itm_group)
        feature_vectors = []
        bilabels = []
        prog_intra_diversity = 0
        generation = 0

        configStr = '[ '
        for item in config:
            configStr += str(item)
            configStr += ', '
        configStr = configStr[:-2]
        configStr += ' ]\n'
        res.write(configStr)
        config_begin_time = time.time()

        for i in range(1, n + 1):
            cmd_program = 'timeout 120 ' + csmith + ' --probability-configuration ' + configFile + ' -o trainprogram' + str(
                i) + '.c'
            cmd_get_seed = 'cat trainprogram' + str(i) + '.c  | grep Seed | awk -F \' \' \'{print $3}\''
            cmd_binary0 = 'timeout 60 ' + gcc + ' -O0 trainprogram' + str(
                i) + '.c -I ' + library + ' -o trainprogram' + str(i) + '_0 > mute'
            cmd_binary1 = 'timeout 60 ' + gcc + ' -O1 trainprogram' + str(
                i) + '.c -I ' + library + ' -o trainprogram' + str(i) + '_1 > mute'
            cmd_binary2 = 'timeout 60 ' + gcc + ' -O2 trainprogram' + str(
                i) + '.c -I ' + library + ' -o trainprogram' + str(i) + '_2 > mute'
            cmd_binary3 = 'timeout 60 ' + gcc + ' -O3 trainprogram' + str(
                i) + '.c -I ' + library + ' -o trainprogram' + str(i) + '_3 > mute'
            cmd_binarys = 'timeout 60 ' + gcc + ' -Os trainprogram' + str(
                i) + '.c -I ' + library + ' -o trainprogram' + str(i) + '_s > mute'
            cmd_checksum0 = 'timeout 60 ./trainprogram' + str(i) + '_0 > trainprogram' + str(i) + '_out_0'
            cmd_checksum1 = 'timeout 60 ./trainprogram' + str(i) + '_1 > trainprogram' + str(i) + '_out_1'
            cmd_checksum2 = 'timeout 60 ./trainprogram' + str(i) + '_2 > trainprogram' + str(i) + '_out_2'
            cmd_checksum3 = 'timeout 60 ./trainprogram' + str(i) + '_3 > trainprogram' + str(i) + '_out_3'
            cmd_checksums = 'timeout 60 ./trainprogram' + str(i) + '_s > trainprogram' + str(i) + '_out_s'
            cmd_diff1 = 'diff trainprogram' + str(i) + '_out_0 trainprogram' + str(i) + '_out_1 > trainprogram' + str(
                i) + '_diff_1'
            cmd_diff2 = 'diff trainprogram' + str(i) + '_out_0 trainprogram' + str(i) + '_out_2 > trainprogram' + str(
                i) + '_diff_2'
            cmd_diff3 = 'diff trainprogram' + str(i) + '_out_0 trainprogram' + str(i) + '_out_3 > trainprogram' + str(
                i) + '_diff_3'
            cmd_diffs = 'diff trainprogram' + str(i) + '_out_0 trainprogram' + str(i) + '_out_s > trainprogram' + str(
                i) + '_diff_s'

            start = time.time()
            prog_start = start
            feature = exccmd2arr(cmd_program)
            end = time.time()
            if (end - start) >= 120:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',generation \n')
                generation += 1
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',generation,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                continue

            if len(feature) == 0:
                continue
            seed = exccmd2string(cmd_get_seed)
            fv = self.feature2vector(feature)
            if extra_feature:
                fv += self.get_extra_features('trainprogram' + str(i) + '.c')

            ## generate binary code for O0 O1 O2 O3 Os ##
            start = time.time()
            exccmd(cmd_binary0)
            end = time.time()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile \n')
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',compile,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary1)
            end = time.time()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile \n')
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',compile,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary2)
            end = time.time()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile \n')
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',compile,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary3)
            end = time.time()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile \n')
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',compile,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binarys)
            end = time.time()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile \n')
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',compile,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
                # i=i-1
                continue

            if not os.path.exists('./trainprogram' + str(i) + '_0') or not os.path.exists(
                                './trainprogram' + str(i) + '_1') or not os.path.exists(
                                './trainprogram' + str(i) + '_2') or not os.path.exists(
                                './trainprogram' + str(i) + '_3') or not os.path.exists(
                                './trainprogram' + str(i) + '_s'):
                res.write(str(i) + ',crash, ' + seed + '\n')
                # exccmd('mkdir ' + crashres + '/trainprogram' + str(i))
                # exccmd('mv ' + 'trainprogram* ' + crashres + '/trainprogram' + str(i))

                # prog_fail += 1
                bilabels.append(categories_dir['crash'])
                feature_vectors.append(fv) # fv and bilabel append at the same time
                if check_time:
                    prog_end = end
                    timelog = open('time.txt', 'a')
                    timelog.write('config id:'+str(len(self.fv_history)+1)+', id'+str(i) + ',crash,'+str(prog_end-prog_start)+'\n')
                    timelog.flush()
                    timelog.close()
            else:
                start = time.time()
                exccmd(cmd_checksum0)
                end = time.time()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',execute,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                    # i=i-1
                    continue
                start = time.time()
                exccmd(cmd_checksum1)
                end = time.time()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',execute,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                    # i=i-1
                    continue
                start = time.time()
                exccmd(cmd_checksum2)
                end = time.time()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',execute,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                    # i=i-1
                    continue
                start = time.time()
                exccmd(cmd_checksum3)
                end = time.time()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',execute,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                    # i=i-1
                    continue
                start = time.time()
                exccmd(cmd_checksums)
                end = time.time()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',execute,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                    # i=i-1
                    continue

                f = open('trainprogram' + str(i) + '_out_0')
                lines = f.readlines()
                f.close()
                if len(lines) == 0:
                    res.write(str(i) + ',wrongcode, ' + seed + '\n')
                    # exccmd('mkdir ' + wrongcoderes + '/trainprogram' + str(i))
                    # exccmd('mv ' + 'trainprogram* ' + wrongcoderes + '/trainprogram' + str(i))

                    # prog_fail += 1
                    bilabels.append(categories_dir['wrongcode'])
                    feature_vectors.append(fv)
                    if check_time:
                        prog_end = end
                        timelog = open('time.txt', 'a')
                        timelog.write(
                            'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',wrongcode,' + str(
                                prog_end - prog_start)+'\n')
                        timelog.flush()
                        timelog.close()
                else:
                    exccmd(cmd_diff1)
                    exccmd(cmd_diff2)
                    exccmd(cmd_diff3)
                    exccmd(cmd_diffs)
                    f = open('trainprogram' + str(i) + '_diff_1')
                    lines1 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) + '_diff_2')
                    lines2 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) + '_diff_3')
                    lines3 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) + '_diff_s')
                    lines4 = f.readlines()
                    f.close()
                    if not (len(lines1) == 0 and len(lines2) == 0 and len(lines3) == 0 and len(lines4) == 0):
                        res.write(str(i) + ',wrongcode, ' + seed + '\n')
                        # exccmd('mkdir ' + wrongcoderes + '/trainprogram' + str(i))
                        # exccmd('mv ' + 'trainprogram* ' + wrongcoderes + '/trainprogram' + str(i))

                        # prog_fail += 1
                        bilabels.append(categories_dir['wrongcode'])
                        feature_vectors.append(fv)
                        if check_time:
                            prog_end = end
                            timelog = open('time.txt', 'a')
                            timelog.write(
                                'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',wrongcode,' + str(
                                    prog_end - prog_start)+ '\n')
                            timelog.flush()
                            timelog.close()
                    else:
                        res.write(str(i) + ',correct, ' + seed + '\n')
                        bilabels.append(categories_dir['correct'])
                        feature_vectors.append(fv)
                        if check_time:
                            prog_end = end
                            timelog = open('time.txt', 'a')
                            timelog.write(
                                'config id:' + str(len(self.fv_history) + 1) + ', id' + str(i) + ',correct,' + str(
                                    prog_end - prog_start)+ '\n')
                            timelog.flush()
                            timelog.close()
        exccmd('rm trainprogram*')
        res.close()
        print(feature_vectors)
        print(bilabels)
        config_end_time = time.time()
        if check_time:
            prog_end = end
            timelog = open('time.txt', 'a')
            timelog.write('config id:' + str(len(self.fv_history) + 1) + ', total:' + str(config_end_time - config_begin_time) + '\n\n')
            timelog.flush()
            timelog.close()
        if len(feature_vectors)!= 0:
            for fv in feature_vectors:
                prog_intra_diversity += self.manhattan_distance(fv, feature_vectors)

            prog_intra_diversity = prog_intra_diversity / len(feature_vectors)
            actual_fail = bilabels.count(categories_dir['crash'])/(len(bilabels)+generation)# or wrongcode, both is 1
            actual_penetrate = (bilabels.count(categories_dir['crash']) - generation) / (len(bilabels)+generation)
            predict_cat, predict_score = self.get_offline_prediction(model_path,np.array(feature_vectors),np.array(bilabels))
            predict_fail = predict_cat.tolist().count(categories_dir['crash']) / len(predict_cat)

            offline_log = open('offline_log.txt', 'a')
            offline_log.write("predict score: " + str(predict_score) + '\n')
            offline_log.close()

            return actual_penetrate, predict_fail, prog_intra_diversity, list(np.mean(np.array(feature_vectors),axis=0))
        else: # config timeout
            return np.nan, np.nan, np.nan, []

    """
    get score of current configuration just updated in step(.)
    """
    def score(self):
        exccmd('rm ' + configFile)

        ## generate config file for CSmith ##
        fp_w = open(configFile, 'w')
        for key in single_keys:
            fp_w.write(key + '=' + str(self.base_itm_single[key]) + '\n')
            fp_w.write('\n')
        for group in group_keys:
            fp_w.write('[' + group + ',')
            group_txt = ''
            group_dir = self.base_itm_group[group]
            for key in group_items_keys[group]:
                group_txt += key + '=' + str(group_dir[key]) + ','
            fp_w.write(group_txt[:-1] + ']\n')
            fp_w.write('\n')
        fp_w.flush()
        fp_w.close()

        actual_fail, predict_fail, intra_diversity, avg_fv = self.execute_config(N_PROGRAMS)
        res = open('status.txt', 'a')
        if len(avg_fv)!=0:
            self.fv_history.append(avg_fv)
            inter_diversity = self.manhattan_distance(avg_fv, self.fv_history) # distance between this config and previous ones
            reward = actual_fail + predict_fail + intra_diversity + inter_diversity

            # reward =  predict_fail
            # reward =  intra_diversity + inter_diversity - 2 * self.distance_baseline(avg_fv,DISTANCE_BASELINE_TYPE)



            res.write("id: " + str(len(self.fv_history)) +
                  ", reward: " + str(reward) +
                  ", actual fail: " + str(actual_fail) +
                  ", predict fail: "+str(predict_fail)+
                  ", intra: " + str(intra_diversity) +
                  ", inter: " + str(inter_diversity) + '\n')
            res.flush()
            res.close()
            return reward
        else:
            res.write("id: " + str(len(self.fv_history)) +", timeout \n")
            res.flush()
            res.close()
            return np.nan


    def step(self, actions):
        actions_ = []
        res = open('loss_log.txt', 'a')
        res.flush()
        res.write('config:[')
        for v in self.vals:
            res.write(str(v)+',')
        res.write(']\n')
        res.write('action:[')
        for a in actions:
            a_ = a - torch.tensor(len(ACTION_BOUND) / 2).long()
            actions_.append(a_)
            res.write(str(a_)+',')
        res.write(']')
        res.close()

        old_vals = self.vals
        for i in range(len(actions)):
            self.vals[i] = np.clip(self.vals[i] + actions_[i], 0, 100).item()
        self.base_itm_single, self.base_itm_group = self.vals2dic(self.vals)
        self.constrainCheck()
        reward = self.score()
        if pd.isna(reward):
            self.vals = old_vals
            self.base_itm_single, self.base_itm_group = self.vals2dic(self.vals)
            return np.array(self.vals), reward, False, {}
        else:
            return np.array(self.vals), reward, True, {}






