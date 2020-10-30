'''
add new logic to handel constrain3
remain tobe test...
'''

# -*- coding: utf-8 -*-
import random, copy
import numpy as np
import os,sys,psutil
import tensorflow as tf
import time
# import copy
import gc


EPISODES = 50
NUMS_OF_OPTION = 5 #70
batch_size = 32
# csmith_home = "../../csmith_recorder/"
N_PROGRAMS = 10 #100
GLOBAL_ENV_SCOPE = 'Global_Env'

check_mem = True
check_time = True



# gcc='gcc'
# csmith_home = '../../csmith_recorder/csmith_record_2.3.0_t1/'
# csmith = csmith_home+'../csmith'

gcc='gcc-4.4'
csmith_home = '/csmith_recorder/csmith_record_2.3.0_t1/'
csmith= csmith_home + '../csmith'
model_path = 'xxx.pkl'

library = csmith_home+'runtime'
passres = './pass'
crashres = './crash'
wrongcoderes = './wrongcode'
configFile = 'config1'

categories_dir={
    'pass':0,
    'fail':1,
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
    # print(p)
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
        # print(line)
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
    def __init__(self, scope):
        if scope == GLOBAL_ENV_SCOPE:
            self.fv_history = []
        self.base_itm_single,self.base_itm_group = defaultConfig()
        singlevals = []
        for key in single_keys:
            singlevals.append(self.base_itm_single[key])
        groupvals = []
        for group in group_keys:
            group_dir = self.base_itm_group[group]
            for key in group_items_keys[group]:
                groupvals.append(group_dir[key])

        self.vals = singlevals + groupvals

        res = open('status.txt', 'a')
        res.write("initail vals: \n")
        for val in self.vals:
            res.write(str(val)+',')
        res.flush()
        res.close()

        # print(self.vals)
        # self.vals = self.dic2vals(self.base_itm_single, self.base_itm_group)
        self.state_size = len(self.vals)

    def vals2dic(self, vals):
        if len(vals) != self.state_size: # if data is not complete, return empty dic
            return {},{}
        # else change
        base_itm_single = {}
        base_itm_group = {}
        index = 0
        for key in single_keys:
            base_itm_single[key] = vals[index]
            index +=1
        for group in group_keys:
            group_dir = self.base_itm_group[group]
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
        if len(vals) != self.state_size: # if data is not complete
            return []
        return vals



    def reset(self):
        # ase
        self.base_itm_single, self.base_itm_group = defaultConfig()
        singlevals = []
        for key in single_keys:
            singlevals.append(self.base_itm_single[key])
        groupvals = []
        for group in group_keys:
            group_dir = self.base_itm_group[group]
            for key in group_items_keys[group]:
                groupvals.append(group_dir[key])

        self.vals = singlevals+groupvals

        # self.vals = self.dic2vals(self.base_itm_single,self.base_itm_group)
        self.state_size = len(self.vals)
        return np.array(self.vals)

    """
    actions@param configuration, used to generate N_PROGRAMS programs using CSmith
    vec@param save average features of current configuration into vec, which keeps average feature vector of all configuration
    res@param record every programs' status and seed for each configuration in file res
    """
    def step(self, actions, vec):
        if check_time:
            rectime = open('time.txt', 'a')
            rectime.write("id: " + str(time.time()) +'\n')
            rectime.flush()
            rectime.close()
        if len(actions) != self.state_size:
            return
        # self.vals = actions
        # ddpg 那样action加入随机元素处理?
        ## constrain1
        for i in range(len(actions)):
            self.vals[i] = np.clip(self.vals[i]+actions[i] , 0, 100)
        self.base_itm_single, self.base_itm_group = self.vals2dic(self.vals)
        self.constrainCheck()
        crash, predict_crash,intra_dis_sum, inter_dis = self.score(vec)


        # reward = crash/N_PROGRAMS + predict_crash + intra_dis_sum/N_PROGRAMS + inter_dis
        reward = intra_dis_sum/N_PROGRAMS + inter_dis


        res = open('status.txt', 'a')
        res.write("id: " + str(len(vec))+
                  ", reward: " + str(reward)+
                  ", crash: "+str(crash/N_PROGRAMS)+
                  # ", predict: "+str(predict_crash)+
                  ", intra: " +str(intra_dis_sum/N_PROGRAMS)+
                  ", inter: " + str(inter_dis)+'\n')
        res.flush()
        res.close()


        done = True  # 更新targetnet的时机
        # print("s_ :")
        # print(np.array(self.vals))
        # print("reward:")
        # print(-reward)
        return np.array(self.vals), reward, True, {}

    """input@param is direct output arr from csmith
    return@param feature vector"""
    def feature2vector(self, feature):
        print("feature is")
        print(feature)
        # print(len(feature))
        feature_vector = []
        for index in range(len(feature) - 1):
            # print(feature[index].split(" ")[-1])
            feature_vector.append(int(feature[index].split(" ")[-1]))
        return feature_vector

    """
    input@param is direct output arr from csmith
    dis@param distance between prog and average ps
    """
    def manhattan_distance(self, feature_vector, feature_vectors):
        dis = 0
        feature_vector = np.array(feature_vector)
        fv_normed = feature_vector / feature_vector.max(axis=0) # nomalization
        fv_normed = fv_normed.tolist()
        fvs_normed = []
        # print("manhattan_distance")
        # print(feature_vectors)
        for fv in feature_vectors:
            if len(fv) == 0:
                feature_vectors.remove(fv)
            else:
                fv = np.array(fv)
                norm = fv / fv.max(axis=0)  # nomalization
                fvs_normed.append(norm.tolist())

        # print(fv_normed)
        # print(fvs_normed)
        for fv in fvs_normed:
            distance = np.sum(np.abs(np.array(fv_normed) - np.array(fv)), axis=0) # add distances of every options in two config
            dis += distance/len(feature_vector) # mean distance for every option
            # print(distance)
        # return
        return dis/len(feature_vectors) # mean distance, between feature_vector and items in feature_vectors

    def get_offline_prediction(self, model_path, X, y):

        from sklearn.externals import joblib
        clf = joblib.load(model_path)

        if 'rf' in model_path:
            clf.n_jobs = 1
        cat = clf.predict(X)
        # print(cat)
        res = clf.score(X, y)
        return cat, res

    """return@param crash/N_PROGRAMS 
    return@param intra diversity of current configration
    return@paramaverage feature vectors of N_PROGRAMS generated programs
    """
    def execute_config(self, n):
        exccmd('rm trainprogram*')
        # print("into execute_config")
        res = open('status.txt', 'a')
        res.flush()
        # print("exectue config in[ut param 2 dic2vals")
        # print(self.base_itm_single)
        # print(self.base_itm_group)
        config = self.dic2vals(self.base_itm_single, self.base_itm_group)
        feature_vectors = []
        bilabels = []
        # prog_fail = 0
        # prog_exe = 0
        # prog_pass = 0
        prog_intra_diversity = 0

        configStr = '[ '
        for item in config:
            configStr += str(item)
            configStr += ', '
            # print(str(item)+', ')
        # print(configStr)
        configStr = configStr[:-2]
        # print(configStr)
        configStr += ' ]\n'
        # print(configStr)
        res.write(configStr)



        for i in range(1,n+1):
            cmd_program = 'timeout 120 ' + csmith +' --probability-configuration ' + configFile +' -o trainprogram'+str(i)+'.c'
            cmd_get_seed = 'cat trainprogram'+str(i)+'.c  | grep Seed | awk -F \' \' \'{print $3}\''
            cmd_binary0 = 'timeout 60 ' + gcc + ' -O0 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_0 > mute'
            cmd_binary1 = 'timeout 60 ' + gcc + ' -O1 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_1 > mute'
            cmd_binary2 = 'timeout 60 ' + gcc + ' -O2 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_2 > mute'
            cmd_binary3 = 'timeout 60 ' + gcc + ' -O3 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_3 > mute'
            cmd_binarys = 'timeout 60 ' + gcc + ' -Os trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_s > mute'
            cmd_checksum0 = 'timeout 60 ./trainprogram'+str(i)+'_0 > trainprogram'+str(i)+'_out_0'
            cmd_checksum1 = 'timeout 60 ./trainprogram'+str(i)+'_1 > trainprogram'+str(i)+'_out_1'
            cmd_checksum2 = 'timeout 60 ./trainprogram'+str(i)+'_2 > trainprogram'+str(i)+'_out_2'
            cmd_checksum3 = 'timeout 60 ./trainprogram'+str(i)+'_3 > trainprogram'+str(i)+'_out_3'
            cmd_checksums = 'timeout 60 ./trainprogram'+str(i)+'_s > trainprogram'+str(i)+'_out_s'
            cmd_diff1 = 'diff trainprogram'+str(i)+'_out_0 trainprogram'+str(i)+'_out_1 > trainprogram'+str(i)+'_diff_1'
            cmd_diff2 = 'diff trainprogram'+str(i)+'_out_0 trainprogram'+str(i)+'_out_2 > trainprogram'+str(i)+'_diff_2'
            cmd_diff3 = 'diff trainprogram'+str(i)+'_out_0 trainprogram'+str(i)+'_out_3 > trainprogram'+str(i)+'_diff_3'
            cmd_diffs = 'diff trainprogram'+str(i)+'_out_0 trainprogram'+str(i)+'_out_s > trainprogram'+str(i)+'_diff_s'


            start = time.time()
            if(check_mem):
                process = psutil.Process(os.getpid())
                print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

            feature = exccmd2arr(cmd_program)
            if len(feature) == 0:
                continue
            # print(cmd_program)
            if(check_mem):
                process = psutil.Process(os.getpid())
                print('Used Memory after excarr:', process.memory_info().rss / 1024 / 1024, 'MB')

            seed = exccmd2string(cmd_get_seed)
            if(check_mem):
                process = psutil.Process(os.getpid())
                print('Used Memory after str:', process.memory_info().rss / 1024 / 1024, 'MB')
            # print("seed is ")
            # print(seed)
            end = time.time()
            fv = self.feature2vector(feature)
            print("GET FV HERE")
            print(feature) #
            print(fv)
            feature_vectors.append(fv)
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate program:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',generation, ' + seed + '\n')
                # i=i-1
                continue

            start = time.time()
            exccmd(cmd_binary0)
            end = time.time()
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate binary0:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile, ' + seed + '\n')
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary1)
            end = time.time()
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate binary1:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile, ' + seed + '\n')
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary2)
            end = time.time()
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate binary2:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile, ' + seed + '\n')
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binary3)
            end = time.time()
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate binary3:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile, ' + seed + '\n')
                # i=i-1
                continue
            start = time.time()
            exccmd(cmd_binarys)
            end = time.time()
            if check_time:
                rectime = open('time.txt', 'a')
                rectime.write("generate binarys:"+str(end - start)+'\n')
                rectime.flush()
                rectime.close()
            if (end - start) >= 60:
                exccmd('rm trainprogram*')
                res.write(str(i) + ',compile, ' + seed + '\n')
                # i=i-1
                continue

            if not os.path.exists('./trainprogram' + str(i) + '_0') or not os.path.exists(
                                './trainprogram' + str(i) + '_1') or not os.path.exists(
                                './trainprogram' + str(i) + '_2') or not os.path.exists(
                                './trainprogram' + str(i) + '_3') or not os.path.exists(
                                './trainprogram' + str(i) + '_s'):
                res.write(str(i) + ',crash, ' + seed + '\n')
                # prog_fail += 1
                bilabels.append(categories_dir['fail'])
            else:
                start = time.time()
                exccmd(cmd_checksum0)
                end = time.time()
                if check_time:
                    rectime = open('time.txt', 'a')
                    rectime.write("generate checksum0:" + str(end - start)+'\n')
                    rectime.flush()
                    rectime.close()
                if (end - start) >= 60:
                    exccmd('rm trainprogram*')
                    res.write(str(i) + ',execute, ' + seed + '\n')
                    # i=i-1
                    continue

                exccmd(cmd_checksum1)
                exccmd(cmd_checksum2)
                exccmd(cmd_checksum3)
                exccmd(cmd_checksums)
                if check_time:
                    rectime = open('time.txt', 'a')
                    rectime.write("done generate checksum23s:" + str(time.time())+'\n')
                    rectime.flush()
                    rectime.close()

                f = open('trainprogram' + str(i) +'_out_0')
                lines = f.readlines()
                f.close()

                if len(lines) == 0:
                    res.write(str(i) + ',wrongcode, ' + seed + '\n')
                    # prog_fail += 1
                    bilabels.append(categories_dir['fail'])

                else:
                    exccmd(cmd_diff1)
                    exccmd(cmd_diff2)
                    exccmd(cmd_diff3)
                    exccmd(cmd_diffs)
                    if check_time:
                        rectime = open('time.txt', 'a')
                        rectime.write("generate diffs:" + str(time.time())+'\n')
                        rectime.flush()
                        rectime.close()

                    f = open('trainprogram' + str(i) +'_diff_1')
                    lines1 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) +'_diff_2')
                    lines2 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) +'_diff_3')
                    lines3 = f.readlines()
                    f.close()
                    f = open('trainprogram' + str(i) +'_diff_s')
                    lines4 = f.readlines()
                    f.close()

                    if not (len(lines1) == 0 and len(lines2) == 0 and len(lines3) == 0 and len(lines4) == 0):
                        res.write(str(i) + ',wrongcode, ' + seed + '\n')
                        # prog_fail += 1
                        bilabels.append(categories_dir['fail'])

                    else:
                        res.write(str(i) + ',correct, ' + seed + '\n')
                        bilabels.append(categories_dir['pass'])
                    if check_time:
                        rectime = open('time.txt', 'a')
                        rectime.write("check diffs:" + str(time.time())+'\n')
                        rectime.flush()
                        rectime.close()
        exccmd('rm trainprogram*')

        res.close()
        for fv in feature_vectors:
            print(feature_vectors)
            prog_intra_diversity += self.manhattan_distance(fv, feature_vectors)
        # return
        actual_fail = bilabels.count(categories_dir['fail'])
        predict_cat,predict_score = self.get_offline_prediction(model_path,np.array(feature_vectors),np.array(bilabels))
        predict_fail = predict_cat.count(categories_dir['fail'])

        print(bilabels)
        print(predict_cat)
        print(predict_score)
        print(len(feature_vectors))
        print(len(bilabels))


        return actual_fail/len(feature_vectors), predict_fail/len(feature_vectors), prog_intra_diversity, list(np.mean(np.array(feature_vectors),axis=0))


    """
    get score of current configuration just updated in step(.)
    """
    def score(self, vec):
        exccmd('rm '+configFile )
        ## generate config file for CSmith
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


        crash, intra_diversity, avg_fv  = self.execute_config(N_PROGRAMS)

        vec.append(avg_fv)
        print("inter dis")
        print(vec)
        inter_diversity = self.manhattan_distance(avg_fv, vec) # distance between this config and previous ones

        predict_crash = random.randint(0,4) # model.predict(avg_fv)

        return crash, predict_crash/4, intra_diversity, inter_diversity

    def constrainCheck(self):
        def csmithFormat2proportionValue(orig_value, exploration_rate):
            orig_value_sorted = copy.deepcopy(orig_value)
            orig_value_sorted.sort()
            proportion_values = []
            proportion_add_info = []
            for v in orig_value:
                index = orig_value_sorted.index(v)  # v is the (index+1)^th small item in its group
                # print('cur v is '+str(v))
                while (proportion_add_info.count(index) != 0):
                    # print("add index "+str(index))
                    index += 1
                proportion_add_info.append(index)
                proportion_v = orig_value_sorted[index]
                if index != 0:
                    proportion_v -= orig_value_sorted[index - 1]
                proportion_values.append(proportion_v)
            overflow = 0
            while (proportion_values.count(0) != 0):
                pos0 = proportion_values.index(0)
                proportion_values[pos0] = random.randint(1, exploration_rate)
                overflow += proportion_values[pos0]
            bias = 0
            while (overflow != 0):
                pos = (overflow + bias) % len(proportion_values)
                if proportion_values[pos] > 1:
                    proportion_values[pos] -= 1
                    overflow -= 1
                else:
                    bias += 1

            return proportion_values, proportion_add_info

        def proportionValue2csmithFormat(orig_value, proportion_add_info):
            csmith_values = []
            for index in range(len(orig_value)):
                csmith_v = orig_value[index]
                for i in range(proportion_add_info[index]):
                    add_pos = proportion_add_info.index(i)
                    csmith_v += orig_value[add_pos]
                csmith_values.append(csmith_v)

            return csmith_values

        print("checking Constrain")
        for group in group_keys:
            group_dic = self.base_itm_group[group]

            ## constrain4
            to100key = max(group_dic, key=group_dic.get)
            if self.base_itm_group[group][to100key] != 100.0:
                print("constrain4")
                self.base_itm_group[group][to100key] = 100.0
                print(self.base_itm_group[group])

            ## constrain3
            keys = group_items_keys[group]
            values = [int(group_dic[key]) for key in keys]
            valuesset = list(set(values))
            if len(valuesset) != len(values):  # dumplicates elem exist
                print("constrain3")
                proportion, addinfo = csmithFormat2proportionValue(values, 3)
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

        ## constrain2
        if self.base_itm_group['simple_types_prob']['void_prob'] != 0:
            print("constrain2")
            self.base_itm_group['simple_types_prob']['void_prob'] = 0
            print(self.base_itm_group['simple_types_prob'])


# if __name__ == '__main__':
    # avg = list(np.mean(np.array(fvs),axis=0))
    # # print(len(fvs))
    # # print(len(fvs[0]))
    # print(avg)
    # print(len(avg))
    # # env = Environment('test')
    # for fv in fvs:
    #     print(len(fv))
    #     if len(fv) == 0:
    #         fvs.remove(fv)
    # for fv in fvs:
    #     dd = env.manhattan_distance(fv, fvs)
    #     print(dd)
