#encoding: utf-8
import sys, os, time,gc

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

def exccmd(cmd):
    return os.system(cmd)
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
def vals2dic(vals):
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

configFile = 'config' #+str(time.time)
gcc='gcc-4.4'
csmith_home = '/csmith_recorder/csmith_record_2.3.0_t1/'
csmith= csmith_home + '../csmith'
library = csmith_home+'runtime'

'''
sys.argv[1] check: crash 
sys.argv[2] seed: 4015900742
sys.argv[3] config vals: 53, 51, 50, 10, 13, 10, 53, 83, 54, 10, 50, 54, 34, 23, 5, 50, 53, 33, 29, 43, 24, 24, 54, 53, 100, 0, 30, 16, 39, 43, 45, 50, 60, 25, 28, 76, 100, 6.55555534362793, 12.11111068725586, 19.66666603088379, 26.22222137451172, 28.77777862548828, 33.33333206176758, 42.88888931274414, 47.44444274902344, 54.0, 55.55555725097656, 64.11111450195312, 70.66666412353516, 76.22222137451172, 80.77777862548828, 87.33333587646484, 91.88888549804688, 98.44444274902344, 100.0, 0, 10, 20, 30, 40, 90, 60, 74, 80, 50, 100, 1, 28, 51, 78, 100
'''

config_vals = [v.replace(',','') for v in sys.argv[3:]]
# print(config_vals)
base_itm_single, base_itm_group = vals2dic(config_vals)

## generate config file for CSmith ##
fp_w = open(configFile, 'w')
for key in single_keys:
    fp_w.write(key + '=' + str(base_itm_single[key]) + '\n')
    fp_w.write('\n')
for group in group_keys:
    fp_w.write('[' + group + ',')
    group_txt = ''
    group_dir = base_itm_group[group]
    for key in group_items_keys[group]:
        group_txt += key + '=' + str(group_dir[key]) + ','
    fp_w.write(group_txt[:-1] + ']\n')
    fp_w.write('\n')
fp_w.flush()
fp_w.close()

seed=sys.argv[2]
progname = 'program'+seed
generation = 'timeout 120 ' + csmith + ' --probability-configuration ' + configFile + ' --seed '+seed+' -o '+progname+'.c'

crash_bi0 = 'timeout 60 ' + gcc + ' -O0 ' + progname + '.c -I ' + library + ' -o ' + progname + '_0'
crash_bi1 = 'timeout 60 ' + gcc + ' -O1 ' + progname + '.c -I ' + library + ' -o ' + progname + '_1'
crash_bi2 = 'timeout 60 ' + gcc + ' -O2 ' + progname + '.c -I ' + library + ' -o ' + progname + '_2'
crash_bi3 = 'timeout 60 ' + gcc + ' -O3 ' + progname + '.c -I ' + library + ' -o ' + progname + '_3'
crash_bis = 'timeout 60 ' + gcc + ' -Os ' + progname + '.c -I ' + library + ' -o ' + progname + '_s'

wrongcode_checksum0 = 'timeout 60 ./'+progname + '_0  > '+progname + '_out_0'
wrongcode_checksum1 = 'timeout 60 ./'+progname + '_1  > '+progname + '_out_1'
wrongcode_checksum2 = 'timeout 60 ./'+progname + '_2  > '+progname + '_out_2'
wrongcode_checksum3 = 'timeout 60 ./'+progname + '_3  > '+progname + '_out_3'
wrongcode_checksums = 'timeout 60 ./'+progname + '_s  > '+progname + '_out_s'


wrongcode_diff1 = 'diff ' +progname+ '_out_0 ' + progname + '_out_1 > ' + progname + '_diff_1'
wrongcode_diff2 = 'diff ' +progname+ '_out_0 ' + progname + '_out_2 > ' + progname + '_diff_2'
wrongcode_diff3 = 'diff ' +progname+ '_out_0 ' + progname + '_out_3 > ' + progname + '_diff_3'
wrongcode_diffs = 'diff ' +progname+ '_out_0 ' + progname + '_out_s > ' + progname + '_diff_s'


exccmd(generation)
print(generation)
if sys.argv[1] == 'crash':
    exccmd(crash_bi0)
    print(crash_bi0)
    exccmd(crash_bi1)
    print(crash_bi1)
    exccmd(crash_bi2)
    print(crash_bi2)
    exccmd(crash_bi3)
    print(crash_bi3)
    exccmd(crash_bis)
    print(crash_bis)
    files = exccmd2arr('ls | grep '+progname)
    print(files)

if sys.argv[1] == 'wrongcode':
    exccmd(crash_bi0)
    print(crash_bi0)
    exccmd(crash_bi1)
    print(crash_bi1)
    exccmd(crash_bi2)
    print(crash_bi2)
    exccmd(crash_bi3)
    print(crash_bi3)
    exccmd(crash_bis)
    print(crash_bis)


    exccmd(wrongcode_checksum0)
    print(wrongcode_checksum0)
    exccmd(wrongcode_checksum1)
    print(wrongcode_checksum1)
    exccmd(wrongcode_checksum2)
    print(wrongcode_checksum2)
    exccmd(wrongcode_checksum3)
    print(wrongcode_checksum3)
    exccmd(wrongcode_checksums)
    print(wrongcode_checksums)

    f = open(progname + '_out_0')
    lines0= f.readlines()
    f.close()
    f = open(progname + '_out_1')
    lines1 = f.readlines()
    f.close()
    f = open(progname + '_out_2')
    lines2 = f.readlines()
    f.close()
    f = open(progname + '_out_3')
    lines3 = f.readlines()
    f.close()
    f = open(progname + '_out_s')
    lines4 = f.readlines()
    f.close()
    files = exccmd2arr('ls | grep ' + progname)


    exccmd(wrongcode_diff1)
    print(wrongcode_diff1)
    exccmd(wrongcode_diff2)
    print(wrongcode_diff2)
    exccmd(wrongcode_diff3)
    print(wrongcode_diff3)
    exccmd(wrongcode_diffs)
    print(wrongcode_diffs)
    f = open(progname + '_diff_1')
    lines11 = f.readlines()
    f.close()
    f = open(progname + '_diff_2')
    lines12 = f.readlines()
    f.close()
    f = open(progname + '_diff_3')
    lines13 = f.readlines()
    f.close()
    f = open(progname + '_diff_s')
    lines14 = f.readlines()
    f.close()

    print(files)
    print(len(lines0),',',len(lines1), ',', len(lines2), ',', len(lines3), ',', len(lines4))
    print(lines11,',',lines12,',',lines13,',',lines14)

