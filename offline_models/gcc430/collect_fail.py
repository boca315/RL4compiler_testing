import os
import datetime
import os.path
import subprocess as subp
import sys
import _thread
import time

def exccmd(cmd):
    p=os.popen(cmd,"r")
    rs=[]
    line=""
    while True:
         line=p.readline()
         if not line:
              break
         # print(line)
         rs.append(line)
    return rs

def exccmd2string(cmd):
    p=os.popen(cmd,"r")
    rs=[]
    line=""
    while True:
         line=p.readline()
         if not line:
              break
         rs.append(line)
    ss=''
    for item in rs:
        ss+=item
    # print(ss)
    return ss

def feature2vector(feature):
    feature_vector = []
    for index in range(len(feature) - 1):
        # print(index)
        # print(feature[index])
        # print(feature[index].split(" ")[-1])
        feature_vector.append(int(feature[index].split(" ")[-1]))
    # feature_vector = np.array(feature_vector)
    # fv_normed = feature_vector / feature_vector.max(axis=0)  # nomalization
    # return fv_normed.tolist()
    return feature_vector

def get_extra_features(prog_name):
    def get_val(string):
        res = string.split(': ')[-1]
        res = res.split('\n')[0]
        if type(eval(res)) == float:
            return float(res)
        return int(res)

    prog = exccmd('cat '+prog_name)
    begin_key = '/************************ statistics *************************\n'
    begin_index = prog.index(begin_key)
    feature_zone = prog[begin_index:]
    extra_features = {
        'XXX max struct depth':0, 'XXX max struct depth occurrence':[], # 0-
        'total union variables':0,
        'non-zero bitfields defined in structs':0, 'zero bitfields defined in structs':0,
        'const bitfields defined in structs':0,'volatile bitfields defined in structs':0,
        'structs with bitfields in the program':0, 'full-bitfields structs in the program':0,
        'times a bitfields struct\'s address is taken':0, 'times a bitfields struct on LHS':0,
        'times a bitfields struct on RHS':0, 'times a single bitfield on LHS':0,
        'times a single bitfield on RHS':0,
        'XXX max expression depth':0,'XXX max expression depth occurrence':[], # 1-
        'total number of pointers':0,'times a variable address is taken':0,
        'XXX times a pointer is dereferenced on RHS':0,'XXX times a pointer is dereferenced on RHS occurrence':[],# 1-
        'XXX times a pointer is dereferenced on LHS':0,'XXX times a pointer is dereferenced on LHS occurrence':[],# 1-
        'times a pointer is compared with null':0,'times a pointer is compared with address of another variable':0,
        'times a pointer is compared with another pointer':0,'times a pointer is qualified to be dereferenced':0,
        'XXX max dereference level':0, 'XXX max dereference level occurrence':[],# 0-
        'number of pointers point to pointers':0, 'number of pointers point to scalars':0,
        'number of pointers point to structs':0,'percent of pointers has null in alias set':0,
        'average alias set size':0,'times a non-volatile is read':0,'times a non-volatile is write':0,
        'times a volatile is read':0,
            'times read thru a pointer':0,
        'times a volatile is write':0,
            'times written thru a pointer':0,
        'times a volatile is available for access':0,'percentage of non-volatile access':0,
        'forward jumps':0,'backward jumps':0,
        'stmts':0,
        'XXX max block depth':0,'XXX max block depth occurrence':[], # 0-
        'percentage a fresh-made variable is used':0, 'percentage an existing variable is used':0

    }

    for key in extra_features.keys():
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


exccmd('export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH')
os.system('export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu')

# gcc='gcc'
# csmith_home = '../../../../csmith_recorder/csmith_record_2.3.0_t1/'
gcc='gcc-4.3'
csmith_home = '/csmith_recorder/csmith_record_2.3.0_t1/'
csmith= '/csmith_recorder/csmith'
library=csmith_home+'runtime'
passres='./pass'
crashres='./crash'
wrongcoderes='./wrongcode'

exccmd('mkdir '+passres)
exccmd('mkdir '+crashres)
exccmd('mkdir '+wrongcoderes)

res=open('status.txt','w')
datafile = open('features.csv','a')
res.write('START '+str(time.time())+' \n')

exccmd('rm trainprogram*')
for i in range(1,20000+1):
    res.flush()
    datafile.flush()
    start=time.time()
    program_feature = exccmd('timeout 60 '+csmith+' -o trainprogram'+str(i)+'.c')
    # get feature vector into file
    # feature_res = open('trainprogram'+str(i)+'_feature','w')
    # feature_res.write(program_feature)
    # feature_res.close()
    # exccmd('ls')
    prog_type = ''
    prog_fv = []

    for v in feature2vector(program_feature):
        prog_fv.append(v)
    for v in get_extra_features('trainprogram'+str(i)+'.c'):
        prog_fv.append(v)


    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',generation\n')
        prog_type = 'generation'
        prog_fv = []
        # i=i-1
        continue


    start=time.time()
    exccmd('timeout 60 '+gcc+' -O0 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_0')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        prog_type = 'compile'
        prog_fv = []
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O1 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_1')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        prog_type = 'compile'
        prog_fv = []
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O2 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_2')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        prog_type = 'compile'
        prog_fv = []
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O3 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_3')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        prog_type = 'compile'
        prog_fv = []
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -Os trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_s')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        prog_type = 'compile'
        prog_fv = []
        # i=i-1
        continue

    if not os.path.exists('./trainprogram'+str(i)+'_0') or not os.path.exists('./trainprogram'+str(i)+'_1') or not os.path.exists('./trainprogram'+str(i)+'_2') or not os.path.exists('./trainprogram'+str(i)+'_3') or not os.path.exists('./trainprogram'+str(i)+'_s'):

        exccmd('mkdir '+crashres+'/trainprogram'+str(i))
        exccmd('mv '+'trainprogram* '+crashres+'/trainprogram'+str(i))
        prog_type = 'crash'
        res.write(str(i)+',crash\n')

    else:
        start=time.time()
        exccmd('timeout 60 ./trainprogram'+str(i)+'_0 > trainprogram_out_0')
        end=time.time()
        if (end-start) >= 60:
            exccmd('rm trainprogram*')
            res.write(str(i)+',execute\n')
            prog_type = 'execute'
            prog_fv = []
            # i=i-1
            continue

        exccmd('timeout 60 ./trainprogram'+str(i)+'_1 > trainprogram_out_1')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_2 > trainprogram_out_2')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_3 > trainprogram_out_3')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_s > trainprogram_out_s')

        f=open('trainprogram_out_0')
        lines=f.readlines()
        f.close()

        if len(lines)==0:
            exccmd('mkdir '+wrongcoderes+'/trainprogram'+str(i))
            exccmd('mv '+'trainprogram* '+wrongcoderes+'/trainprogram'+str(i))
            prog_type = 'wrongcode'
            res.write(str(i)+',wrongcode\n')
        else:
            exccmd('diff trainprogram_out_0 trainprogram_out_1 > trainprogram_diff_1')
            exccmd('diff trainprogram_out_0 trainprogram_out_2 > trainprogram_diff_2')
            exccmd('diff trainprogram_out_0 trainprogram_out_3 > trainprogram_diff_3')
            exccmd('diff trainprogram_out_0 trainprogram_out_s > trainprogram_diff_s')

            f=open('trainprogram_diff_1')
            lines1=f.readlines()
            f.close()

            f=open('trainprogram_diff_2')
            lines2=f.readlines()
            f.close()

            f=open('trainprogram_diff_3')
            lines3=f.readlines()
            f.close()

            f=open('trainprogram_diff_s')
            lines4=f.readlines()
            f.close()

            if not (len(lines1)==0 and len(lines2)==0 and len(lines3)==0 and len(lines4)==0):
                exccmd('mkdir '+wrongcoderes+'/trainprogram'+str(i))
                exccmd('mv '+'trainprogram* '+wrongcoderes+'/trainprogram'+str(i))
                prog_type = 'wrongcode'
                res.write(str(i)+',wrongcode\n')
            else:
                exccmd('mv '+'trainprogram'+str(i)+'.c '+passres)
                prog_type = 'pass'
                # exccmd('mv '+'trainprogram'+str(i)+'_feature '+passres)

                exccmd('rm trainprogram*')
                res.write(str(i)+',correct\n')


    datafile.write(str(i)+','+prog_type+',')
    if len(prog_fv)==0:
        continue
    for v in prog_fv:
        datafile.write(str(v)+',')
    datafile.write('\n')


res.write('END '+str(time.time())+' \n')
res.close()
datafile.close()






