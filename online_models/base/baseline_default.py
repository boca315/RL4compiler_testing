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
         # rs.append(line.strip())
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

# exccmd('export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH')
# os.system('export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu')

# gcc='gcc'
# csmith_home = '../../../../csmith_recorder/csmith_record_2.3.0_t1/'
gcc='gcc-4.4'
csmith_home = '/csmith_recorder/csmith_record_2.3.0_t1/'
csmith= csmith_home + '../csmith'

library = csmith_home+'runtime'
passres = './pass'
crashres = './crash'
wrongcoderes = './wrongcode'

# exccmd('mkdir '+passres)
# exccmd('mkdir '+crashres)
# exccmd('mkdir '+wrongcoderes)

res=open('status_base.txt','w')

exccmd('rm trainprogram*')
t1 = time.time()
for i in range(1,200000+1):
    t2 = time.time()
    if t2 - t1 >= 43200.0:  # 12h
        break
    # res.flush()
    start=time.time()
    program_feature = exccmd2string('timeout 60 '+csmith+' -o trainprogram'+str(i)+'.c')
    # get feature vector into file
    feature_res = open('trainprogram'+str(i)+'_feature','w')
    feature_res.write(program_feature)
    feature_res.close()
    exccmd('ls')

    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',generation\n')
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O0 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_0')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O1 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_1')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O2 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_2')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -O3 trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_3')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        # i=i-1
        continue

    start=time.time()
    exccmd('timeout 60 '+gcc+' -Os trainprogram'+str(i)+'.c -I '+library+' -o trainprogram'+str(i)+'_s')
    end=time.time()
    if (end-start) >= 60:
        exccmd('rm trainprogram*')
        res.write(str(i)+',compile\n')
        # i=i-1
        continue

    if not os.path.exists('./trainprogram'+str(i)+'_0') or not os.path.exists('./trainprogram'+str(i)+'_1') or not os.path.exists('./trainprogram'+str(i)+'_2') or not os.path.exists('./trainprogram'+str(i)+'_3') or not os.path.exists('./trainprogram'+str(i)+'_s'):

        # exccmd('mkdir '+crashres+'/trainprogram'+str(i))
        # exccmd('mv '+'trainprogram* '+crashres+'/trainprogram'+str(i))
        exccmd('rm '+'trainprogram* ')

        res.write(str(i)+',crash\n')

    else:
        start=time.time()
        exccmd('timeout 60 ./trainprogram'+str(i)+'_0 > trainprogram_out_0')
        end=time.time()
        if (end-start) >= 60:
            exccmd('rm trainprogram*')
            res.write(str(i)+',execute\n')
            # i=i-1
            continue

        exccmd('timeout 60 ./trainprogram'+str(i)+'_1 > trainprogram_out_1')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_2 > trainprogram_out_2')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_3 > trainprogram_out_3')
        exccmd('timeout 60 ./trainprogram'+str(i)+'_s > trainprogram_out_s')

        f=open('trainprogram_out_0')
        lines=f.readlines()
        f.close()

        f = open('trainprogram_out_1')
        lines1 = f.readlines()
        f.close()

        f = open('trainprogram_out_2')
        lines2 = f.readlines()
        f.close()

        f = open('trainprogram_out_3')
        lines3 = f.readlines()
        f.close()

        f = open('trainprogram_out_s')
        liness = f.readlines()
        f.close()

        if len(lines)==0 or len(lines1)==0 or len(lines2)==0 or len(lines3)==0 or len(liness)==0 :
            # exccmd('mkdir '+wrongcoderes+'/trainprogram'+str(i))
            # exccmd('mv '+'trainprogram* '+wrongcoderes+'/trainprogram'+str(i))
            exccmd('rm '+'trainprogram* ')

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
                # exccmd('mkdir '+wrongcoderes+'/trainprogram'+str(i))
                # exccmd('mv '+'trainprogram* '+wrongcoderes+'/trainprogram'+str(i))
                exccmd('rm '+'trainprogram* ')

                res.write(str(i)+',wrongcode\n')
            else:
                # exccmd('mv '+'trainprogram'+str(i)+'.c '+passres)
                # exccmd('mv '+'trainprogram'+str(i)+'_feature '+passres)
                exccmd('rm ' + 'trainprogram' + str(i) + '.c ')
                exccmd('rm ' + 'trainprogram' + str(i) + '_feature ')

                exccmd('rm trainprogram*')
                res.write(str(i)+',correct\n')

res.close()


