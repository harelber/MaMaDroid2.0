import os
#run mamadroid2.0 full feature set,mamadroid and permission set
path="with_perms/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_fullset.py")
#run mamadroid1.0
path="only_mama/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_only_mama.py")
#run permission as the only features
path="only_perms/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_drebin_only_perms.py")
#run mb attacks against mamadroid2.0
path="mb_check_mamadroid/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_mb_check.py")
#run mb attacks against permission set
path="check_mb_only_perms/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_mb_check_only_perms.py")
#run feature importance on DT and RF
path="feat_importance/"
if not os.path.exists(path):
        os.makedirs(path)
os.system("python3 run_feat_impt.py")


