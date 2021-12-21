import os
import pathlib

for i in range(0,5):
        itera="check_mb_only_perms/"+str(i)
        pathlib.Path(itera).mkdir(exist_ok=True) 
        os.system("python mamadroid_classifier_drebin_all_check_mbs_only_perms.py "+itera+" 1 1 1")
