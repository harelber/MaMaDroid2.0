import os
import pathlib

for i in range(0,5):
        itera="feat_importance/"+str(i)
        pathlib.Path(itera).mkdir(exist_ok=True) 
        os.system("python mamadroid_classifier_drebin_all_check_importance.py "+itera+" 1 1 1")

