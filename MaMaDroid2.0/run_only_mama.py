import os
import pathlib

for i in range(0,5):
        itera="only_mama/"+str(i)
        pathlib.Path(itera).mkdir(exist_ok=True) 
        os.system("python mamadroid_classifier_no_drebin.py "+itera+" 1 1 1")

