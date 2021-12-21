from __future__ import division
import csv
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,confusion_matrix
import numpy as np
import random
import math
import sys,os
import time
from random import sample 
from sklearn import preprocessing, model_selection
#from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier#,StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
#from pycaret.classification import *
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd
import json
from sklearn.ensemble import AdaBoostClassifier


#setup the models
def set_models(df_set,s_id,numeric_,test_dataframe,res_file,model_labels_test,partial):
        original_stdout = sys.stdout
        
        model_setup = setup(data=df_set,target='label',session_id=s_id,html=False,silent=True,verbose=False)
        
        d=models()
        alg_names=[]
        for i in range(d.shape[0]):
                if d.iloc[i].name=="gpc":
                        continue
                alg_names.append(d.iloc[i].name)
        
        results = alg_names
        params = {"penalty": ['l1', 'l2', 'elasticnet', 'none'],"solver":['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}  
        optim="Recall"
        #add the loop of betas here and the optimums
        
        for i in range(0,len(results)):
                try:
                     if results[i]=='lr':
                        classifier_process(results[i],test_dataframe,res_file,model_labels_test,optim,params,partial)
                     else:
                        classifier_process(results[i],test_dataframe,res_file,model_labels_test,optim,0,partial)
                except Exception as e:
                        print(e)
                        sys.stdout = original_stdout
                        continue
#run the classifier
def classifier_process(alg_name,test_dataframe,res_file,truth,optim,params,partial):
        cl=create_model(alg_name)
        #if alg_name=='lr':
                #cl=create_model(alg_name)
                #cl=tune_model(cl,optimize = optim,custom_grid = params)
        #else:
                #cl=create_model(alg_name)
                #cl=tune_model(cl,optimize = optim)
        cl=finalize_model(cl)
        names=["former","old","java","xml"]
        tn=open(sys.argv[1]+os.path.sep+"tuned.txt","a")
        tn.write(str(partial)+","+alg_name+","+str(cl)+"\n")
        tn.close()
        for i in range(4):
                test_predictions = predict_model(cl, data=test_dataframe[i])
                test_labels=test_predictions.iloc[:, -2:-1]
                original_stdout = sys.stdout
                sys.stdout = res_file
                con=test_labels.to_dict(orient='list')
                predicts=con['Label']
                y_actu = pd.Series(truth[i], name='Actual')
                y_pred = pd.Series(predicts, name='Predicted')
                confusion_matrix = pd.crosstab(y_actu, y_pred)
                ac,prec,rec,f1=get_metrics(truth[i],predicts)
                print(names[i]+","+str(partial)+","+\
                alg_name+","+\
                str(ac)+","+str(prec)+","+str(rec)+","+str(f1))
                print(confusion_matrix)
                sys.stdout = original_stdout

            
#predict of 2 KNNs and RF
def pred_former_clfs(name,clfs,X_test,partial,truth):
        #predict the attacks
        y_pred1=clfs[0].predict(X_test)
        y_pred2=clfs[1].predict(X_test)
        y_pred3=clfs[2].predict(X_test)
        y_pred4=clfs[3].predict(X_test)
        
        ac0,prec0,rec0,f1_0=get_metrics(truth,y_pred1)
        ac1,prec1,rec1,f1_1=get_metrics(truth,y_pred2)
        ac2,prec2,rec2,f1_2=get_metrics(truth,y_pred3)
        ac3,prec3,rec3,f1_3=get_metrics(truth,y_pred4)
        
        tn0,fp0,fn0,tp0=confusion_matrix(truth,y_pred1,labels=[0,1]).ravel()
        tn1,fp1,fn1,tp1=confusion_matrix(truth,y_pred2,labels=[0,1]).ravel()
        tn2,fp2,fn2,tp2=confusion_matrix(truth,y_pred3,labels=[0,1]).ravel()
        tn3,fp3,fn3,tp3=confusion_matrix(truth,y_pred4,labels=[0,1]).ravel()

        acs=[ac0,ac1,ac2,ac3]
        precs=[prec0,prec1,prec2,prec3]
        recs=[rec0,rec1,rec2,rec3]
        f1s=[f1_0,f1_1,f1_2,f1_3]
        tfs1=[tn0,fp0,fn0,tp0]
        tfs2=[tn1,fp1,fn1,tp1]
        tfs3=[tn2,fp2,fn2,tp2]
        tfs4=[tn3,fp3,fn3,tp3]
        tfs=[tfs1,tfs2,tfs3,tfs4]
        print("name,partial,accuracy,precision,recall,f1,tn,fp,fn,tp")
        for i in range(4):#print the metrics
            print(name+","+str(partial)+","+\
            str(acs[i])+","+str(precs[i])+","+\
            str(recs[i])+","+str(f1s[i])+","+str(tfs[i][0])+","+str(tfs[i][1])+","+str(tfs[i][2])+","+str(tfs[i][3]))

#get all metrics results
def get_metrics(truth,predicts):
    ac=accuracy_score(truth, predicts)
    prec=precision_score(truth, predicts)
    rec=recall_score(truth, predicts)
    f1=f1_score(truth, predicts, average='macro')
    return ac,prec,rec,f1
#correct for of train and test
def get_train_test_correct(df_ben_train,df_mal_train,df_mal_test,fam):
        data_former= pd.concat([df_ben_train,df_mal_train,df_mal_test],sort=False)
        data_former=data_former.fillna(0)
        data_former = data_former.sample(frac=1).reset_index(drop=True)#shuffle
        
        #cut the memory for floats
        floats = data_former.select_dtypes(include=['float64']).columns.tolist()
        data_former[floats] = data_former[floats].astype('float32')
        #get the name of files
        
        #getting X and y
        train=data_former[(data_former.category==0)]
        train=train.reset_index()
        #train=train.reset_index()
        test=data_former[(data_former.category==1)]
        test=test.reset_index()
        
        X_train=train.drop(columns=[fam,"type","category"])
        X_test=test.drop(columns=[fam,"type","category"])
        X_train = X_train[(X_train.T != 0).any()]
        X_test = X_test[(X_test.T != 0).any()]
        #get permutation of the train data
        X_train = X_train[X_train.columns].replace({'\[':'','\]':''}, regex=True)
        X_train.astype('float')
        y_train=train["type"]
        y_train=y_train.astype('int')
        #get permutation of the test data former
        X_test.astype('float')
        y_test=test["type"]
        y_test=y_test.astype('int')
        return X_train,y_train,X_test,y_test

#get the results of the prediction
def get_results(df_ben_train,df_mal_train,df_mal_test,fam,partial,name):
        #get model data and 
        X_train,y_train,test_former_X,y_test_former=get_train_test_correct(df_ben_train,df_mal_train,df_mal_test,fam)
      
        #running classifiers
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = KNeighborsClassifier(n_neighbors=3)
        clf3 = DecisionTreeClassifier(random_state=0)
        clf4=RandomForestClassifier(n_estimators=random_forest_param[features]["trees"],max_depth=random_forest_param[features]["max_depth"])
        
        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
        clf3.fit(X_train, y_train)
        clf4.fit(X_train, y_train)
        
        
        clfs=[clf1,clf2,clf3,clf4]
        res_file=open(sys.argv[1]+os.path.sep+str(partial)+"_other_algs.txt",'a+')
        original_stdout = sys.stdout
        sys.stdout = res_file
        
        pred_former_clfs(name,clfs,test_former_X,partial,y_test_former)
        
        sys.stdout = original_stdout
        df_x = X_train.assign(label = y_train)
        
        #set_models(df_x,130,None,[test_former_X,test_old_X,test_java_X,test_xml_X],res_file,[y_test_former,y_test1,y_test2,y_test3],partial)

        
def merge_two_dfs(df1,df2):#take two dfs, return them merged by the file name ([0])
        #df2[df2.columns[0]] = df2[df2.columns[0]].replace({"'":""}, regex=True)
        #merge the two datasets
        merge= df1.append(df2, sort=False)
        
        for i in range(len(merge.columns)):
                try:
                        merge[merge.columns[i]] = merge[merge.columns[i]].replace({'\]':''}, regex=True)
                except Exception as e:
                        pass
                try:
                        merge[merge.columns[i]] = merge[merge.columns[i]].replace({'\[':''}, regex=True)
                except Exception as e:
                        pass
        
        return merge.fillna(0)

def merge_two_dfs_ben_mal(df1,df2):#take two dfs, return them merged by the file name ([0])
        #df1=df1.sort_values(by=df1.columns[0])
        df2[df2.columns[0]] = df2[df2.columns[0]].replace({"'":""}, regex=True)
        #df2=df2.sort_values(by=df2.columns[0])
        #merge the two datasets
        
        merge=pd.concat([df2, df1]).reset_index(drop=True)
        
        for i in range(len(merge.columns)):
                try:
                        merge[merge.columns[i]] = merge[merge.columns[i]].replace({'\]':''}, regex=True)
                except Exception as e:
                        pass
                try:
                        merge[merge.columns[i]] = merge[merge.columns[i]].replace({'\[':''}, regex=True)
                except Exception as e:
                        pass
        
        return merge

#classifier parameters
random_forest_param={"family":{"trees":51,"max_depth":8},"package":{"trees":101,"max_depth":64}}

feats=[" 'selfdefinedTodalvik.'"," 'selfdefinedToselfdefined'"]         

#benign_train,benign_test,train_mal,test_mal,features,report_file,data_file,framework_dir,first_row,group,state=sys.argv[1:]

#getting the csv paths
features="family"


train_mal="mam_feat/DrebinApps.csv"
benign_train="mam_feat/benign_all.csv"
#benign_train="benign_actual_features.csv"
#train_mal="DrebinApps.csv"
#train_mal=features_folder+"/DrebinApps.csv"

#test_mal=features_folder+"/former_apps.csv"
#test_mal=features_folder+"/mama_attacks3.csv"
test_new="mam_feat/black.csv"
test_new2="mam_feat/full.csv"
test_new3="mam_feat/random.csv"

dreb_ben="drebin_ben_one_file_perms.txt"
dreb_mal="drebin_mal_one_file_perms.txt"
dreb_mal1="mb2_feats_mama2.txt"
dreb_mal2="mb3_feats_mama2.txt"
dreb_mal3="mb4_feats_mama2.txt"

#mal data

   
drebin_pd_ben=pd.read_csv(dreb_ben)#load benign drebin small feature set
drebin_pd_mal=pd.read_csv(dreb_mal)#load benign drebin small feature set

drebin_pd_mal1=pd.read_csv(dreb_mal1)#load malicious drebin small feature set
drebin_pd_mal2=pd.read_csv(dreb_mal2)#load malicious drebin small feature set
drebin_pd_mal3=pd.read_csv(dreb_mal3)#load malicious drebin small feature set
drebin_pd_mal1[drebin_pd_mal1.columns[0]] = drebin_pd_mal1[drebin_pd_mal1.columns[0]].replace({'\[':''}, regex=True)
drebin_pd_mal2[drebin_pd_mal2.columns[0]] = drebin_pd_mal2[drebin_pd_mal2.columns[0]].replace({'\[':''}, regex=True)
drebin_pd_mal3[drebin_pd_mal3.columns[0]] = drebin_pd_mal3[drebin_pd_mal3.columns[0]].replace({'\[':''}, regex=True)
drebin_pd_ben[drebin_pd_ben.columns[0]] = drebin_pd_ben[drebin_pd_ben.columns[0]].replace({'\[':''}, regex=True)
drebin_pd_mal[drebin_pd_mal.columns[0]] = drebin_pd_mal[drebin_pd_mal.columns[0]].replace({'\[':''}, regex=True)
#update names of mb2 attack
drebin_pd_mal2[drebin_pd_mal2.columns[0]] =drebin_pd_mal2[drebin_pd_mal2.columns[0]].apply(lambda x:x[2:])
ben_train,ben_test= train_test_split(drebin_pd_ben, test_size=0.2)

drebin_pd_mal1["type"]=1
drebin_pd_mal1["category"]=1
drebin_pd_mal2["type"]=1
drebin_pd_mal2["category"]=1
drebin_pd_mal3["type"]=1
drebin_pd_mal3["category"]=1
drebin_pd_mal["type"]=1
drebin_pd_mal["category"]=0
ben_test["type"]=0
ben_test["category"]=1
ben_train["type"]=0
ben_train["category"]=0




#assemble the data for detection
df_mal_test1=drebin_pd_mal1#merge_two_dfs(drebin_pd_mal,drebin_pd_mal1)#add features to test from drebin

df_mal_test2=drebin_pd_mal2#merge_two_dfs(drebin_pd_mal,drebin_pd_mal2)#add features to test from drebin
df_mal_test3=drebin_pd_mal3#merge_two_dfs(drebin_pd_mal,drebin_pd_mal3)#add features to test from drebin
df_mal_test1["category"]=1
df_mal_test2["category"]=1
df_mal_test3["category"]=1

df_mal_test1=merge_two_dfs_ben_mal(ben_test,df_mal_test1)#add benign to the test mix
df_mal_test2=merge_two_dfs_ben_mal(ben_test,df_mal_test2)#add benign to the test mix
df_mal_test3=merge_two_dfs_ben_mal(ben_test,df_mal_test3)#add benign to the test mix
df_ben_train=ben_train#update benign train



mal_train_former,mal_test_former= train_test_split(drebin_pd_mal, test_size=0.2)
mal_test_former["category"]=1
fam=mal_train_former.columns[0]

mal_test_former=merge_two_dfs_ben_mal(ben_test,mal_test_former)#add benign to the test mix
get_results(df_ben_train,mal_train_former,mal_test_former,fam,10/10,"former")
others=[df_mal_test1,df_mal_test2,df_mal_test3]

df_mal_train=drebin_pd_mal
i=1
for attack in others:#iterate of the other attacks
        df_test=attack

        df_test_samples_names=df_test[df_test.columns[0]]
        df_mal_train_current= df_mal_train[~df_mal_train[df_mal_train.columns[0]].isin(df_test_samples_names)]
        get_results(df_ben_train,df_mal_train_current,df_test,fam,10/10,"mb_"+str(i))
        i+=1
exit(0)


get_results(df_ben_train,df_mal_train,df_test,fam,10/10,name)
exit(0)
added_bens=[]
for i in range(10):
    cur=apps_names[str(i)]
    added_bens.append(df_ben_train.loc[df_ben_train[name_col].isin(cur)])

df_ben_train = pd.DataFrame(columns=df_ben_train.columns)

for i in range(0,10):
        df_ben_train=pd.concat([df_ben_train, added_bens[i]], sort=False)
        
        #test the former apps
        get_results(df_ben_train,df_mal_train,[df_mal_test1,df_mal_test_old,df_mal_test_java,df_mal_test_xml],fam,(i+1)/10)
