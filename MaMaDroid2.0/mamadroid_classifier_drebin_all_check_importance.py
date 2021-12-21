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
        data_former[" 'obfuscatedToobfuscated']"]=pd.to_numeric(data_former[" 'obfuscatedToobfuscated']"].replace(']','', regex=True))
        #cut the memory for floats
        floats = data_former.select_dtypes(include=['float64']).columns.tolist()
        data_former[floats] = data_former[floats].astype('float32')
        #get the name of files
        fam=fam[0]
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
        X_train.astype('float')
        y_train=train["type"]
        y_train=y_train.astype('int')
        #get permutation of the test data former
        X_test.astype('float')
        y_test=test["type"]
        y_test=y_test.astype('int')
        return X_train,y_train,X_test,y_test

#get the results of the prediction
def get_results(df_ben_train,df_mal_train,df_mal_test,fam,partial):
        #get model data and 
        X_train,y_train,test_former_X,y_test_former=get_train_test_correct(df_ben_train,df_mal_train,df_mal_test[0],fam)
        _,_,test_old_X,y_test1=get_train_test_correct(df_ben_train,df_mal_train,df_mal_test[1],fam)
        _,_,test_java_X,y_test2=get_train_test_correct(df_ben_train,df_mal_train,df_mal_test[2],fam)
        _,_,test_xml_X,y_test3=get_train_test_correct(df_ben_train,df_mal_train,df_mal_test[3],fam)
      
        #running classifiers
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = KNeighborsClassifier(n_neighbors=3)
        clf3 = DecisionTreeClassifier(random_state=0)
        clf4=RandomForestClassifier(n_estimators=random_forest_param[features]["trees"],max_depth=random_forest_param[features]["max_depth"])
        '''
        clf4=AdaBoostClassifier(n_estimators=100, random_state=0)
        estimators = [('ada', clf4),('1nn',clf1)]
        stack1 = StackingClassifier(estimators=estimators, final_estimator=clf3)
        estimators = [('dt', clf3),('ada', clf4)]
        stack2 = StackingClassifier(estimators=estimators, final_estimator=clf1)
        estimators = [('3nn', clf2)]
        stack3 = StackingClassifier(estimators=estimators, final_estimator=clf1)
        #vote1 = VotingClassifier(estimators=[('dt', clf3), ('s1', stack1), ('s2', stack2)],voting='soft')
        vote2 = VotingClassifier(estimators=[('s1', stack1), ('s2', stack2), ('s3', stack3)],voting='soft')
        
        clf1=stack1
        clf2=stack2
        clf3=vote2
        '''
        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
        clf3.fit(X_train, y_train)
        clf4.fit(X_train, y_train)
        #fs=clf4.feature_names_in
        imps3=clf3.feature_importances_
        imps4=clf4.feature_importances_
        
        clfs=[clf1,clf2,clf3,clf4]
        res_file=open(sys.argv[1]+os.path.sep+str(partial)+"_other_algs.txt",'w')
        original_stdout = sys.stdout
        fs=X_train.columns
        sys.stdout = res_file
        d_imp3={}
        d_imp4={}
        for i in range(len(imps3)):
                d_imp3[imps3[i]]=fs[i]
                d_imp4[imps4[i]]=fs[i]
        print("\n---dt features---\n")
        for key in sorted(d_imp3):
                print "%s: %s" % (key, d_imp3[key])
        print("\n---rf features---\n")
        for key in sorted(d_imp4):
                print "%s: %s" % (key, d_imp4[key])        
        exit(0)
        print("done")
        pred_former_clfs("former",clfs,test_former_X,partial,y_test_former)
        pred_former_clfs("old",clfs,test_old_X,partial,y_test1)
        pred_former_clfs("java",clfs,test_java_X,partial,y_test2)
        pred_former_clfs("xml",clfs,test_xml_X,partial,y_test3)
        sys.stdout = original_stdout
        df_x = X_train.assign(label = y_train)
        
        #set_models(df_x,130,None,[test_former_X,test_old_X,test_java_X,test_xml_X],res_file,[y_test_former,y_test1,y_test2,y_test3],partial)


def get_names_and_chosen(test_new):
        #get attack data
        with open(test_new,'r') as r:
                con=r.readlines()
        r.close()
        tests_data={}
        titles=con[0].strip("\r\n").split(",")
        for line in con[1:]:
                splits=line.strip("\r\n").replace("]","").split(",")
                if "_" not in splits[0]:#former attacks
                        name=splits[0][2:].split(".")[0][:-6]+".txt"
                else:
                        name=splits[0][2:].split("_")[0]+".txt"
                
                if name in tests_data.keys():
                        tests_data[name].append(splits)
                else:
                        tests_data[name]=[splits]
        return tests_data
        
def merge_two_dfs(df1,df2):#take two dfs, return them merged by the file name ([0])
        df1=df1.sort_values(by=df1.columns[0])
        df2[df2.columns[0]] = df2[df2.columns[0]].replace({"'":""}, regex=True)
        df2=df2.sort_values(by=df2.columns[0])
        #merge the two datasets
       
        merge=pd.merge(df2, df1, on=df2.columns[0])
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

#mal data
df_mal_train=pd.read_csv(train_mal)
df_mal_train=df_mal_train.sample(frac=float(sys.argv[2]))
df_mal_train["type"]=1
df_mal_train["category"]=0
fam=df_mal_train.columns
df_mal_train[fam[0]]=df_mal_train[fam[0]].str.replace('[','')
df_mal_train[fam[0]]=df_mal_train[fam[0]].str.replace("'","")
df_mal_train[fam[0]]=df_mal_train[fam[0]].str.replace("\n","")
test_amount=int(0.2*(df_mal_train.shape[0]))



#create the attacks pds
dict_xml=get_names_and_chosen(test_new)
dict_java=get_names_and_chosen(test_new2)
dict_old=get_names_and_chosen(test_new3)

keys=[dict_old.keys(),dict_java.keys(),dict_xml.keys()]
#keys=[dict_java.keys(),dict_xml.keys()]

keys=list(set.intersection(*[set(x) for x in keys]))

chosen_old=[]
chosen_xml=[]
chosen_java=[]

n_items = random.sample(keys, test_amount)
for name in n_items:
        chosen_old.append(random.sample(dict_old[name],1)[0])
        chosen_xml.append(random.sample(dict_xml[name],1)[0])
        chosen_java.append(random.sample(dict_java[name],1)[0])


df_mal_test_old = pd.DataFrame.from_records(chosen_old, columns =fam[:-2])
df_mal_test_old["type"]=1
df_mal_test_old["category"]=1

df_mal_test_java = pd.DataFrame.from_records(chosen_java, columns =fam[:-2])
df_mal_test_java["type"]=1
df_mal_test_java["category"]=1

df_mal_test_xml = pd.DataFrame.from_records(chosen_xml, columns =fam[:-2])
df_mal_test_xml["type"]=1
df_mal_test_xml["category"]=1


#create the former apps pd
names=n_items
df_mal_test1 = pd.DataFrame(columns = fam)
for name in names:
        try:
                line=df_mal_train.loc[df_mal_train[fam[0]] == name]
                df_mal_test1 = pd.concat([df_mal_test1,line])
                df_mal_train.drop(df_mal_train.loc[df_mal_train[fam[0]]==name].index, inplace=True)

        except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                
                pass


df_mal_test1=df_mal_test1.sample(frac=float(sys.argv[3]))
df_mal_test1["type"]=1
df_mal_test1["category"]=1


df_ben_train=pd.read_csv(benign_train)
#df_ben_train=df_ben_train.sample(frac=float(sys.argv[1]))
df_ben_train["type"]=0
df_ben_train["category"]=0

col_names=df_ben_train.columns
name_col=col_names[0]
f = open("ben_apps_split_"+sys.argv[1].split(os.path.sep)[1]+'.txt','r')
apps_names=json.load(f)

   
drebin_pd_ben=pd.read_csv(dreb_ben)#load benign drebin small feature set
drebin_pd_mal=pd.read_csv(dreb_mal)#load malicious drebin small feature set
drebin_pd_mal[drebin_pd_mal.columns[0]] = drebin_pd_mal[drebin_pd_mal.columns[0]].replace({'\[':''}, regex=True)
drebin_pd_ben[drebin_pd_ben.columns[0]] = drebin_pd_ben[drebin_pd_ben.columns[0]].replace({'\[':''}, regex=True)
df_mal_test_old[df_mal_test_old.columns[0]] =df_mal_test_old[df_mal_test_old.columns[0]].apply(lambda x:x[1:].split(".")[0][:-6]+".txt")
df_mal_test_java[df_mal_test_java.columns[0]] =df_mal_test_java[df_mal_test_java.columns[0]].apply(lambda x:x[2:].split("_")[0]+".txt")
df_mal_test_xml[df_mal_test_xml.columns[0]] =df_mal_test_xml[df_mal_test_xml.columns[0]].apply(lambda x:x[2:].split("_")[0]+".txt")
df_ben_train[df_ben_train.columns[0]] =df_ben_train[df_ben_train.columns[0]].apply(lambda x:x[2:-1])

df_mal_train=merge_two_dfs(drebin_pd_mal,df_mal_train)
df_mal_test1=merge_two_dfs(drebin_pd_mal,df_mal_test1)
df_mal_test_old=merge_two_dfs(drebin_pd_mal,df_mal_test_old)

df_mal_test_java=merge_two_dfs(drebin_pd_mal,df_mal_test_java)
df_mal_test_xml=merge_two_dfs(drebin_pd_mal,df_mal_test_xml)
df_ben_train=merge_two_dfs(drebin_pd_ben,df_ben_train)

ben_train,ben_test= train_test_split(df_ben_train, test_size=0.2)
ben_train["type"]=0
ben_train["category"]=0
ben_test["type"]=0
ben_test["category"]=1
df_mal_test1=merge_two_dfs_ben_mal(ben_test,df_mal_test1)

df_mal_test_old=merge_two_dfs_ben_mal(ben_test,df_mal_test_old)

df_mal_test_java=merge_two_dfs_ben_mal(ben_test,df_mal_test_java)
df_mal_test_xml=merge_two_dfs_ben_mal(ben_test,df_mal_test_xml)
df_ben_train=ben_train

get_results(df_ben_train,df_mal_train,[df_mal_test1,df_mal_test_old,df_mal_test_java,df_mal_test_xml],fam,10/10)
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
