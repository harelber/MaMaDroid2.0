import os
import sys
import time
from collections import deque,OrderedDict
import random
import math
import itertools
import shutil
import re
import pickle
import string
import csv

#get the full tree
def tree_files(root):
    tree=[]
    main=root#save the root dir for later
    for root, dirs, files in os.walk(root):#iterate the tree 
        for f in files:
            full_path=os.path.join(root,f).split(main)[1]
            tree.append(full_path)
    return tree


#get the dir tree with a dict
def tree_dir(root):
    tree_dict={}
    main=root#save the root dir for later
    for root, dirs, files in os.walk(root):#iterate the tree 
        for d in dirs:
            full_path=os.path.join(root,d).split(main)[1]
            count=full_path.count("/")-1
            if count in tree_dict:
                 tree_dict[count].append(full_path)
            else:
                 tree_dict[count]=[full_path]
    return tree_dict
#get the branch inside the dirs to manipulate
def branching2(app_tree, level,percent,root_dir):
        branch_dict={}
        #get the level of manipulation
        roots=app_tree[level]
        #shuffle dirs for randomness
        random.shuffle(roots)
        #get the number of dirs to manipulate
        chosen_amount=int(math.ceil(percent*len(roots)))
        #actual dirs
        chosen_roots=roots[:chosen_amount]                   
        return chosen_roots




#get the branch inside the dirs to manipulate
def branching(app_tree, level,percent,root_dir):
        branch_dict={}
        #get the level of manipulation
        roots=app_tree[level]
        #shuffle dirs for randomness
        random.shuffle(roots)
        #get the number of dirs to manipulate
        chosen_amount=int(math.ceil(percent*len(roots)))
        #actual dirs
        chosen_roots=roots[:chosen_amount]   
        for root in chosen_roots:
                full_path=root_dir+root
                #get the files of this subtree
                files=tree_dir(full_path).values()         
                files_merge=list(itertools.chain.from_iterable(files))
                #list_of_dirs=(string_set(files_merge))
                #branch_dict[root]=list_of_dirs
                branch_dict[root]=files_merge
                        
        branch_list=[]
        for key in branch_dict:
                for v in branch_dict[key]:
                        full_path=key+v
                        branch_list.append(full_path)
        return branch_list

#substring set function
def string_set(string_list):
    return set(i for i in string_list 
               if not any(i in s for s in string_list if i != s))

#soundness check function - to see if the actual change is relevant to the app
def soundness_check_and_replace(fname,dirs,root_full_path,delim,start_char,dummy,starting_point,flag,ending_index):
        #need to check the layout file and why it is not changing        
        try:
                with open(fname) as f:
                    s = f.read()
                f.close()
                soundness_test=set()#save only sound changes to the files
                #check of occurances to replace
                for d in dirs:
                        test=[m.start() for m in re.finditer(start_char+d[1:],s)]
                        for index in test:
                                workspace=s[index:]
                                workspace=workspace.split(delim)[ending_index]
                                try:
                                        workspace2=workspace.split('"')[ending_index]
                                        #bugfix for L not in a start of a class
                                        x=workspace.split("(")[0]
                                        y=workspace.split("/")[0]
                                        
                                        if len(workspace)>len(workspace2):
                                                workspace=workspace2
                                
                                except:
                                        pass
                                workspace=workspace[starting_point:]
                                
                                soundness_test.add(workspace) 
                soundness_check_pass=set() 
                for item in soundness_test:
                            
                            item_path=root_full_path+os.path.sep+item
                            if flag!="s":#soundness conversion for manifest
                                item_path=item_path.replace(".",os.path.sep)
                            if len(item.split("("))>1:
                                continue
                            if os.path.isdir(item_path) or os.path.isfile(item_path+".smali"):
                                soundness_check_pass.add(item)
                #print soundness_check_pass
                for item in soundness_check_pass:
                        
                        s=s.replace(start_char+item,start_char+dummy+item)
                        
                with open(fname,"w") as f:
                    f.write(s)
                f.close() 
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            #exit(0)
            
            
#soundness check function - to see if the actual change is relevant to the app
def soundness_check_and_replace_lay(fname,dirs,root_full_path,delim,start_char,dummy,starting_point,flag,ending_index):
        #need to check the layout file and why it is not changing        
        try:
                with open(fname) as f:
                    s = f.read()
                f.close()
                
                soundness_test=set()#save only sound changes to the files
                #check of occurances to replace
                for d in dirs:
                        test=[m.start() for m in re.finditer(start_char+d[1:],s)]
                        
                        for index in test:
                                workspace=s[index:]
                                workspace=workspace.split(delim)[ending_index]
                                try:
                                        workspace2=workspace.split('"')[ending_index]
                                        #bugfix for L not in a start of a class
                                        x=workspace.split("(")[0]
                                        y=workspace.split("/")[0]
                                        
                                        if len(workspace)>len(workspace2):
                                                workspace=workspace2
                                
                                except:
                                        pass
                                workspace=workspace[starting_point:]
                                
                                soundness_test.add(workspace)
                        #replace scheme occurance
                        s=s.replace(d+".",d[0]+dummy+d[1:]+".")
                        
                soundness_check_pass=set()
                
                for item in soundness_test:
                            
                            item_path=root_full_path+os.path.sep+item
                            item_path=item_path.replace(".","/")
                            if len(item.split("("))>1:
                                continue
                            if os.path.isdir(item_path.replace(".","/")) or os.path.isfile(item_path+".smali"):
                                soundness_check_pass.add(item)
                
                #print soundness_check_pass
                for item in soundness_check_pass:
                        s=s.replace(start_char+item,start_char+dummy+item)
                with open(fname,"w") as f:
                    f.write(s)
                f.close() 
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            #exit(0)


#actual manipulation func
def manipulate_app(app,dirs,root_full_path,list_of_dummies,level,key,new_apps,random_name):
        try:
                random.shuffle(list_of_dummies)
                dummy=list_of_dummies[0]
                try:
                        os.makedirs(root+os.path.sep+dummy)
                except:
                        pass
                
                start=time.time()
                #get to the dir
 
                os.chdir(root_full_path)
           
                files=tree_files(root_full_path)
                #loop of change in files only
                for i in range(0,len(files)):
                        fname=root_full_path+files[i]
                        soundness_check_and_replace(fname,dirs,root_full_path,';',"L",dummy,1,"s",0)
                
                #change in manifest
                manifest=os.getcwd()[:-6]+"/AndroidManifest.xml"
                tmp_dirs=[x.replace(os.path.sep,".") for x in dirs]

                soundness_check_and_replace(manifest,dirs,root_full_path,'"','android:name="',dummy[:-1]+".",0,"m",1)
                #change the package in manifest file

                soundness_check_and_replace(manifest,tmp_dirs,root_full_path,'"','package="',dummy[:-1]+".",0,"m",1)
                #change the layout files
                #changes in directories

                os.chdir("../")
                resources_root="res"+os.path.sep
                #tmp_dirs=["L"+x.replace(os.path.sep,".") for x in dirs]
                
                files=[xml for xml in tree_files(resources_root) if xml.endswith(".xml")]
                for i in range(0,len(files)):
                        fname=resources_root+files[i]
                
                        soundness_check_and_replace_lay(fname,dirs,root_full_path,' ',"<",dummy[:-1]+".",1,"x",0)
                        
                for i in range(0,len(dirs)):
                       former=root_full_path+os.path.sep+dirs[i][1:]

                       new=root_full_path+os.path.sep+dummy+dirs[i][1:]  
                       try:
                            tmp_dir="tmp_"+str(i)
                            if os.path.isdir(tmp_dir):
                                shutil.rmtree(tmp_dir) 
                            if os.path.isdir(new):
                                shutil.rmtree(new)  
                            os.makedirs(tmp_dir)
                            os.system("mv "+former+"/* "+tmp_dir+os.path.sep)
                            
                       except Exception as e:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno,e)
                            #exit(0)
                
                #get the files back
                for i in range(0,len(dirs)):
                        try:
                                tmp_dir="tmp_"+str(i)
                                former=root_full_path+os.path.sep+dirs[i][1:]
                                new=root_full_path+os.path.sep+dummy+dirs[i][1:]  
                                os.makedirs(new)#create the directory if not exist
                                os.system("mv "+tmp_dir+"/* "+new+os.path.sep)#move the former directory to the new place
                                
                                shutil.rmtree(tmp_dir)
                                #don't erase the new root folder
                                if os.path.normpath(former)!=os.path.normpath(root_full_path+os.path.sep+dummy):
                                        shutil.rmtree(former)
                        except Exception as e:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno,e)
                            #exit(0)
                #os.chdir("../")
                
                os.system("apktool -f b "+app)
                
                file_to_sign=app+"/dist/"+app.split(os.path.sep)[-1]+".apk"
                
                try:
                    
                        os.system("jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -storepass apkkey -keystore "+key+" "+file_to_sign+" alias_name >> tmp.csv")
                        
                        #os.system("zipalign -p 4 "+file_to_sign+" "+os.getcwd()+"/dist/new.apk")
                        os.remove("tmp.csv")
                        #os.system("rm "+file_to_sign)
                        os.system("mv "+file_to_sign+" "+new_apps+os.path.sep+app.split(os.path.sep)[-1]+"_"+random_name+".apk")
                        #os.system("rm "+file_to_sign)
                        os.chdir("../")
                        os.system("rm -r "+app.split(os.path.sep)[-1])                      
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    #exit(0)
                end=time.time()
                print "time for evil doing: ",str(end-start), " seconds"
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            #exit(0)


def app_processing(app,percent,key,new_apps,random_name,report):
        try:
                #start by opening the app
                os.system("apktool -f d "+app+".apk")
                os.system("cp -r "+app+" former_app/")
                #get the directory tree of the app
                smali_root=app+"/smali"
                app_tree=tree_dir(smali_root)

                #get the height of the tree of files
                try:
                   height=max(app_tree.keys())
                except:
                   print("no height")
                   return
                level=0#int(random.randint(0,height))
                dirs_count=0
                
                #list_of_dummies=["android/"]#list of dummies for manipulation
                list_of_dummies=["javax"]#sandroid/","google/","com/"]#list of dummies for manipulation
                #get the actual dirs to manipulate
                #dirs_to_manipulate=branching(app_tree, int(level),percent,smali_root)
                #manipulate the apps
                dirs_to_manipulate=branching2(app_tree, int(level),percent,smali_root)
                
                writer = csv.writer(open(report,'a'))
                writer.writerow([app+".apk",app.split(os.path.sep)[-1]+"_"+random_name+".apk",height,level,percent])                              
                manipulate_app(app,dirs_to_manipulate,smali_root,list_of_dummies,level,key,new_apps,random_name)
    
        except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno,e)
                return
        

#argument list:
#1-path to output directory
#2-directory of input
#3-keystore path
#4-directory for report of the attack process

#path to the new apps
new_apps=os.path.abspath(sys.argv[1])
#get other parameters
key=sys.argv[3]
report=os.path.abspath(sys.argv[4]+os.path.sep+'test_mama.csv')

with open(report, 'a') as file:  
    writer = csv.writer(file)
    writer.writerow(['name','random name','height','level','percent'])
#list the malicious directory        
new_path=sys.argv[2]
os.chdir(new_path)
files=[x for x in os.listdir(os.getcwd()) if x.endswith(".apk")]
done=["_".join(x.split("_")[:2])+".apk" for x in os.listdir(new_apps)]
new_files=[x for x in files if not (x in done)]
new_files=[x for x in new_files if x.endswith(".apk")]
#random.shuffle(files)
#loop over apps
#loop over apps
files=new_files
files=sorted(files,key=lambda x: os.stat(x).st_size)
PERMUTATIONS=1
for i in range(PERMUTATIONS):
        for app in files:
                
                app=os.path.abspath(app)
                try:
                        
                        #manipulate each app
                        random_name=''.join(random.sample((string.ascii_uppercase+string.digits),6))
                        percent=1#random.uniform(0, 1)
                        app_processing(app.replace(".apk",""),float(percent),key,new_apps,random_name,report)
                
                        os.system("rm -r former_app")
                except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno,e)
                        exit(0)
                break      
