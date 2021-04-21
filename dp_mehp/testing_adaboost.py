from all_aux_files import test_results_subsampling_rate, test_gen_data, prep_data
import os

seed=0
ep=10
hp=100
bs=200
lr=0.01
kernel=0.0002

data='digits'

#log_dir='/Users/margaritavinaroz/Desktop/DPDR/dp_mehp/logs/gen/'
#log_name='fashion_CNN_lr' + str(lr) + '_kernel' + str(kernel)+'-bs-' + str(bs)+'-seed-'+str(seed)+'-epochs-'+ str(ep)+ '-hp-' + str(hp)
log_name='digits_FC_lr' + str(lr) + '_kernel' + str(kernel)+'-bs-' + str(bs)+'-seed-'+str(seed)+'-epochs-'+ str(ep)+ '-hp-' + str(hp)

skip_downstream_model=False
sampling_rate_synth=0.1

#test_results_subsampling_rate(data, log_name + '/' + data, log_dir, skip_downstream_model, sampling_rate_synth)
#final_score = test_gen_data(log_name + '/' + data, data, subsample=0.1, custom_keys='adaboost')    

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


log_dir='logs/gen/' + log_name
gen_data_dir = os.path.join(log_dir,  data)
log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
data_path = os.path.join(gen_data_dir, 'synthetic_mnist.npz')

data_from_torch=False
shuffle_data=False
sub_balanced_labels=True


dc = prep_data(data, data_from_torch, data_path, shuffle_data, sampling_rate_synth, sub_balanced_labels)


list_acc=[]

"""Testing Adaboost parameters"""
#list_estimators=[1000]
#list_lr=[0.7]


#for i in list_estimators:
#    for j in list_lr:
#        model = AdaBoostClassifier(n_estimators=i, learning_rate=j, algorithm='SAMME.R')  # improved
#        model.fit(dc.x_gen, dc.y_gen)
#        y_pred = model.predict(dc.x_real_test) 
#        acc = accuracy_score(y_pred, dc.y_real_test)
#        print("For num_iter and lr:", (i,j) )
#        print("the accuracy on adaboost is: ", acc)
#        list_acc.append(acc)
        
#max_acc=max(list_acc)



"""Testing Gaussian_nb parameters"""
list_var=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

for i in list_var:
    model = GaussianNB(var_smoothing=i)
    model.fit(dc.x_gen, dc.y_gen)
    y_pred = model.predict(dc.x_real_test) 
    acc = accuracy_score(y_pred, dc.y_real_test)
    print("For var_smothing:", i )
    print("the accuracy on Gaussian_nb is: ", acc)
    list_acc.append(acc)
    
max_acc=max(list_acc)   
    


"""Testing Decision_tree parameters"""

#For GINI criterion and different max_features options
#model = DecisionTreeClassifier(criterion='gini', class_weight='balanced')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree gini is: ", acc)
#list_acc.append(acc)
    
#model = DecisionTreeClassifier(criterion='gini', class_weight='balanced', max_features='log2')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree gini with max_features=log2 is: ", acc)
#list_acc.append(acc)
    
#model = DecisionTreeClassifier(criterion='gini', class_weight='balanced', max_features='auto')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree gini with max_features=auto is: ", acc)
#list_acc.append(acc)
    
    
#model = DecisionTreeClassifier(criterion='gini', class_weight='balanced', max_features='sqrt')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree gini with max_features=sqrt is: ", acc)
#list_acc.append(acc)
    
    
#For ENTROPY criterion and different max_features options
#model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree entropy is: ", acc)
#list_acc.append(acc)
    
#model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='log2')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree entropy with max_features=log2 is: ", acc)
#list_acc.append(acc)
    
#model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='auto')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree entropy with max_features=auto is: ", acc)
#list_acc.append(acc)
    
    
#model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features='sqrt')
#model.fit(dc.x_gen, dc.y_gen)
#y_pred = model.predict(dc.x_real_test) 
#acc = accuracy_score(y_pred, dc.y_real_test)
#print("the accuracy on decision_tree entropy with max_features=sqrt is: ", acc)
#list_acc.append(acc)
