import os
from sklearn.datasets import load_svmlight_file
from settings import RAW_DATA_DIR
from settings import base_dir
from sklearn.naive_bayes import MultinomialNB
from statistics import *
from settings import labeled_feature_file_dir
import numpy

def initial_labeled_features():
    os.system('rm -r %s'%labeled_feature_file_dir)
    os.system('cp -r %s/initial_labeled_features %s'%(base_dir,labeled_feature_file_dir))

def random_initialize_labeled_feature(count=2):
    import random
    os.system('rm -r %s'%labeled_feature_file_dir)
    os.system('cp -r %s/initial_labeled_features %s'%(base_dir,labeled_feature_file_dir))
    for attribute in ['gender','age','location','kids']:
        print attribute
        f=open('%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute))
        con0=[]
        con1=[]
        for line in f:
            if float(line.split(' ')[1].split(':')[1])>0.5:
                con0.append(line)
            else:
                con1.append(line)
        f=open('%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute),'w')
        for i in xrange(count/2):
            f.write(con0[i])
            f.write(con1[i])

def linear_initialize_labeled_feature(count=2):
    import random
    os.system('rm -r %s'%labeled_feature_file_dir)
    os.system('cp -r %s/initial_labeled_features %s'%(base_dir,labeled_feature_file_dir))
    for attribute in ['gender','age','location','kids']:
        print attribute
        f=open('%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute))
        con0=[]
        con1=[]
        for line in f:
            if float(line.split(' ')[1].split(':')[1])>0.5:
                con0.append(line)
            else:
                con1.append(line)
        f=open('%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute),'w')
        for i in xrange(min(len(con0),count/2)):
            f.write(con0[random.randint(0,len(con0)-1)])
        for i in xrange(min(len(con1),count/2)):
            f.write(con1[random.randint(0,len(con1)-1)])

def get_data(attribute,kind):
    data = load_svmlight_file(RAW_DATA_DIR+'iterate_label2trainset/%s_%s.data'%(attribute,kind),n_features=34000)
    uids=[line[:-1] for line in open(RAW_DATA_DIR+'iterate_label2trainset/%s_%s_uids.data'%(attribute,kind))]
    return data[0].toarray(), data[1], uids

def get_labels(result):
    labels=dict()
    for uid in result:
        label=numpy.argmax(result[uid])
        confidence=result[uid][label]
        labels[uid]=label
    return labels

def learn(attribute):
    from sklearn.metrics import f1_score
    unlabel_train_x,unlabel_train_y,unlabel_train_uids=get_data(attribute,'train_unlabel')
    train_x,train_y,train_uids=get_data(attribute,'train')
    test_x,test_y,_=get_data(attribute,'test')

    clf = MultinomialNB()
    clf.fit(train_x, train_y)
    result=dict(zip(unlabel_train_uids,clf.predict_proba(unlabel_train_x)))
    #score=clf.score(test_x,test_y)
    y_true=test_y
    y_pred=clf.predict(test_x)
    score=f1_score(y_true, y_pred, average=None)
    print score
    score=numpy.mean(score)
    print '------'
    print 'Labeled training data size: %d'%(len(train_x))
    print 'Unlabeled training data size: %d'%(len(unlabel_train_x))
    print 'Testing data size: %d'%(len(test_x))
    print 'Accurate: %0.4f'%(score)
    print '------'
    return score,result

def update_labeled_feature(attribute,score,feature_distribute, max_count=1):
    fin=open(base_dir+'/labeled_features/review_constraint_%s.constraints'%attribute)
    exist_labes=[line.split(' ')[0].decode('utf8') for line in fin]
    fin.close()
    score=sorted(score.items(),key=lambda d:d[1], reverse=True)
    fout=open(base_dir+'/labeled_features/review_constraint_%s.constraints'%attribute,'a')
    count=0
    for f,_ in score:
        if f in exist_labes:
            continue
        if sum(feature_distribute[f])==0:
            break
        d=feature_distribute[f]
        print max(d)
        if max(d)<0.8:
            break
        d=map(lambda i:'%d:%0.4f'%(i,d[i]),xrange(len(d)))
        d=' '.join(d)
        fout.write('%s %s\n'%(f.encode('utf8'),d))
        count+=1
        if count==max_count:
            break

def iterate_learn(attribute,iterate_count,initial_data_count,new_data_count):
    from data_constructor import construct
    print 'Attribute: %s'%attribute
    fout=open(base_dir+'/iterate_result_%s.result'%attribute,'w')
    fbest=open(base_dir+'/best_accurate_%s.result'%attribute,'a')
    best_accurate=0.0
    for i in xrange(iterate_count):
        training_count=initial_data_count+i*new_data_count
        construct(attribute,training_count)
        print ''
        print '============'
        print 'Iterate: %d'%i
        print '============'
        accurate,result=learn(attribute)
        best_accurate=max(best_accurate,accurate)
        fout.write('%d %f\n'%(i,accurate))
        #labels=get_labels(result)
        #score,feature_distribute=statistics(labels=labels,feature_file_name=base_dir+'/features/all_features.feature',threshold=new_data_count)
        #score,feature_distribute=statistics(labels=labels,feature_file_name=base_dir+'/features/all_features.feature',threshold=10)
        #update_labeled_feature(attribute,score,feature_distribute,max_count=1)
    fbest.write('%d %f\n'%(new_data_count,best_accurate))
    fout.close()
    fbest.close()

if __name__=='__main__':
    for i in xrange(1):
        print i
        #initial_labeled_features()
        #random_initialize_labeled_feature(count=4)
        linear_initialize_labeled_feature(count=6)
        count=200
        iterate_count=1
        iterate_learn('gender',iterate_count,10000,count)
        iterate_learn('age',iterate_count,10000,count)
        iterate_learn('location',iterate_count,10000,count)
        iterate_learn('kids',iterate_count,5000,count)
    print 'Done'
