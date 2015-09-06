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

def get_data(attribute,kind):
    data = load_svmlight_file(RAW_DATA_DIR+'iterate_label2trainset/%s_%s.data'%(attribute,kind),n_features=32000)
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
    unlabel_train_x,unlabel_train_y,unlabel_train_uids=get_data(attribute,'train_unlabel')
    train_x,train_y,train_uids=get_data(attribute,'train')
    test_x,test_y,_=get_data(attribute,'test')

    clf = MultinomialNB()
    clf.fit(train_x, train_y)
    result=dict(zip(train_uids,clf.predict_proba(train_x)))
    score=clf.score(test_x,test_y)
    print clf.score(train_x,train_y)
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
    for i in xrange(iterate_count):
        construct(attribute,initial_data_count+i*new_data_count)
        print ''
        print '============'
        print 'Iterate: %d'%i
        print '============'
        accurate,result=learn(attribute)
        fout.write('%d %f\n'%(i,accurate))
        labels=get_labels(result)
        score,feature_distribute=statistics(labels=labels,feature_file_name=base_dir+'/features/product.feature',threshold=30)
        update_labeled_feature(attribute,score,feature_distribute,max_count=1)

if __name__=='__main__':
    initial_labeled_features()
    iterate_learn('age',50,1000,100)
    iterate_learn('gender',50,1000,100)
    iterate_learn('location',50,1000,100)
    iterate_learn('kids',50,1000,100)
