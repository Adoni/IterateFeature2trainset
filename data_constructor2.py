#coding:utf8
from tools import balance
from pymongo import Connection
from small_utils.progress_bar import progress_bar
from collections import Counter
from tools import get_features
from settings import RAW_DATA_DIR
from tools import get_balance_params
import random
from collections import Counter
from settings import base_dir
from settings import labeled_feature_file_dir
from label_arbiter import LabelArbiter

feature_file_name=base_dir+'/features/all_features.feature'

def combine_features(a,b):
    c=dict()
    for key in a:
        c[key]=a[key]
    for key in b:
        c[key]=b[key]
    return c

def construct_all_data():
    '''
    The format of labeled_feature_file is as the same as mallet
    '''
    all_features=get_features(feature_file_name=feature_file_name)
    all_features_1=get_features(feature_file_name=base_dir+'/features/mention_1.feature',start_index=max(all_features.values())+1)
    collection=Connection().jd.train_users
    bar=progress_bar(collection.count())
    fout=open(RAW_DATA_DIR+'iterate_label2trainset/all_train.data','w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/all_train_uids.data','w')
    for index,user in enumerate(collection.find()):
        label=0
        fout.write('%d'%label)
        uid_output.write('%s\n'%user['_id'])
        features=combine_features(user['mentions_1'],Counter(user['products']))
        sorted_feature=[]
        for f in features:
            if f not in all_features:
                continue
            sorted_feature.append((all_features[f],features[f]))
        for f,v in user['mentions_1_1'].items():
            f=f+'_1'
            if f not in all_features_1:
                continue
            sorted_feature.append((all_features_1[f],v))
        sorted_feature=sorted(sorted_feature,key=lambda d:d[0])
        for f in sorted_feature:
            fout.write(' %s:%d'%f)
        fout.write('\n')
        bar.draw(index+1)

def construct_train_set(attribute,training_count):
    '''
    The format of labeled_feature_file is as the same as mallet
    '''
    all_features=get_features(feature_file=feature_file_name)
    all_features_1=get_features(feature_file=base_dir+'/features/mention_1.feature',existent_features=all_features)
    review_featuers=get_features(feature_file=base_dir+'/features/review.feature',existent_features=all_features_1)
    labeled_feature_file=open('%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute))
    label_arbiter=LabelArbiter(labeled_feature_file='%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute))
    labeled_features=dict()
    for line in labeled_feature_file:
        line=line[:-1].split(' ')
        labeled_features[line[0].decode('utf8')]=map(lambda d:float(d.split(':')[1]),line[1:])
    collection=Connection().jd.train_users

    bar=progress_bar(collection.count())
    confidence=[]
    for index,user in enumerate(collection.find()):
        label_distributed=[1,1]
        for f,value in combine_features(user['mentions'],Counter('products')).items():
            if f in labeled_features:
                label_distributed[0]*=labeled_features[f][0]*value
                label_distributed[1]*=labeled_features[f][1]*value
        s=1.0*sum(label_distributed)
        if not s==0:
            label_distributed[0]/=s
            label_distributed[1]/=s
        label_distributed=label_arbiter.get_label_distribute(combine_features(user['mentions'],Counter('products')))
        if label_distributed[0]>label_distributed[1]:
            label=0
        elif label_distributed[0]<label_distributed[1]:
            label=1
        else:
            label=-1

        features=combine_features(user['mentions_0'],Counter(user['products']))
        sorted_feature=[]
        for f in features:
            if f not in all_features:
                continue
            sorted_feature.append((all_features[f],features[f]))

        user['mentions_1_1']={}
        for f,v in user['mentions_1_1'].items():
            f=f+'_1'
            if f not in all_features_1:
                continue
            sorted_feature.append((all_features_1[f],v))

        for f,v in Counter(user['review']).items():
            if f not in review_featuers:
                continue
            sorted_feature.append((review_featuers[f],v))

        keys=map(lambda d:d[0], sorted_feature)
        if not len(keys)==len(set(keys)):
            print Counter(keys).values()
        sorted_feature=sorted(sorted_feature,key=lambda d:d[0])
        str_features=' '.join(map(lambda f:'%s:%f'%f,sorted_feature))
        confidence.append(
                (user['_id'],
                    label,
                    abs(label_distributed[0]-label_distributed[1]),
                    str_features,
                    sum(user['mentions'].values()),
                    ))
        bar.draw(index+1)

    confidence0=filter(lambda d:d[1]==0,confidence)
    confidence0=sorted(confidence0,key=lambda d:d[2],reverse=True)
    confidence1=filter(lambda d:d[1]==1,confidence)
    confidence1=sorted(confidence1,key=lambda d:d[2],reverse=True)
    confidence2=filter(lambda d:d[1]==-1,confidence)
    confidence2=sorted(confidence2,key=lambda d:d[4],reverse=True)

    dimention=min(len(confidence0),len(confidence1),training_count/2)
    confidence0=confidence0[:dimention]
    confidence1=confidence1[:dimention]
    confidence2=confidence2[:dimention]


    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_uids.data'%attribute,'w')
    for d in confidence0+confidence1:
        fout.write('%d %s\n'%(d[1],d[3]))
        uid_output.write('%s\n'%d[0])

    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_unlabel.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_unlabel_uids.data'%attribute,'w')
    for d in confidence2:
        fout.write('%d %s\n'%(d[1],d[3]))
        uid_output.write('%s\n'%d[0])

def construct_test_set(attribute):
    all_features=get_features(feature_file=feature_file_name)
    all_features_1=get_features(feature_file=base_dir+'/features/mention_1.feature',existent_features=all_features)
    review_featuers=get_features(feature_file=base_dir+'/features/review.feature',existent_features=all_features_1)
    collection=Connection().jd.test_users
    balance_params=get_balance_params(attribute,collection)
    print balance_params
    bar=progress_bar(collection.count())
    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_test.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_test_uids.data'%attribute,'w')
    for index,user in enumerate(collection.find()):
        try:
            label=user['profile'][attribute].index(1)
        except Exception as e:
            continue
        if random.random()>balance_params[label]:
            continue

        features=combine_features(user['mentions_0'],Counter(user['products']))
        sorted_feature=[]
        for f in features:
            if f not in all_features:
                continue
            sorted_feature.append((all_features[f],features[f]))

        for f,v in user['mentions_1_1'].items():
            f=f+'_1'
            if f not in all_features_1:
                continue
            sorted_feature.append((all_features_1[f],v))

        for f,v in Counter(user['review']).items():
            if f not in review_featuers:
                continue
            sorted_feature.append((review_featuers[f],v))

        if len(sorted_feature)==0:
            continue
        fout.write('%d'%label)
        uid_output.write('%s\n'%user['_id'])
        keys=map(lambda d:d[0], sorted_feature)
        if not len(keys)==len(set(keys)):
            print Counter(keys).values()
        sorted_feature=sorted(sorted_feature,key=lambda d:d[0])
        for f in sorted_feature:
            fout.write(' %s:%f'%f)
        fout.write('\n')
        bar.draw(index+1)

def construct(attribute,training_count):
    construct_train_set(attribute,training_count)
    construct_test_set(attribute)

if __name__=='__main__':
    construct('gender',10)
