#coding:utf8
from tools import balance
from pymongo import Connection
from small_utils.progress_bar import progress_bar
from collections import Counter
from tools import get_features
from settings import RAW_DATA_DIR
from tools import get_balance_params
from tools import combine_dict
import random
from collections import Counter
from settings import base_dir
from settings import labeled_feature_file_dir
from label_arbiter import LabelArbiter
from test import get_test_uids

def construct_train_set(attribute,training_count):
    product_features=get_features(feature_file=base_dir+'/features/product.feature')
    mention_features=get_features(feature_file=base_dir+'/features/mention.feature',existent_features=product_features)
    review_featuers=get_features(feature_file=base_dir+'/features/review.feature',existent_features=mention_features)
    mention_features_1=get_features(feature_file=base_dir+'/features/mention_1.feature',existent_features=review_featuers)
    test_uids=get_test_uids()

    labeled_feature_file='%s/review_constraint_%s.constraints'%(labeled_feature_file_dir,attribute)
    label_arbiter=LabelArbiter(labeled_feature_file=labeled_feature_file)
    collection=Connection().jd.train_users
    bar=progress_bar(collection.count())
    guess=[]
    for index,user in enumerate(collection.find()):
        if user['_id'] in test_uids:
            continue
        #features=combine_dict(user['mentions_0'],Counter(user['products']))
        features=combine_dict(user['mentions_0'],Counter('products'))
        label,confidence=label_arbiter.arbitrate_label(features)
        x=[]

        #user['products']=[]
        for f,v in Counter(user['products']).items():
            if f not in product_features:
                continue
            x.append((product_features[f],v))

        #user['mentions_0']={}
        for f,v in user['mentions_0'].items():
            if f not in mention_features:
                continue
            x.append((mention_features[f],v))

        #user['review']=[]
        for f,v in Counter(user['review']).items():
            if f not in review_featuers:
                continue
            x.append((review_featuers[f],v))

        #user['mentions_1']={}
        for f,v in user['mentions_1'].items():
            f=f+'_1'
            f0=f
            if f not in mention_features_1:
                continue
            if f0 in user['mentions_0']:
                v-=user['mentions_0'][f0]
            x.append((mention_features_1[f],v))

        x=sorted(x,key=lambda d:d[0])
        str_x=' '.join(map(lambda f:'%s:%f'%f,x))
        guess.append(
                (user['_id'],
                    label,
                    abs(confidence),
                    str_x,
                    sum(user['mentions'].values()),
                    ))
        bar.draw(index+1)

    data0=filter(lambda d:d[1]==0,guess)
    data0=sorted(data0,key=lambda d:d[2],reverse=True)
    data1=filter(lambda d:d[1]==1,guess)
    data1=sorted(data1,key=lambda d:d[2],reverse=True)
    data2=filter(lambda d:d[1]==-1,guess)
    data2=sorted(data2,key=lambda d:d[4],reverse=True)

    dimention=min(len(data0),len(data1),training_count/2)

    data0=data0[:dimention]
    data1=data1[:dimention]
    data2=data2[:dimention]


    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_uids.data'%attribute,'w')
    for d in data0+data1:
        fout.write('%d %s\n'%(d[1],d[3]))
        uid_output.write('%s\n'%d[0])

    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_unlabel.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_train_unlabel_uids.data'%attribute,'w')
    for d in data2:
        fout.write('%d %s\n'%(d[1],d[3]))
        uid_output.write('%s\n'%d[0])

def construct_test_set(attribute):
    product_features=get_features(feature_file=base_dir+'/features/product.feature')
    mention_features=get_features(feature_file=base_dir+'/features/mention.feature',existent_features=product_features)
    review_featuers=get_features(feature_file=base_dir+'/features/review.feature',existent_features=mention_features)
    mention_features_1=get_features(feature_file=base_dir+'/features/mention_1.feature',existent_features=review_featuers)

    collection=Connection().jd.test_users
    balance_params=get_balance_params(attribute,collection)
    print 'Balance params: ',balance_params
    bar=progress_bar(collection.count())
    fout=open(RAW_DATA_DIR+'iterate_label2trainset/%s_test.data'%attribute,'w')
    uid_output=open(RAW_DATA_DIR+'iterate_label2trainset/%s_test_uids.data'%attribute,'w')
    for index,user in enumerate(collection.find()):
        try:
            label=user['profile'][attribute].index(1)
        except Exception as e:
            continue
        #if random.random()>balance_params[label]:
        #    continue

        '============'
        x=[]

        #user['products']=[]
        for f,v in Counter(user['products']).items():
            if f not in product_features:
                continue
            x.append((product_features[f],v))

        #user['mentions_0']={}
        for f,v in user['mentions_0'].items():
            if f not in mention_features:
                continue
            x.append((mention_features[f],v))

        #user['review']=[]
        for f,v in Counter(user['review']).items():
            if f not in review_featuers:
                continue
            x.append((review_featuers[f],v))

        #user['mentions_1']={}
        for f,v in user['mentions_1'].items():
            f=f+'_1'
            f0=f
            if f not in mention_features_1:
                continue
            if f0 in user['mentions_0']:
                v-=user['mentions_0'][f0]
            x.append((mention_features_1[f],v))

        x=sorted(x,key=lambda d:d[0])
        str_x=' '.join(map(lambda f:'%s:%f'%f,x))

        fout.write('%d %s\n'%(label,str_x))
        uid_output.write('%s\n'%(user['_id']))
        bar.draw(index+1)

def construct(attribute,training_count):
    construct_train_set(attribute,training_count)
    construct_test_set(attribute)

if __name__=='__main__':
    construct('age',10000)
    construct('location',10000)
