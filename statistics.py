#coding:utf8
from pymongo import Connection
from settings import RAW_DATA_DIR
from settings import base_dir
from collections import Counter
from small_utils.progress_bar import progress_bar
from tools import get_balance_params
from tools import get_features
from label_arbiter import LabelArbiter
import numpy
from tools import combine_dict

def eta_score(P):
    '''
    score需要和feature的暗示性呈正相关，所以这里去信息熵的相反数
    '''
    e=0.0
    for p in P:
        if p==0:
            continue
        e+=p*numpy.log(p)
    #e=-1.0*e
    return e

def KL_score(P,Q):
    assert len(P)==len(Q)
    e=0.0
    for i in xrange(len(P)):
        log_p=0.0 if P[i]==0.0 else numpy.log(P[i])
        log_q=0.0 if Q[i]==0.0 else numpy.log(Q[i])
        e+=P[i]*log_p-P[i]*log_q
    return e

def abs_score(P):
    return 1.0*max(P)/sum(P)

def statistics(labels,feature_file_name,threshold):
    collection=Connection().jd.train_users
    label_dimention=max(labels.values())+1
    label_distribute=Counter(labels.values())
    label_distribute=[label_distribute[i] if i in label_distribute else 0 for i in xrange(label_dimention)]
    all_features=get_features(feature_file_name)
    bar=progress_bar(collection.count())
    feature_distribute=dict([f,[0.]*label_dimention] for f in all_features)
    for index,user in enumerate(collection.find()):
        try:
            label=labels[user['_id']]
        except:
            continue
        features=combine_dict(user['mentions'],Counter(user['products']))
        for f in features:
            if f in feature_distribute:
                feature_distribute[f][label]+=1.0
        bar.draw(index)

    for f in feature_distribute.keys():
        s=1.0*sum(feature_distribute[f])
        if s==0 or s<threshold:
            feature_distribute.pop(f)
            continue
        for i in xrange(label_dimention):
            feature_distribute[f][i]/=label_distribute[i]

    for f in feature_distribute.keys():
        s=1.0*sum(feature_distribute[f])
        for i in xrange(label_dimention):
            feature_distribute[f][i]/=s
    score=dict()
    for f,v in feature_distribute.items():
        #score[f]=eta_score(v)
        score[f]=abs_score(v)
    return score,feature_distribute

def test():
    from pymongo import Connection
    label_arbiter=LabelArbiter(labeled_feature_file='./labeled_features/review_constraint_gender.constraints')
    collection=Connection().jd.train_users
    bar=progress_bar(collection.count())
    labels=dict()
    for index,user in enumerate(collection.find()):
        label,confidence=label_arbiter.arbitrate_label(user['mentions'])
        if label==-1:
            continue
        labels[user['_id']]=label
        bar.draw(index+1)
        if index>10000:
            break
    score,feature_distribute=statistics(labels,feature_file_name=base_dir+'/features/mention.feature',threshold=50)
    for f,v in sorted(score.items(),key=lambda d:d[1],reverse=True)[:50]:
        print f,v,feature_distribute[f]

if __name__=='__main__':
    test()
    print 'Done'
