import numpy
from settings import base_dir
def save_vector_to_text(vector, file_name, folder,dimention=0):
    import os
    path=MATRIXES_DIR+folder+'/'
    print 'Try to save vector in '+path+file_name
    if not os.path.exists(path):
        os.makedirs(path)
    if type(vector[0])==int or type(vector[0])==str or type(vector[0])==unicode:
        fout=open(path+file_name,'w')
        for item in vector:
            fout.write(str(item)+'\n')
        return
    if type(vector[0])==numpy.ndarray:
        numpy.savetxt(path+file_name,vector)
        return
    if type(vector[0])==dict:
        fout=open(path+file_name,'w')
        fout.write(str(dimention)+'\n')
        for item in vector:
            item=' '.join(map(lambda x:str(x)+':'+str(item[x]),item.keys()))+'\n'
            fout.write(item)
        return
    print 'Fail to save vector'
    print 'Vector type should be list or numpy.ndarray'

def load_vector_from_text(file_name, folder, file_type):
    import os
    path=MATRIXES_DIR+folder+'/'
    print 'Try to load vector in '+path+file_name
    if file_type=='list':
        fout=open(path+file_name)
        vector=[line[:-1] for line in open(path+file_name)]
        return vector
    if file_type=='NParray':
        vector=numpy.loadtxt(path+file_name,dtype='float64')
        return vector
    if file_type=='LightArray':
        fin=open(path+file_name)
        dimension=int(fin.readline()[:-1])
        vector=[]
        for line in fin:
            line=line[:-1].split(' ')
            v=numpy.zeros((dimension+1))
            for item in map(lambda d:d.split(':'),line):
                v[int(item[0])]=int(item[1])
            vector.append(v)
        return vector
    print 'Fail to load vector'
    print 'Vector type should be list or numpy.ndarray'
    return None

def doc2vec_embedding(file_name):
    import sys
    import gensim
    from pymongo import Connection

    users=Connection().jd.jd_users
    dimensionality_size=200
    window_size=8
    workers=5
    min_count=5

    # load sentences
    finish_count=0
    total_count=users.find({'got_review':True}).count()
    #total_count=users.count()
    sentences = []
    print total_count
    old_review=''
    for user in users.find({'got_review':True}):
    #for user in users.find():
        if finish_count%10000==0:
            sys.stdout.write("\r%f"%(finish_count*1.0/total_count))
            sys.stdout.flush()
        finish_count+=1
        content=[]
        for behavior in user['behaviors']:
            #content.append(str(behavior['item']))
            #content.append(behavior['item_class'][0])
            review=' '.join(behavior['review']['parsed_review_general'])
            if review==old_review:
                continue
            old_review==review
            content+=review.split()
        for ch in [' ','\n','\r','\u3000']:
            while 1:
                try:
                    content.remove(ch)
                except:
                    break
        #print ' '.join(content)
        if len(content)<10:
            continue
        sentence = gensim.models.doc2vec.LabeledSentence(words=content,labels=['USER_%d'%user['_id']])
        sentences.append(sentence)

    print 'load corpus completed...'

    # train word2vc
    model = gensim.models.Doc2Vec(sentences,size=200,window=7, workers=20,min_count=3,sample=1e-3)
    model.save_word2vec_format('/mnt/data1/adoni/jd_data/vectors/'+file_name+'.data',binary=False)
    print 'embedding done'
    return model

def load_doc2vec_embedding(file_name):
    import gensim
    print 'Loading user embedding'
    embedding=gensim.models.Word2Vec.load_word2vec_format('/mnt/data1/adoni/jd_data/'+file_name+'.data',binary=False)
    print 'Done'
    return embedding

def get_mentions():
    file_name=base_dir+'/features/mention.feature'
    mentions=[]
    for line in open(file_name):
        mentions.append(line[:-1].decode('utf8'))
    return mentions

def balance(data,target_index):
    from collections import Counter
    import random
    counts=Counter(map(lambda d:d[target_index],data))
    min_count=min(counts.values())
    ratio=dict()
    for target in counts:
        ratio[target]=1.0*min_count/counts[target]
    balanced_data=[]
    for d in data:
        if random.random()>ratio[d[target_index]]:
            continue
        balanced_data.append(d)
    return balanced_data

def split(data,distribute):
    import random
    if sum(distribute)>1:
        raise Exception('Sum of distribute must be smaller than 1.0')
    datas=[[] for s in distribute]
    for d in data:
        r=random.random()
        for i in xrange(len(distribute)):
            if sum(distribute[:i+1])>r:
                datas[i].append(d)
                break
    return datas

def get_balance_params(attribute,collections):
    from collections import Counter
    distribute=[]
    for user in collections.find():
        try:
            label=user['profile'][attribute].index(1)
        except:
            continue
        distribute.append(label)
    distribute=Counter(distribute)
    params=[0]*len(distribute)
    for d in distribute:
        params[d]=1.0*min(distribute.values())/distribute[d]
    return params

def get_features(feature_file,existent_features={}):
    start_index=0 if existent_features=={} else max(existent_features.values())+1
    all_features=dict()
    for index,line in enumerate(open(feature_file)):
        all_features[line[:-1].decode('utf8')]=index+start_index
    return all_features

def combine_dict(*args):
    '''
    Used to combine several dict with different keys
    '''
    result=dict()
    for d in args:
        for k,v in d.items():
            assert k not in result
            result[k]=v
    return result

if __name__=='__main__':
    pass
