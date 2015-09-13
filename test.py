from pymongo import Connection
from small_utils.progress_bar import progress_bar

def get_test_uids():
    collection=Connection().jd.test_users
    uids=set()
    for user in collection.find():
        uids.add(user['_id'])
    return uids

def get_train_uids():
    collection=Connection().jd.train_users
    uids=set()
    for user in collection.find():
        uids.add(user['_id'])
    collection=Connection().jd.train_users
    bar=progress_bar(len(uids))
    for index,uid in enumerate(uids):
        collection.delete_one({'_id':uid})
        bar.draw(index+1)

def remove():
    uids=get_test_uids() & get_train_uids()
    collection

if __name__=='__main__':
    remove()
    print 'Done'
