from data_constructor import construct_test_set
from sklearn.datasets import load_svmlight_file
from settings import RAW_DATA_DIR
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
import numpy

s=[]
for attribute in ['gender','age','location','kids']:
    print attribute
    data = load_svmlight_file(RAW_DATA_DIR+'iterate_label2trainset/%s_test.data'%(attribute),n_features=32000)
    x,y=data[0].toarray(),data[1]
    print len(x)
    kf=KFold(len(x), n_folds=5)
    scores=[]
    for train, test in kf:
        clf=MultinomialNB()
        train_x, test_x, train_y, test_y = x[train], x[test], y[train], y[test]
        clf.fit(train_x,train_y)
        y_true=test_y
        y_pred=clf.predict(test_x)
        score=f1_score(y_true, y_pred, average=None)
        print score
        score=numpy.mean(score)
        scores.append(score)
    s.append('%0.3f'%numpy.mean(scores))
print '&'.join(s)
