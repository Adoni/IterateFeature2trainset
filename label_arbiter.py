import numpy
class LabelArbiter:
    def __init__(self,labeled_features=None,labeled_feature_file=None):
        assert labeled_features!=None or labeled_feature_file!=None
        if not labeled_feature_file==None:
            self.labeled_features=self.get_labeled_features(labeled_feature_file)
        else:
            self.labeled_features=labeled_features

    def get_labeled_features(self,labeled_feature_file):
        labeled_features=dict()
        for line in open(labeled_feature_file):
            line=line[:-1].split(' ')
            line[1]=line[1].split(':')
            line[2]=line[2].split(':')
            labeled_features[line[0].decode('utf8')]=numpy.array([float(line[1][1]),float(line[2][1])])
        return labeled_features

    def get_label_distribute(self,features):
        label_distribute=numpy.array([1.,1.])
        for f,v in features.items():
            if f in self.labeled_features:
                label_distribute*=self.labeled_features[f]*v
        s=1.0*sum(label_distribute)
        if s==0.0:
            return numpy.array([0.5,0.5])
        label_distribute/=s
        return label_distribute

    def arbitrate_label(self,features):
        label_distribute=self.get_label_distribute(features)
        if label_distribute[0]==label_distribute[1]:
            return -1,0.0
        else:
            return numpy.argmax(label_distribute),abs(label_distribute[0]-label_distribute[1])

if __name__=='__main__':
    l=LabelArbiter(labeled_feature_file='./labeled_features/review_constraint_age.constraints')
