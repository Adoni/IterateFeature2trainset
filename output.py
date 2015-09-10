import numpy
def output(attribute):
    print '--------'
    print attribute
    print '--------'
    for line in open('./iterate_result_%s.result'%attribute):
        print float(line[:-1].split(' ')[1])

def output_max(attribute):
    print '--------'
    print attribute
    attributes=[]
    for line in open('./iterate_result_%s.result'%attribute):
        attributes.append(float(line[:-1].split(' ')[1]))
    print numpy.max(attributes)

if __name__=='__main__':
    for attribute in ['gender','age','location','kids']:
        output_max(attribute)
        #output(attribute)
