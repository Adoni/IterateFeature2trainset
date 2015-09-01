def output(attribute):
    print '--------'
    print attribute
    print '--------'
    for line in open('./iterate_result_%s.result'%attribute):
        print float(line[:-1].split(' ')[1])

if __name__=='__main__':
    for attribute in ['gender','age','location','kids']:
        output(attribute)
