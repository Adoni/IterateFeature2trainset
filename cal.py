import numpy
import random
s=[]
for attribute in ['gender','age','location','kids']:
    scores=[]
    for line in open('best_accurate_%s.result'%attribute):
        score=float(line[:-1].split(' ')[1])
        #score+=random.gauss(0., 0.015)
        scores.append(score)
    #s.append('%0.3f (+/- %0.3f)'%(numpy.mean(scores),numpy.std(scores)*2))
    s.append('%0.3f'%(numpy.max(scores[-1:])))
print '\t'.join(s)
