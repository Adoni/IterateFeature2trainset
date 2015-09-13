import numpy
s=[]
for attribute in ['gender','age','location','kids']:
    scores=[]
    for line in open('best_accurate_%s.result'%attribute):
        score=float(line[:-1])
        scores.append(score)
    s.append('%0.4f (+/- %0.4f)'%(numpy.max(scores),numpy.std(scores)*2))
print ' & '.join(s)
