def remove_num(fname,max_line=20000):
    f=open(fname).readlines()
    fout=open(fname,'w')
    for line in f[:max_line]:
        line=line[:-1].split(' ')[0]+'\n'
        fout.write(line)


if __name__=='__main__':
    remove_num('./review.feature')
