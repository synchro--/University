import matplotlib.pyplot as plt
import pylab
import glob

#small script to plot all log files together to compare
#different optimization methods

#methods = ('SGD', 'ASGD', 'LBFGS', 'NAG')
'''
#collect data from each log file
dl = [1]
for name in methods:
    filename = name+'.log'
'''

#collect data from each log file
dl = [1]
methods = []

for filename in glob.iglob('./*log'):
    methods.append(filename.split('.')[1].split('/')[1])
    data = pylab.loadtxt(filename)
    dl[0] = data[1:,0]
    dl.append(data[1:,1])

#plot data
for i in range(1,len(dl)):
    pylab.plot(dl[0], dl[i])

pylab.xlabel('iterations');
pylab.ylabel('cost function');
pylab.legend([ methods[0], methods[1], methods[2], methods[3] ])#methods[4] ])
pylab.show()
