##plot accuracy log file
# arguments
import argparse
parser = argparse.ArgumentParser(description='PyLab plot accuracy log')
parser.add_argument('--file-name', default = 'dummy_accuracy.log', help ='file name')
args = parser.parse_args()


import matplotlib.pylab as pylab
init_row = 1 #skip heading of the log file
data = pylab.loadtxt(args.file_name)#, dtype = np.float128)
for it, cost in data:
       pylab.plot(data[init_row:,0], data[init_row:,1])

pylab.xlabel('iterations')
pylab.ylabel('cost function')
#pylab.legend('SGD')
pylab.title('Training')
pylab.show()
