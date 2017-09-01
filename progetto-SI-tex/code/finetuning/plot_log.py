#import matplotlib.pyplot as plt
import pylab as pylab

filename = 'training.log'
data = pylab.loadtxt(filename)

it = data[1:,0]
train = data[1:,1]
val = data[1:,2]

#plot train and validation data
pylab.plot(it, train)
pylab.plot(it, val)
pylab.title("Resnet learning curves: finetuning on custom dataset")
pylab.xlabel('epoch')
pylab.ylabel('accuracy')
pylab.legend(['training', 'validation'])
pylab.show()
