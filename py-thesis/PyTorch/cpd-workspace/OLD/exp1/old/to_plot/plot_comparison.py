import matplotlib.pyplot as plt 
import glob
import numpy as np 
import sys 

layers = ('1 Conv-layer', '2 Conv-layers', '3 Conv-layers', '4 Conv-layers', '5 Conv-layers')  

epochs = []
step = []
acc = []
loss = [] 
for i, filename in enumerate(sorted(glob.iglob('*.csv'))):
    print(filename)
    steps, A, L = np.loadtxt(filename, delimiter=",", unpack=True) #, skiprows=1)
    step.append(steps)
    epochs.append(steps/1000)
    acc.append(A)
    loss.append(L)


if len(sys.argv) > 1: 
    title = sys.argv[1]

plt.figure() 
plt.title(title)

# Print accuracy comparison
for i,val in enumerate(acc): 
    lines = plt.plot(epochs[i], val)
    plt.setp(lines, linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend([*layers])
plt.show()

#-----------------------------------
plt.figure()
plt.title(title)

for i in range(len(loss)):
    lines = plt.plot(epochs[i], loss[i])
    plt.setp(lines, linewidth=2)    

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend([*layers])
plt.show()
