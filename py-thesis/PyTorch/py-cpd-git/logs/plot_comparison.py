import matplotlib.pyplot as plt 
import glob
import numpy as np 

layers = ('conv1', 'conv1+conv2', 'conv1+conv2+conv3', 'ALL 4 conv')  

epochs = []
step = []
acc = []
loss = [] 
for i, filename in enumerate(sorted(glob.iglob('*.csv'))): 
    steps, A, L = np.loadtxt(filename, delimiter=",", unpack=True, skiprows=1)
    step.append(steps)
    epochs.append(steps/1000)
    acc.append(A)
    loss.append(L)

plt.figure() 
plt.title('Fine-tuning - LeNet 4-Conv - incrementive CPD')

# Print accuracy comparison
for i in range(len(acc)): 
    lines = plt.plot(epochs[i], acc[i])
    plt.setp(lines, linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend([*layers])
plt.show()

#-----------------------------------
plt.figure()
plt.title('Fine-tuning - LeNet 4-Conv - incrementive CPD')

for i in range(len(loss)):
    lines = plt.plot(epochs[i], loss[i])
    plt.setp(lines, linewidth=2)    

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend([*layers])
plt.show()
