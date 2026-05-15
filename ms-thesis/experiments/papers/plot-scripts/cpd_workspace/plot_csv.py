import numpy as np 
import matplotlib.pyplot as plt
import sys 

filename = sys.argv[1]
title = sys.argv[2]
# retrieve data and skip first row 
# epoch, acc, dummy, loss = np.loadtxt(filename, delimiter=" ", unpack=True, skiprows=1) 
step, acc, loss = np.loadtxt(filename, delimiter=",", unpack=True, skiprows=1) 

# Add fixed value to the last 100 epochs 
# step[100:] = step[100:] + 75000 
step = step/1000 

plt.figure() 
plt.suptitle(title) 

plt.subplot(1,2,1)
i=len(loss)
lines = plt.plot(step, loss) 
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.setp(lines, linewidth=2) 
plt.legend(['loss']) 
# plt.grid(True)

plt.subplot(1,2,2) 
lines = plt.plot(step, acc)
plt.ylabel('Acc')
plt.xlabel('Epochs') 
plt.setp(lines, linewidth=2) 
plt.legend(['acc']) 
# plt.grid(True)
 

# show plot 
plt.show() 
