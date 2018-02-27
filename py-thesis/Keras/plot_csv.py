import numpy as np 
import matplotlib.pyplot as plt
import sys 

filename = sys.argv[1]
title = sys.argv[2]
# retrieve data and skip first row 
epoch, acc, loss, val_acc, val_loss = np.loadtxt(filename, delimiter=",", unpack=True, skiprows=1) 

plt.figure() 
plt.suptitle(title) 

plt.subplot(2,1,1)
lines = plt.plot(epoch, acc, val_acc) 
plt.setp(lines, linewidth=2) 
plt.legend(['acc', 'val_acc']) 


plt.subplot(2,1,2) 
lines = plt.plot(epoch, loss, val_loss) 
plt.setp(lines, linewidth=2) 
plt.legend(['loss', 'val_loss']) 

# show plot 
plt.show() 
