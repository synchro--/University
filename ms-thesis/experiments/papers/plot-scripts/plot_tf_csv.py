import numpy as np 
import matplotlib.pyplot as plt
import sys 

filename = sys.argv[1]
title = sys.argv[2]
# retrieve data and skip first row 
_, step, loss, = np.loadtxt(filename, delimiter=",", unpack=True, skiprows=1) 

plt.figure() 
plt.suptitle(title) 

lines = plt.plot(step, loss) 
plt.setp(lines, linewidth=2) 
plt.xlabel('step') 
plt.ylabel('loss')
plt.legend(['loss']) 

# show plot 
plt.show() 
