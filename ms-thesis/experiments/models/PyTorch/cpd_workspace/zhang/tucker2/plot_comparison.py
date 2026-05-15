import matplotlib.pyplot as plt 
import glob
import numpy as np 
import sys 

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
         (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
         (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
         (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
         (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

'''  
# You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
# exception because of the number of lines being plotted on it.    
# Common sizes: (10, 7.5) and (12, 9)    
# plt.figure(figsize=(12, 14))    
  
# Remove the plot frame lines. They are unnecessary chartjunk.    
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(True)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(True)    
  
# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()    
  
# Limit the range of the plot to only where the data is.    
# Avoid unnecessary whitespace.    
# plt.ylim(0, 100)    
# plt.xlim(0, 50)    
  
# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    
# plt.yticks(range(0, 101, 10), [str(x) + "%" for x in range(0, 101, 10)], fontsize=14)    
# plt.xticks(fontsize=14)    
  
# Provide tick lines across the plot to help your viewers trace along    
# the axis ticks. Make sure that the lines are light and small so they    
# don't obscure the primary data lines.    
# for y in range(10, 91, 10):    
#     plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)     


plt.show() 
'''

######## Actual Plotting ########
layers = ['1 Conv-layer', '2 Conv-layers', '3 Conv-layers']#, '4 Conv-layers', '5 Conv-layers'] 
layers.reverse() 

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

#plt.figure(figsize=(12, 14))
plt.title(title, fontsize=14)
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(True)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(True)

# Print accuracy comparison
for i,val in enumerate(acc): 
    plt.plot(epochs[i], val, lw=2.5, color=tableau20[i+4])

plt.xlabel('Epochs', fontsize=11.5)
plt.ylabel('Accuracy (%)', fontsize=11.5)
plt.legend([*layers], fontsize=11)
plt.show()

# Finally, save the figure as a PNG.    
# You can also save it as a PDF, JPEG, etc.    
# Just change the file extension in this call.    
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
plt.savefig("acc.png", bbox_inches="tight") 

#-----------------------------------
#plt.figure(figsize=(12, 14))    
plt.title(title, fontsize=14)
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(True)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(True)  

for i in range(len(loss)):
    lines = plt.plot(epochs[i], loss[i], lw=2.5, color=tableau20[i+4])

plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend([*layers], fontsize=11)
plt.show()

# Finally, save the figure as a PNG.    
# You can also save it as a PDF, JPEG, etc.    
# Just change the file extension in this call.    
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
plt.savefig("loss.png", bbox_inches="tight") 
