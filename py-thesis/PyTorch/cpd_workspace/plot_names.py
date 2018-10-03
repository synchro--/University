import matplotlib.pyplot as plt
import numpy as np
import pylab
import glob
import sys

### Beautiful plotting with Tableau-like colors ###

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


#small script to plot all log files together to compare
#different optimization methods

#methods = ('SGD', 'ASGD', 'LBFGS', 'NAG')
'''
#collect data from each log file
dl = [1]
for name in methods:
    filename = name+'.log'
'''

#collect data from each csv file
methods = []

epochs = []
step = []
acc = []
loss = []

for i, filename in enumerate(sorted(glob.iglob('*.csv'))):
        print(filename)
        methods.append(filename.split('.')[0])
        steps, A, L = np.loadtxt(filename, delimiter=",",
                             unpack=True)  # , skiprows=1)
        step.append(steps)
        epochs.append(steps / 1000)
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
for i, val in enumerate(acc):
    plt.plot(epochs[i], val, lw=2.0, color=tableau20[i+1])

plt.xlabel('Epochs', fontsize=11.5)
plt.ylabel('Accuracy (%)', fontsize=11.5)
plt.legend([*methods], fontsize=11)
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
    lines = plt.plot(epochs[i], loss[i], lw=2.0, color=tableau20[i+1])

plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend([*methods], fontsize=11)
plt.show()

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
plt.savefig("loss.png", bbox_inches="tight")
