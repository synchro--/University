Here's the digit classifier from Clement Farabet '<link here>' with some little changes to make the user see with hands the result of the training. 
You will need Lua and Torch7 framework in order to run the example above. 

Usage: 

Training: 
th train-on-mnist.lua

Running with visualization: 
First of all open a terminal tab and do : 
luajit -l gfx.go 
this will launch gfx.js server for the image rendering (needed)

then simply do: 
th train-on-mnist.lua -v 

This will let you visualize every image tested with a label in the left corner telling you what the predicted value of the network is. 

N.B. To do this, you should train the network before. 
