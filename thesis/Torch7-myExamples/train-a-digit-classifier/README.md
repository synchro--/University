<h2> Digit Classifier </h2>

Here's the digit classifier from Clement Farabet (https://github.com/torch/demos)
with some little changes to make the user see the result of the training. 
You will need Lua and Torch7 framework (<a href="torch.ch">torch.ch</a>) in order to run the example above. 

N.B. You will also need all the rocks required, in particularly : gfx.js for the visualization. The other comes preinstalled with Torch7. 
For gfx just do: (see https://github.com/clementfarabet/gfx.js)
> luarocks install gfx.js 


<h3> Usage </h3>
First of all you have to train the network. 
Training: 
> th train-on-mnist.lua

Running with visualization: 
First of all open a terminal tab and do : 
> luajit -l gfx.go

this will launch gfx.js server for the image rendering (needed)

then simply do: 
> th train-on-mnist.lua -v 

This will let you visualize every image tested with a label in the left corner telling you what the predicted value of the network is. 
