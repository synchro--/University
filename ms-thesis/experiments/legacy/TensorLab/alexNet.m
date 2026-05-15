%% AlexNet is a pretrained Convolutional Neural Network (CNN) that has been trained on approximately 1.2 million images from the ImageNet Dataset (http://image-net.org/index). The model has 23 layers and can classify images into 1000 object categories (e.g. keyboard, mouse, coffee mug, pencil). 
% Opening the alexnet.mlpkginstall file from your operating system or from within MATLAB will initiate the installation process for the release you have. 
% This mlpkginstall file is functional for R2016b and beyond. 
% Usage Example: 

% Access the trained model 
net = alexnet 
% See details of the architecture 
net.Layers 
% Read the image to classify 
I = imread('peppers.png'); 
% Adjust size of the image 
sz = net.Layers(1).InputSize 
I = I(1:sz(1),1:sz(2),1:sz(3)); 
% Classify the image using AlexNet 
label = classify(net, I) 
% Show the image and the classification results 
figure 
imshow(I) 
text(10,20,char(label),'Color','white')