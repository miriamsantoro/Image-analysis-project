% This function simply resizes the images to fit in AlexNet
% Copyright 2017 The MathWorks, Inc.
function I = readFunctionTrain2(filename)
% Resize the images to the size required by the network.
I = imread(filename);

I = imresize(I, [227 227]);