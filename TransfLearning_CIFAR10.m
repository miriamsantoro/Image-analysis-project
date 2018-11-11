% Clear Command Window and Workspace
clear;
clc;

categories = {'Dog','Cat'};

rootFolder = 'cifar10/cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain2;

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

net = alexnet;
%%
analyzeNetwork(net)

%%
layersTransfer = net.Layers(1:end-3);

numClasses = 2;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',400, ...
    'Verbose',true, ...
    'VerboseFrequency', 400, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'auto');

netTransfer_CIFAR10 = trainNetwork(imdsTrain,layers,options);

save netTransfer_CIFAR10
%%
load netTransfer_CIFAR10;
analyzeNetwork(netTransfer_CIFAR10);
%%
load netTransfer_CIFAR10;
rootFolder2 = 'cifar10/cifar10Test';
imdsTest = imageDatastore(fullfile(rootFolder2, categories), ...
    'LabelSource', 'foldernames');
imdsTest.ReadFcn = @readFunctionTrain2;

labels = classify(netTransfer_CIFAR10, imdsTest);
accuracy = mean(labels == imdsTest.Labels)

figure
plotconfusion(imdsTest.Labels, labels)

%%
%Display some testing images
for i = 1:10
    ii = randi(2000);
    im = imread(imdsTest.Files{ii});
    figure
    imshow(im);
    if labels(ii) == imdsTest.Labels(ii)
       colorText = 'g'; 
    else
        colorText = 'r';
    end
    title(char(labels(ii)),'Color',colorText);
end
