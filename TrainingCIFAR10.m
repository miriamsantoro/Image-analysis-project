% Clear Command Window and Workspace
clear;
clc;

% Please note: these are 2 of the 10 categories available
% Feel free to choose which ever you like best!
categories = {'Dog','Cat'};

rootFolder = 'cifar10/cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

%%
varSize = 32;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(2,'BiasLearnRateFactor',2);
fc2.Weights = gpuArray(single(randn([2 64])*0.1));

layers = [
    imageInputLayer([varSize varSize 3], 'Name', 'input');
    conv1;
    maxPooling2dLayer(3,'Stride',2, 'Name', 'max_pool');
    reluLayer('Name', 'relu_1');
    convolution2dLayer(5,48,'Padding',2,'BiasLearnRateFactor',2, 'Name', 'conv_2');
    reluLayer('Name', 'relu_2');
    averagePooling2dLayer(3,'Stride',2, 'Name', 'avg_pool_1');
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2, 'Name', 'conv_3');
    reluLayer('Name', 'relu_3');
    averagePooling2dLayer(3,'Stride',2, 'Name','avg_pool_2');
    fc1;
    reluLayer('Name', 'relu_4');
    fc2;
    softmaxLayer('Name', 'softmax');
    classificationLayer('Name', 'classification')];

opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 5, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize', 10, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',400, ...
    'Verbose', true, ...
    'VerboseFrequency', 400, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'auto');

[net, info] = trainNetwork(imdsTrain, layers, opts);

cifar10_net= net;
save cifar10_net

%%
load cifar10_net;
analyzeNetwork(cifar10_net)
%%
load cifar10_net;
rootFolder2 = 'cifar10/cifar10Test';
imdsTest = imageDatastore(fullfile(rootFolder2, categories), ...
    'LabelSource', 'foldernames');
imdsTest.ReadFcn = @readFunctionTrain;

labels = classify(cifar10_net, imdsTest);
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
