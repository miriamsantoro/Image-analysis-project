% Clear Command Window and Workspace
clear;
clc;

categories = {'Dog','Cat'};

rootFolder = 'cifar10/cifar10Train';
imdsTrain = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
imdsTrain.ReadFcn = @readFunctionTrain2;

rootFolder = 'cifar10/cifar10Test';
imdsTest = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
imdsTest.ReadFcn = @readFunctionTrain2;

%%
tic;
net = alexnet;
layer = 'fc7';
featuresTrain = activations(alexnet,imdsTrain,layer,'OutputAs','rows');
featuresTest = activations(alexnet,imdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain,YTrain);
toc;

%%
%Display some images
YPred = predict(classifier,featuresTest);

for i = 1:10
    ii=randi(2000);
    I = imread(imdsTest.Files{ii});
    figure
    imshow(I);
    if YPred(ii) == YTest(ii)
       colorText = 'g'; 
    else
        colorText = 'r';
    end
    title(char(YPred(ii)),'Color',colorText);
end

accuracy = mean(YPred == YTest)

figure
plotconfusion(YTest, YPred)

%%
%Show activations in first convolutional layer
figure
M = readimage(imdsTrain,2);
imshow(M)
act1=activations(net,M,'conv1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);

figure
L = montage(mat2gray(act1))

[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);

%Show strongest channel
figure
N = montage({M,act1chMax});
