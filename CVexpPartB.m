 clear all
 close all
 clc

%% sec 1.1
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

%% sec 1.2
imds = imageDatastore(imageDir);

%% sec 1.3
I = readimage(imds,1);
figure();
imshow(I);

%% sec 1.4
classNames = ["triangle","background"];
labelIDs = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%% sec 1.5
C = (uint8(readimage(pxds,1))-1)*255;
figure();
imshow(C);

I2 = readimage(imds,2);
figure();
imshow(I2);
C2 = (uint8(readimage(pxds,2))-1)*255;
figure();
imshow(C2);
I3 = readimage(imds,3);
figure();
imshow(I3);
C3 = (uint8(readimage(pxds,3))-1)*255;
figure();
imshow(C3);

%% sec 1.6
layers = [
imageInputLayer([32 32 1])
convolution2dLayer(3,8,'Padding',1)
reluLayer()
convolution2dLayer(3,16,'Padding',1)
reluLayer()
convolution2dLayer(1,2);
softmaxLayer()
pixelClassificationLayer()];

%% sec 1.7
opts = trainingOptions('sgdm', ...
'InitialLearnRate', 2e-4, ...
'MaxEpochs', 100, ...
'MiniBatchSize', 64, ...
'ExecutionEnvironment', 'cpu','Plots', 'training-progress');



%% sec 1.8
trainingData = pixelLabelImageSource(imds,pxds);

%% sec 1.9
net = trainNetwork(trainingData,layers,opts);

%% sec 1.10
testImage = imread('triangleTest.jpg');
imshow(testImage);
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure; imshow(B)

%% sec 1.15
testImagesDir = fullfile(dataSetDir, 'testImages');
imdsTest = imageDatastore(testImagesDir);
testLabelsDir = fullfile(dataSetDir, 'testLabels');
pxdsTruthTest = pixelLabelDatastore(testLabelsDir, classNames, labelIDs);
pxdsResults = semanticseg(imdsTest, net, "WriteLocation", tempdir);
evaluationMetrics = ["accuracy"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruthTest, "Metrics", evaluationMetrics);

%% sec 1.16
tbl = countEachLabel(trainingData)

%% sec 1.17
layers(end) = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',[10,1]);
net = trainNetwork(trainingData,layers,opts);
testImage = imread('triangleTest.jpg');
imshow(testImage);
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure; imshow(B)
 

%% sec 1.18
testImagesDir = fullfile(dataSetDir, 'testImages');
imdsTest = imageDatastore(testImagesDir);
testLabelsDir = fullfile(dataSetDir, 'testLabels');
pxdsTruthTest = pixelLabelDatastore(testLabelsDir, classNames, labelIDs);
pxdsResults = semanticseg(imdsTest, net, "WriteLocation", tempdir);
evaluationMetrics = ["accuracy"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruthTest, "Metrics", evaluationMetrics);

%% sec 1.19
evaluationMetrics = ["accuracy" "iou"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruthTest, "Metrics",evaluationMetrics);

%% sec 1.21

dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');


imds = imageDatastore(imageDir);

I = readimage(imds,1);
figure();
imshow(I);


classNames = ["triangle","background"];
labelIDs = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
layers = [
imageInputLayer([32 32 1])



convolution2dLayer(3,8,'Padding',1)
reluLayer()
convolution2dLayer(3,16,'Padding',1)
reluLayer()

convolution2dLayer(3,32,'Padding',2)
reluLayer()

convolution2dLayer(5,48,'Padding',1)
reluLayer()

convolution2dLayer(1,2);
softmaxLayer()
pixelClassificationLayer()];

opts = trainingOptions('sgdm', ...
'InitialLearnRate', 2e-4, ...
'MaxEpochs', 100, ...
'MiniBatchSize', 64, ...
'ExecutionEnvironment', 'cpu','Plots', 'training-progress');


trainingData = pixelLabelImageSource(imds,pxds);

net = trainNetwork(trainingData,layers,opts);


testImage = imread('triangleTest.jpg');
imshow(testImage);
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure; imshow(B)


testImagesDir = fullfile(dataSetDir, 'testImages');
imdsTest = imageDatastore(testImagesDir);
testLabelsDir = fullfile(dataSetDir, 'testLabels');
pxdsTruthTest = pixelLabelDatastore(testLabelsDir, classNames, labelIDs);
pxdsResults = semanticseg(imdsTest, net, "WriteLocation", tempdir);
evaluationMetrics = ["accuracy"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruthTest, "Metrics", evaluationMetrics);
%% sec 1.222
evaluationMetrics = ["accuracy" "iou"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruthTest, "Metrics",evaluationMetrics)


 %% sec 2.1
 baseDir = '.\CV_exp\CV_exp';
 addpath(fullfile(baseDir,'functions'));
 pretrainedSegNet = fullfile(baseDir,'segnetVGG16CamVid.mat');
 imDir = fullfile(baseDir,'Raw');
 labelDir = fullfile(baseDir,'Labeled');
 labelIDs = camvidPixelLabelIDs;
 classes = SegNetClasses;
 cmap = camvidColorMap;
 imds = imageDatastore(imDir);
 pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%% sec 2.2
I = readimage(imds, 72);
figure; imshow(I)
C = readimage(pxds, 72);

B = labeloverlay(I,C,'ColorMap',cmap);
figure; imshow(B)
pixelLabelColorbar(cmap,classes);

I = readimage(imds, 534);
figure; imshow(I)
C = readimage(pxds, 534);

B = labeloverlay(I,C,'ColorMap',cmap);
figure; imshow(B)
pixelLabelColorbar(cmap,classes);

%% sec 2.3
lgraph = segnetLayers([360,480,3],11,'vgg16');

%% sec 2.4
lgraph.Layers

%% sec 2.5
data = load(pretrainedSegNet);
net = data.net;

%% sec 2.6
I1 = readimage(imds, 72);
I1_rs=imresize(I1,[360,480]);
figure; imshow(I1_rs)

C1 = readimage(pxds,72);
B1=labeloverlay(I1,C1,'Colormap',cmap);
figure;imshow(B1)
pixelLabelColorbar(cmap,classes);



I2 = readimage(imds, 300);
I2_rs=imresize(I2,[360,480]);
figure; imshow(I2_rs)

C2 = readimage(pxds,300);
B2=labeloverlay(I2,C2,'Colormap',cmap);
figure;imshow(B2)
pixelLabelColorbar(cmap,classes);



I3 = readimage(imds, 680);
I3_rs=imresize(I3,[360,480]);
figure; imshow(I3_rs)

C3 = readimage(pxds,680);
B3=labeloverlay(I3,C3,'Colormap',cmap);
figure;imshow(B3)
pixelLabelColorbar(cmap,classes);

%% sec 2.8
videoWriterOBJ =vision.VideoFileWriter('video.mp4','FileFormat','MPEG4','FrameRate',2);
for ii=150:200
 img = imresize(readimage(imds,ii),[360 480]);
 C = semanticseg(img, net);
 img_segmented = labeloverlay(img,C,'ColorMap',cmap);
 img_both = [img,zeros(360,5,3),img_segmented];
 step(videoWriterOBJ,img_both);
end
release(videoWriterOBJ);

implay('video.mp4');


% sec 3.1
datasetPath_train = '.\CV_exp\CV_exp\fashionMNIST\trainImages';
imds_train = imageDatastore(datasetPath_train, ...
'IncludeSubfolders',true,'LabelSource','foldernames');
datasetPath_test = '.\CV_exp\CV_exp\fashionMNIST\testImages';
imds_test = imageDatastore(datasetPath_test, ...
'IncludeSubfolders',true,'LabelSource','foldernames');

% sec 3.2
figure()
for i=1:30
    index = randi(60000);
    subplot(6,5,i)
    imshow(readimage(imds_train, index));
end

% sec 3.3
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding',1)
    reluLayer()
    convolution2dLayer(3,16,'Padding',1)
    reluLayer()
    convolution2dLayer(3,24,'Padding',1)
    reluLayer()
    convolution2dLayer(3,32,'Padding',1)
    reluLayer()
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];
    
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 1000, ...
    'ExecutionEnvironment', 'cpu','Plots', 'training-progress');
% sec 3.4

net = trainNetwork(imds_train,layers,opts);
% sec 3.5
YPred = classify(net,imds_test,'ExecutionEnvironment','cpu');

loss_percent= 100-((sum(YPred == imds_test.Labels).*100)./length(imds_test.Labels))

% sec 3.6

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding',1)
    reluLayer()
    convolution2dLayer(3,16,'Padding',1)
    reluLayer()
    convolution2dLayer(3,24,'Padding',1)
    reluLayer()
    convolution2dLayer(3,32,'Padding',1)
    reluLayer()
    
    convolution2dLayer(3,48,'Padding',1)
    reluLayer()
    
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];
    
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 1000, ...
    'ExecutionEnvironment', 'cpu','Plots', 'training-progress');


net = trainNetwork(imds_train,layers,opts);

% sec 3.61
YPred = classify(net,imds_test,'ExecutionEnvironment','cpu');

loss_percent= 100-((sum(YPred == imds_test.Labels).*100)./length(imds_test.Labels))
% sec 3.7
figure()
for i=1:30
    index = randi(10000);
    subplot(6,5,i)
    imshow(readimage(imds_test, index));
    title(char(YPred(index)));
end

