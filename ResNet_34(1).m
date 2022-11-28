clear all;
load ast_11650_19.mat;
%load G_ast_19_8.mat;
%load ast_19_8.mat;
%load pdb_19_8.mat;
%load G_12_19_8.mat;
%load G_13_19_8.mat;
%load G_14_19_8.mat;
load casp13_19.mat;
load casp14_19.mat;
%load casp9_19_8.mat;
%load casp10_19_8.mat;
%load casp11_19_8.mat;
load casp12_19_8.mat;



layers = [
    imageInputLayer([40 19 1],'Name','input')
     
     convolution2dLayer(5,64,'Stride',1,'Padding','same','Name','conv_frist')
     maxPooling2dLayer(3,'Stride',2,'Name','max')
     batchNormalizationLayer('Name','BN_frist')
     reluLayer('Name','re_frist')
     
     convolutionalUnit3(64,1,'S1U1')
    % reluLayer('Name','re1_1')
     additionLayer(2,'Name','ad1_1')
     convolutionalUnit3(64,1,'S1U2')
     additionLayer(2,'Name','ad1_2')
     %reluLayer('Name','re1_2')
     convolutionalUnit3(64,1,'S1U3')
     %reluLayer('Name','re1_2')
     additionLayer(2,'Name','ad1_3')
    
     
     convolutionalUnit4(128,1,'S2U1')
     %reluLayer('Name','re2_1')
     additionLayer(2,'Name','ad2_1')
     convolutionalUnit3(128,1,'S2U2')
     additionLayer(2,'Name','ad2_2')
     %reluLayer('Name','re2_2')
     convolutionalUnit3(128,1,'S2U3')
     additionLayer(2,'Name','ad2_3')
     %reluLayer('Name','re2_2')
     convolutionalUnit3(128,1,'S2U4')
     %reluLayer('Name','re2_2')
     additionLayer(2,'Name','ad2_4')
     
     
     convolutionalUnit4(256,1,'S3U1')
     %reluLayer('Name','re3_1')
     additionLayer(2,'Name','ad3_1')
     convolutionalUnit3(256,1,'S3U2')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad3_2')
     % convolutionalUnit(256,2,'S3U1')
     convolutionalUnit3(256,1,'S3U3')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad3_3')
     convolutionalUnit3(256,1,'S3U4')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad3_4')
     convolutionalUnit3(256,1,'S3U5')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad3_5')
     convolutionalUnit3(256,1,'S3U6')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad3_6')
  %
     convolutionalUnit4(512,1,'S4U1')
     %reluLayer('Name','re3_1')
     additionLayer(2,'Name','ad4_1')
     convolutionalUnit3(512,1,'S4U2')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad4_2')
     convolutionalUnit3(512,1,'S4U3')
     %reluLayer('Name','re3_2')
     additionLayer(2,'Name','ad4_3')
     % convolutionalUnit(256,2,'S3U1')
     averagePooling2dLayer(1,'Name','pooling')
     
     %maxPooling2dLayer(2,'Stride',1,'Name','max')
    
     fullyConnectedLayer(8,'Name','fully')
     softmaxLayer('Name','soft')
     classificationLayer('Name','classoutput')
 ];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
     ...
    'ValidationFrequency',512, ...
    'Verbose',false);
%analyzeNetwork(lgraph)

skipConv2 = convolution2dLayer(1,128,'stride',2,'Padding','same','Name','skip2');
skipConv3 = convolution2dLayer(1,256,'stride',2,'Padding','same','Name','skip3');
skipConv4 = convolution2dLayer(1,512,'stride',2,'Padding','same','Name','skip4');
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,skipConv3);
lgraph = addLayers(lgraph,skipConv2);
lgraph = addLayers(lgraph,skipConv4);
lgraph = connectLayers(lgraph,"re_frist","ad1_1/in2");
lgraph = connectLayers(lgraph,"ad1_1","ad1_2/in2");
lgraph = connectLayers(lgraph,"ad1_2","ad1_3/in2");
%lgraph = connectLayers(lgraph,"skip1","ad1/in2");
lgraph = connectLayers(lgraph,"ad1_3","skip2");

lgraph = connectLayers(lgraph,"skip2","ad2_1/in2");
lgraph = connectLayers(lgraph,"ad2_1","ad2_2/in2");
lgraph = connectLayers(lgraph,"ad2_2","ad2_3/in2");
lgraph = connectLayers(lgraph,"ad2_3","ad2_4/in2");
lgraph = connectLayers(lgraph,"ad2_4","skip3");

lgraph = connectLayers(lgraph,"skip3","ad3_1/in2");
lgraph = connectLayers(lgraph,"ad3_1","ad3_2/in2");
lgraph = connectLayers(lgraph,"ad3_2","ad3_3/in2");
lgraph = connectLayers(lgraph,"ad3_3","ad3_4/in2");
lgraph = connectLayers(lgraph,"ad3_4","ad3_5/in2");
lgraph = connectLayers(lgraph,"ad3_5","ad3_6/in2");
lgraph = connectLayers(lgraph,"ad3_6","skip4");

lgraph = connectLayers(lgraph,"skip4","ad4_1/in2");
lgraph = connectLayers(lgraph,"ad4_1","ad4_2/in2");
lgraph = connectLayers(lgraph,"ad4_2","ad4_3/in2");
%lgraph = connectLayers(lgraph,"ad4_1","ad4_2/in2");

net = trainNetwork(ast_11650_19,dssps_ast11650,lgraph,options);                                                                                                                                            


%YPred_pdb = classify(net,pdb_19);
%YValidation_pdb = dssps_pdb;
%accuracy_pdb= sum(YPred_pdb == YValidation_pdb)/numel(dssps_pdb);

YPred_cb = classify(net,cb_19);
YValidation_cb = dssps_cb;
accuracy_cb= sum(YPred_cb == YValidation_cb)/numel(dssps_cb);

%YPred_9 = classify(net,casp9_19);
%YValidation_9 = dssps_9;
%accuracy_9 = sum(YPred_9 == YValidation_9)/numel(dssps_9);

YPred_10 = classify(net,casp10_19);
YValidation_10 = dssps_10;
accuracy_10 = sum(YPred_10 == YValidation_10)/numel(dssps_10);

YPred_11 = classify(net,casp11_19);
YValidation_11 = dssps_11;
accuracy_11 = sum(YPred_11 == YValidation_11)/numel(dssps_11);

YPred_12 = classify(net,casp12_19);
YValidation_12 = dssps_12;
accuracy_12 = sum(YPred_12 == YValidation_12)/numel(dssps_12);

YPred_13 = classify(net,casp13_19);
YValidation_13 = dssps_13;
accuracy_13(1) = sum(YPred_13 == YValidation_13)/numel(dssps_13);

YPred_14 = classify(net,casp14_19);
YValidation_14 = dssps_14;
accuracy_14 = sum(YPred_14 == YValidation_14)/numel(dssps_14);