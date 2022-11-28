function layers = convolutionalUnit3(numF,stride,tag)
layers = [
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN2'])
    reluLayer('Name',[tag,'relu2'])
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv2'])
    ];
end