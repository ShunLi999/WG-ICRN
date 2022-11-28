function layers = convolutionalUnit2(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',2*stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu'])
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])
    
    ];
end