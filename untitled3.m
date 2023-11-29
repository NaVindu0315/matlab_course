% Load the dataset
load fisheriris.mat;

% Report the total number of rows
totalRows = size(meas, 1);
fprintf('Total number of rows (objects, cases): %d\n\n', totalRows);

% Compute and display statistics for each column
for col = 1:4
    columnData = meas(:, col);
    displayColumnStatistics(col, columnData);
end

% Task 2.2 - Neural Network Setup
% Shuffle the dataset randomly
rng(1); % set random seed for reproducibility
indices = randperm(totalRows);
trainPercentage = 0.6;
trainIndices = indices(1:round(trainPercentage * totalRows));
testIndices = indices(round(trainPercentage * totalRows) + 1:end);

% Create training and testing datasets
[trainData, testData, trainTarget, testTarget] = createDatasets(meas, species, trainIndices, testIndices);

% Select Feedforward Recognition Neural Networks
% Define the hidden layer size for NN using iteration
hiddenLayerSizes = [10];

for hiddenSize = hiddenLayerSizes
    fprintf('\nHidden Layer Size: %d\n', hiddenSize);
    
    for iteration = 1:3 % Repeat the experiment 3 times
        fprintf('Iteration: %d\n', iteration);
        
        % Train and evaluate the neural network
        [net, accuracy] = trainAndEvaluateNN(trainData, trainTarget(:, 1), testData, testTarget(:, 1), hiddenSize);
        
        fprintf('Classifier Accuracy: %.2f%%\n\n', accuracy);
    end
end

% Function to display column statistics
function displayColumnStatistics(columnIndex, columnData)
    fprintf('Column %d:\n', columnIndex);
    fprintf('  Mean: %f\n', mean(columnData));
    fprintf('  Standard Deviation: %f\n', std(columnData));
    fprintf('  Maximum: %f\n', max(columnData));
    fprintf('  Minimum: %f\n', min(columnData));
    fprintf('  Root Mean Square: %f\n\n', rms(columnData));
end

% Function to create training and testing datasets
function [trainData, testData, trainTarget, testTarget] = createDatasets(data, labels, trainIndices, testIndices)
    trainData = data(trainIndices, :);
    testData = data(testIndices, :);
    trainTarget = dummyvar(categorical(labels(trainIndices)));
    testTarget = dummyvar(categorical(labels(testIndices)));
end

% Function to train and evaluate the neural network
function [net, accuracy] = trainAndEvaluateNN(trainData, trainTarget, testData, testTarget, hiddenSize)
    net = feedforwardnet(hiddenSize);
    net.layers{end}.size = 1;

    % Train the neural network
    net.trainParam.showWindow = true;
    net = train(net, trainData', trainTarget');

    % Test the trained net
    testOutput = net(testData');
    predictedLabels = round(testOutput);
    
    % Evaluate performance
    correctClassifications = sum(all(testTarget == predictedLabels, 2));
    accuracy = correctClassifications / size(testTarget, 1) * 100;
end
