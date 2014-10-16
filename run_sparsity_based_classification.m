% ---------------------------------------------------
% Fast Sparse Representation with Prototypes: run_sparsity-based classification 
% Functionality: 
%       Script for testing sparsity-based classification
%
% Reference: Jia-Bin Huang and Ming-Hsuan Yang, "Fast Sparse Representation with Prototypes.", the 23th IEEE Conference
%            on Computer Vision and Pattern Recognition (CVPR 10'), San Francisco, CA, USA, June 2010.
% Contact: For any questions, email me by jbhuang@ieee.org
% ---------------------------------------------------

%% Add path
addpath(genpath('tools/'));
addpath(genpath('data/'));


%% Load training and testing data

% dataset
% yale: The Extended Yale Face Database B
% AR: AR Face Database
% COIL-100: Columbia Object Image Library
% Digit recognition

dataset = 'yale';
fileExt = 'png';
imgSide = 12;

datasetPath = ['data/',dataset];
imgDataDir = dir([datasetPath, '*.',fileExt]);


percentTraining = 0.5;
normalization   = true;

fprintf('Loading and sperating training and testing samples ...\n');
[trainLabel, testLabel, trainSample, testSample, numClass] = getTrainTestData(datasetPath, fileExt, imgSide, percentTraining, normalization);

% Representation
% Downsample, Random projection, PCA, LDA, 


%% Learning dictionary from the training samples

% Parameters for dictionary learning
redundencyFactor = 2;                           % The number of the atoms = data dimension x redundencyFactor 
param.L = 10;                                   % The number of atoms used in representation a signal
param.InitializationMethod =  'DataElements';   % Initialize a dictionary with random sampling
param.errorFlag = 0;                            % Decompose signal without reaching an error bound
param.K = redundencyFactor*size(trainSample,1); % The number of atoms in the dictionary
param.numIteration = 5;                         % The number iteration for the K-SVD algorithm 
param.preserveDCAtom = 0;                       % Presearve a DC atom or not
param.displayProgress= 1;                       % Display the progress and the error at each iteration

% K-SVD dictionary learning
fprintf('K-SVD dictionary learning...\n');
[Dictionary,output] = KSVD(trainSample, param);

%% Sparsity coding based classification

sc_algo= 'l1magic';         % Select one sparse coding method
displayProgressFlag = 1;    % Display the prediction progress
FastSparseCodingFag = 1;    % Use fast sparse coding
fprintf('Solving sparse coding...\n');

if(FastSparseCodingFag)
    [X, accuracy, avgTime] = fsc_main(testSample, trainSample, Dictionary, param.L, sc_algo, trainLabel, testLabel, numClass, displayProgressFlag);
else
    [X, accuracy, avgTime] = sc_main(testSample, trainSample, sc_algo, trainLabel, testLabel, numClass, displayProgressFlag);
end

% Report the classification accuracy and the speed

fprintf('Sparse coding method: %s.\n', sc_algo);
fprintf('The accuracy of the classification task is %f %%.\n', accuracy*100);
fprintf('The average speed of sparse coding is %s second.\n', avgTime);
