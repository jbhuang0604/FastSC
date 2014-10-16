% Read cropped yale database
clear;
clc;

% resizeRatio: 3, 6, 12, 18, 24
DOWNSIZE_IMAGE = true;
resizeRatio = 2;

sampleCount = 1400;

if DOWNSIZE_IMAGE==false
    resizeRatio = 1;
end

databasePath = 'AR/';
matFilePath='AR_matfiles/';

directoryFiles = dir(databasePath);
if DOWNSIZE_IMAGE
    imageSize = floor(120/resizeRatio)*floor(165/resizeRatio);
else
    imageSize = 120*165;
end
dataMatrix = zeros(imageSize,sampleCount);
sampleLabels = zeros(1,sampleCount);
dataCount = 0; 
for directoryIndex=1:length(directoryFiles)
    currentFile = directoryFiles(directoryIndex).name;
    
    if length(currentFile)>5 & strcmp('.bmp', currentFile(end-3:end))
        % It is a face database directory

        disp(currentFile);

        imageOffset = str2num(currentFile(7:8));
        if imageOffset<=7 || (imageOffset>13 && imageOffset<=20)
            % Exclude disguise images
            
            dataCount = dataCount + 1;

            % Assign the label
            if currentFile(1)=='M'
                sexOffset = 0;
            else
                sexOffset = 50;
            end

            labelCount = str2num(currentFile(3:5)) + sexOffset;

            imageFullPath = [databasePath currentFile];

            image = imread(imageFullPath);
            if size(image,3)>1
                image = rgb2gray(image);
            end
            if DOWNSIZE_IMAGE
                image = imresize(image,1/resizeRatio);
            end
            dataMatrix(:,dataCount) = double(reshape(image,imageSize,1));
            sampleLabels(dataCount) = labelCount;
        end
    end
end

% Compute average face
save([matFilePath 'AR_database_',num2str(resizeRatio), '.mat'], 'dataMatrix', 'sampleLabels', 'trainingIndices', '-v6');