function [X, accuracy, avgTime] = sc_main(Y, A, sc_algo, trainLabel, testLabel, numClass, displayProgressFlag)

% ---------------------------------------------------
% Fast Sparse Representation with Prototypes: sc_main
% Functionality: 
%       Find the approaximated sparse solution x of the linear system y=Ax
% Dimension: m --- number of measurement
%            Nt--- number of testing samples
%            Nd--- number of training samples
%
%                   Dimension          Description
% input:  Y          m x Nt        --- the testing sample
%         A          m x Nd        --- the training sample
%         trainLabel 1 x Nt        --- the label of training sample
%         sc_algo                  --- the sparse coding algorithm
%          e.g., l1magic, SparseLab, fast_sc, SL0, YALL1
% output: X          K x Nt        --- the sparse coefficient matrix of Y
%         accuracy                 --- accuracy of the classification task
%         avgTime                  --- average runtime for sparse coding
% 
% Reference: Jia-Bin Huang and Ming-Hsuan Yang, "Fast Sparse Representation with Prototypes.", the 23th IEEE Conference
%            on Computer Vision and Pattern Recognition (CVPR 10'), San Francisco, CA, USA, June 2010.
% Contact: For any questions, email me by jbhuang@ieee.org
% ---------------------------------------------------

[m Nt]= size(Y);
[m Nd]= size(A);

X = zeros(Nd, Nt);

% Compute the sparse representation X

Ainv = pinv(A);
sumTime=0;
correctSample=0;
for i = 1: Nt
    % Inital guess
    xInit = Ainv * Y(:,i);
      
    % sparse coding: solve a linear system
    tic
    xp = sparse_coding_methods(xInit, A, Y(:,i), sc_algo);
    t = toc;
    sumTime = sumTime+t;
  
    X(:, i) = xp;
    
    % Predict label of the test sample
    residuals = zeros(1,numClass);
    for iClass = 1: numClass
        xpClass = xp;
        xpClass(trainLabel~= iClass) = 0;
        residuals(iClass) = norm(Y(:,i) - A*xpClass);
    end
    [val, ind] = min(residuals);
    if(ind==testLabel(i))
        correctSample = correctSample+1;
    end
    if(displayProgressFlag)
        avgTime = sumTime/i;
        accuracy = correctSample / i;
        fprintf('Accuracy = %f %% (%d out of %d), speed = %f s\n', accuracy*100, correctSample, i, avgTime);
    end
end
accuracy = correctSample/Nt;
avgTime=sumTime/Nt;

end