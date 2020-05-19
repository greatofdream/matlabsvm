function [dinout_w] = nn_bw_l2(deltaoutker, inout_w, outker, inker)
% compute the gradients for the weights in the deep kernel framework
% inhid_w: the weights beween the input layer and hidden layer
% hidout_w: the weights betweeen the hidden layer and output layer
% outker: the kernel in the output layer
% hidker: the kernel in the hidden layer
% inker: the kernel in the input layer

% Mingyuan Jiu (mingyuan.jiu@telecom-paristech.fr)
% Beg: 19/10/2014

numcase = size(deltaoutker, 1); 

dmap1 = deltaoutker .* outker; 

% gradient of the weights in the second layer
dinout_w = inker' * dmap1; 

