function [map1] = nn_fw_l2 (input, inout_w)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel forward in the networks
% input: a matrix of two dimensions (numbercase*basic_kernel_size)
% inhid_weights: a matrix of weights between input and hidden layer
% hidout_weights: a matrix of weights between hidden and output layer
% output: a matrix of kernel values

% Mingyuan Jiu (mingyuan.jiu@telecom-paristech.fr)
% Beg: 19/10/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numcase = size(input, 1);

% input = [input, ones(numcase, 1)];

map1 = exp ( input * inout_w ); 