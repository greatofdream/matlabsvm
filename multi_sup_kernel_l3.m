% This file is to train a construct a gram matrix and use it to train a
% kernel

% **********************************************************************
% Mingyuan Jiu, mingyuan.jiu@telecom-paristech.fr
%
% Change log:
% 1.08.2014 mjiu: -Begin
% **********************************************************************

clear;
% clc;
% External utiliaries must be on path
if isempty(strfind(path,'../libsvm-3.18'))
    addpath('../libsvm-3.18');
end
if isempty(strfind(path, '../libsvm-3.18/matlab'))
    addpath('../libsvm-3.18/matlab');
end
if isempty(strfind(path, '../network'))
    addpath('../network');
end
if isempty(strfind(path, '../utils'))
    addpath('../utils');
end

lambta = 1; 

learn_file = ['learn_msup_l3_0_10sub_0917_500_3f_1.mat'];    % num2str(tt)

stime=RandStream.create('mrg32k3a', 'seed', 123455); % sum(100*clock)
RandStream.setGlobalStream(stime);

load ../running_data/mulinput_dev_data_10sub_0629_500_3finfo.mat;
global_laplacian = full(global_laplacian); 
basic_kernel_size = 60;
uns_size = all_size;   
conNum = length(devConList); 
conid = (1 : conNum); 

% load in_kernel 
load ../running_data/in_kernel_0917.mat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize the deep kernel network and svm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nlev = 3;                               % three layers: input hiddel and outpu layer,
in_size = basic_kernel_size;                           % input layer size   
hid_size = 2*basic_kernel_size;                          % hidde layer size 
out_size = 1;                           % output layer size
inhid_w = 0.02*(randn(in_size, hid_size, 'single'));
inhid_w(inhid_w<0) = 0; 
hidout_w = 0.002*(rand(hid_size, out_size, 'single'));
final_models.kernel_model = {inhid_w, hidout_w}; 

% svm number
svm_num = size(sampled_train_idx, 2); 

% SCO_matrix = zeros(conNum, test_size);      % score
% % determine the C values for different class using the training data
% c_vec = zeros(conNum, svm_num); 
% svm_models = cell(conNum, 4, svm_num);
% mean_kernel = double(mean(in_kernel, 2)); 
% mean_kernel = reshape(mean_kernel, all_size, all_size); 
% for nConcept = 1:conNum  
%     for nSvm = 1:svm_num
%     
%     id = conid(nConcept); 
%     % get the train index of the training instances
%     cur_train_idx = sampled_train_idx{id, nSvm}; 
%     cur_sup_size = length(cur_train_idx); 
%     
%     % reshape the training and test as [training idx; test idx]
%     cur_idx = [cur_train_idx; test_idx];
%     cur_all_size = length(cur_idx); 
%     cur_sup_idx = 1 : cur_sup_size; 
%     cur_test_idx = cur_sup_size+1 : cur_all_size; 
%     
%     indicator = dev_gt_mat(id, cur_train_idx)'; 
%     train_lab = ones(length(indicator), 1); 
%     train_lab(~indicator) = -1;  
% 
%     out_kernel = mean_kernel(cur_idx, cur_idx); 
%     sup_out_kernel = out_kernel(cur_sup_idx, cur_sup_idx); 
%     
%     % train one vs rest svm
%     [~, ~, c_penalty] = svm_train_validate(sup_out_kernel, train_lab);  
%     [alpha, rho, energy, c_penalty] = svm_train(sup_out_kernel, train_lab, c_penalty);   
%     svm_models{nConcept, 1, nSvm} = single(alpha);
%     svm_models{nConcept, 2, nSvm} = single(rho); 
%     svm_models{nConcept, 3, nSvm} = single(energy);
%     svm_models{nConcept, 4, nSvm} = single(c_penalty);
%     
%     c_vec(nConcept, nSvm) = c_penalty;
%     fprintf('%d, pos: %d, neg: %d, C: %f\n', id, sum(train_lab==1), sum(train_lab==-1),...
%            c_penalty);
%        
%     % predict the test   
%     test_out_kernel = out_kernel(cur_test_idx, cur_sup_idx); 
%     dec_values = test_out_kernel * alpha + rho; 
%     SCO_matrix( nConcept, : ) = ( SCO_matrix( nConcept, : ) + double(dec_values') );
%     
%     end 
% end
% 
% SCO_matrix = SCO_matrix / svm_num;
% DEC_matrix = zeros(conNum, test_size); 
% for jj = 1 : test_size
%     [~, idx] = sort( SCO_matrix(:,jj) );  
%     DEC_matrix(idx(end-4:end), jj) = 1; 
% end
% DEC_matrix = (DEC_matrix==1); 
% cnptRES = zeros(conNum, 3); 
% %%% Measures per concept %%%
% cnptRES(:,1) = sum(DEC_matrix&test_lab_mat,2)./sum(DEC_matrix,2); %%% Precision %%%
% cnptRES(~isfinite(cnptRES(:,1)),1) = 0;
% cnptRES(:,2) = sum(DEC_matrix&test_lab_mat,2)./sum(test_lab_mat,2); %%% Recall %%%
% cnptRES(~isfinite(cnptRES(:,2)),2) = 0;
% cnptRES(:,3) = 2*(cnptRES(:,1).*cnptRES(:,2))./(cnptRES(:,1)+cnptRES(:,2)); %%% F-measure %%%
% cnptRES(~isfinite(cnptRES(:,3)),3) = 0;    
% results = [mean(cnptRES, 1)*100, sum(cnptRES(:, 2)>0)]; 
% 
% save multi_c_validated_10sub_0917_500_3f.mat c_vec results;
% fprintf('validate finish\n');

load multi_c_validated_10sub_0917_500_3f.mat; 

%-----------------------------
% define the iteration
iter = 300;
energy_list = zeros(iter, 1); 
learn_models = cell(iter, 1); 
final_te_res = cell(iter, 4); 
learn_te_res = []; 

mu1 = 2*1e-4; 
mu2 = 1*1e-7;     % learning rate; 
iter_observed = 100;
momentum = 0.9;

dninhid_w_old = 0;
dnhidout_w_old = 0; 

% % reshape the dev pre-kernels
% in_kernel = permute(dev_pre_kernels, [3 1 2]); 
% in_kernel = reshape(in_kernel, basic_kernel_size, all_size*all_size); 
% in_kernel = in_kernel';
% clear dev_pre_kernels; 

samples_size = 500; 
samples_idx = randSample(all_size, samples_size); 
samples = zeros(all_size, all_size);  
samples(samples_idx, samples_idx) = 1; 
samples_idx = find(samples==1); 

for it = 1 : iter
    
    fprintf('ITER: %d ', it);
    
    tStart = tic;
    % update svm models
    TRAIN_SVM_CODE
    
%     learn_models{it, 1} = final_models;
    final_te_res{it, 1} = SCO_matrix;    
    save(learn_file, 'learn_models', 'energy_list', 'final_te_res', 'learn_te_res', 'iter',...
        'mu1', 'mu2', 'iter_observed', 'momentum', '-v7.3');    
    
    all_delta = all_delta(:);
    all_delta = all_delta(samples_idx); 
    cur_in_kernel = in_kernel(samples_idx, :); 
    cur_in_kernel = cur_in_kernel(all_delta~=0, :);
    all_delta = all_delta(all_delta~=0);
    all_delta = all_delta / conNum / svm_num; 
    
    [cur_out_kernel, cur_hid_kernel] = nn_fw_l3(cur_in_kernel, inhid_w, hidout_w);
    [dinhid_w, dhidout_w] = nn_bw_l3(all_delta, inhid_w, hidout_w, cur_out_kernel, cur_hid_kernel, cur_in_kernel);  
 
    cur_mu1 = mu1 / (1 + it/iter_observed);
    cur_mu2 = mu2 / (1 + it/iter_observed);    
    dninhid_w = cur_mu1 * (dinhid_w) + momentum * dninhid_w_old;
    dnhidout_w = cur_mu2 * (dhidout_w) + momentum * dnhidout_w_old;
    
    inhid_w = inhid_w -  dninhid_w;
    inhid_w(inhid_w<0) = 0; 
    hidout_w = hidout_w -  dnhidout_w;
    hidout_w(hidout_w<0) = 0;     
    
    dninhid_w_old = dninhid_w;
    dnhidout_w_old = dnhidout_w; 
    
    final_models.kernel_model = {inhid_w, hidout_w};        
    
    save -v7.3 inter_vars.mat final_models all_delta dinhid_w dhidout_w dninhid_w dnhidout_w;    

    tElapsed = toc(tStart); 
    fprintf('cost: %f m.\n', tElapsed/60);
end

fprintf('\n');
% end the iteration
%-------------------------------
