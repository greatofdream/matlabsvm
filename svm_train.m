function [alpha, rho, energy, c_penalty] = svm_train(train_kernel, train_lab, c_penalty)

% SVM training function
% 
% Author: Mingyuan Jiu (mingyuan.jiu@telecom-paristech.fr)
% Log: 2014/08/25 --- Beg


if nargin<3
   c_penalty = 0; 
end
    
tra_size = size(train_kernel, 1);
pos_size = sum(train_lab==1);
neg_size = sum(train_lab==-1);

pos_w = tra_size / pos_size;
neg_w = tra_size / neg_size; 
if isinf(pos_w) || isnan(pos_w)
    pos_w = 1;
end
if isinf(neg_w) || isnan(neg_w)
    neg_w = 1;
end

if pos_w==neg_w
    pos_w = 1; 
    neg_w = 1; 
end

train_kernel = cat(2, (1:tra_size)', train_kernel);      % for libsvm

%-------------------------
% parameter selection C 
%-------------------------
if c_penalty == 0 
    acc_best = -1;
    for c_lev = -10 : 10
        c_param = 2^c_lev;
        % ' -w-1 ' num2str(neg_w) ' -w1 ' num2str(pos_w)
        libsvm_train_options = ['-s 0 -t 4 -c ' num2str(c_param) ' -w-1 ' num2str(neg_w) ' -w1 ' num2str(pos_w) ' -q'];
        svm_model = svmtrain(train_lab, train_kernel, libsvm_train_options);
        
        [~, accu_val, ~] = svmpredict( train_lab, train_kernel(:, 2:end), svm_model, '-q');                             
        tmp_acc = accu_val(1);        
        if tmp_acc > acc_best
            c_penalty = c_param;
            acc_best = tmp_acc;
        end
    end
end
        
libsvm_train_options = ['-s 0 -t 4 -c ' num2str(c_penalty) ' -w-1 ' num2str(neg_w) ' -w1 ' num2str(pos_w) ' -q'];
svm_model = svmtrain(train_lab, train_kernel, libsvm_train_options);

if isempty(svm_model)
    fprintf('empty svm \n');
end

% computed energy
alpha = zeros(tra_size, 1);
for i = 1 : svm_model.totalSV
    cur_idx = svm_model.sv_indices(i);
    alpha(cur_idx) = svm_model.sv_coef(i);
end
nlab = svm_model.Label; 
rho = svm_model.rho; 
if nlab(1)==-1
    alpha=-alpha;
else
    rho=-rho;
end

% delete the first column for traing kernel
train_kernel = train_kernel(:, 2:end);

% the current energy at x
deltaoutker = - alpha * alpha'/ 2;
energy = sum(sum(deltaoutker .* train_kernel)) + train_lab' * alpha;
