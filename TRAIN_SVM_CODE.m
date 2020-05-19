
SCO_matrix = zeros(conNum, test_size);      % score

% load the deep kernel model
inhid_w = final_models.kernel_model{1};   
hidout_w = final_models.kernel_model{2}; 

tmp_bt_num = 20; 
tmp_all_size = size(in_kernel, 1);
out_kernel = zeros(tmp_all_size, 1, 'single'); 
tmp_bt_size = floor(tmp_all_size / tmp_bt_num); 
for ii = 1 : tmp_bt_num
    % forward the network for all the kernels 
    if ii ~= tmp_bt_num
        cur_in_kernel = in_kernel(tmp_bt_size*(ii-1)+1:tmp_bt_size*ii, :); 
        out_kernel(tmp_bt_size*(ii-1)+1:tmp_bt_size*ii, 1) = nn_fw_l3(cur_in_kernel, inhid_w, hidout_w);
    else
        cur_in_kernel = in_kernel(tmp_bt_size*(ii-1)+1:end, :); 
        out_kernel(tmp_bt_size*(ii-1)+1:end, 1) = nn_fw_l3(cur_in_kernel, inhid_w, hidout_w);
    end
end
out_kernel = reshape(out_kernel, all_size, all_size);  

svm_models = cell(conNum, 4, svm_num); 
fx = 0;                                 % energy
all_delta = zeros(all_size, all_size);  % gradients
for nConcept = 1:conNum 
    for nSvm = 1 : svm_num 
    
    id = conid(nConcept); 
    % get the train index of the training instances
    cur_train_idx = sampled_train_idx{id, nSvm}; 
    indicator = dev_gt_mat(id, cur_train_idx)'; 
    train_lab = ones(length(indicator), 1); 
    train_lab(~indicator) = -1;  
    
    cur_sup_size = length(cur_train_idx); 
    % reshape the training and test as [training idx; test idx]
    cur_idx = [cur_train_idx; test_idx];
    cur_all_size = length(cur_idx); 
    cur_sup_idx = 1 : cur_sup_size; 
    cur_test_idx = cur_sup_size+1 : cur_all_size;   
    
%     cur_in_kernel = dev_pre_kernels(cur_idx, cur_idx, :); 
%     cur_in_kernel = permute(cur_in_kernel, [3 1 2]); 
%     cur_in_kernel = reshape(cur_in_kernel, basic_kernel_size, cur_all_size*cur_all_size); 
%     cur_in_kernel = cur_in_kernel';
% 
%     % forword the network
%     cur_out_kernel = nn_fw_l3(cur_in_kernel, inhid_w, hidout_w); 
%     cur_out_kernel = reshape(cur_out_kernel, cur_all_size, cur_all_size); 
    cur_out_kernel = out_kernel(cur_idx, cur_idx); 
    sup_out_kernel = cur_out_kernel(cur_sup_idx, cur_sup_idx);
    sup_out_kernel = double(sup_out_kernel);  
    
    % train one vs rest svm
    [alpha, rho, energy, c_penalty] = svm_train(sup_out_kernel, train_lab, c_vec(nConcept, nSvm));    

    svm_models{nConcept, 1, nSvm} = single(alpha);
    svm_models{nConcept, 2, nSvm} = single(rho); 
    svm_models{nConcept, 3, nSvm} = single(energy);
    svm_models{nConcept, 4, nSvm} = single(c_penalty);
%     fprintf('%d, pos: %d, neg: %d, C: %f\n', id, sum(train_lab==1), sum(train_lab==-1),...
%            c_penalty);
 
    % predict the test   
    test_out_kernel = cur_out_kernel(cur_test_idx, cur_sup_idx); 
    dec_values = test_out_kernel * alpha + rho; 
    SCO_matrix( nConcept, : ) = ( SCO_matrix( nConcept, : ) + double(dec_values') );  
    
    % compute the gradient
    % combine the neighbors 
    cur_Laplacian = global_laplacian(cur_idx, cur_idx); 
    cur_simi_mat = cur_Laplacian - diag(diag(cur_Laplacian)); 
    cur_simi_mat = full(cur_simi_mat); 
    cur_union_flag = zeros(cur_all_size, cur_all_size); 
    cur_union_flag(cur_sup_idx, cur_sup_idx) = 1; 
    cur_union_flag(cur_simi_mat~=0) = 1; 
    cur_sup_flag = zeros(cur_all_size, cur_all_size);
    cur_sup_flag(cur_sup_idx, cur_sup_idx) = 1; 
    cur_neighbor_flag = zeros(cur_all_size, cur_all_size);
    cur_neighbor_flag(cur_simi_mat~=0) = 1; 

    tmp = cur_union_flag(:); 
    rout_kernel = cur_out_kernel(:);
    cur_rout_kernel = rout_kernel(tmp==1);
    cur_sup_flag = cur_sup_flag(tmp==1); 
    cur_neighbor_flag = cur_neighbor_flag(tmp==1); 
    cur_similarity = single(-cur_simi_mat(tmp==1));
    
    % initialize the svm gradient
    [deltaoutker, cur_fx] = cal_gradient_semi_neig(cur_rout_kernel, svm_models(nConcept, :), cur_sup_flag, ...
                            cur_similarity, cur_neighbor_flag, lambta);
    fx = fx + cur_fx;
    cur_rdelta = zeros(size(rout_kernel));
    cur_rdelta(tmp==1) = deltaoutker;
    cur_rdelta = reshape(cur_rdelta, cur_all_size, cur_all_size);
    all_delta(cur_idx, cur_idx) = all_delta(cur_idx, cur_idx) + cur_rdelta;
    
    end
end
% fprintf('\n'); 
final_models.svm_models = svm_models; 

SCO_matrix = SCO_matrix / svm_num; 
DEC_matrix = zeros(conNum, test_size); 
for jj = 1 : test_size
    [~, idx] = sort( SCO_matrix(:,jj) );  
    DEC_matrix(idx(end-4:end), jj) = 1; 
end
DEC_matrix = (DEC_matrix==1); 
cnptRES = zeros(conNum, 3); 
%%% Measures per concept %%%
cnptRES(:,1) = sum(DEC_matrix&test_lab_mat,2)./sum(DEC_matrix,2); %%% Precision %%%
cnptRES(~isfinite(cnptRES(:,1)),1) = 0;
cnptRES(:,2) = sum(DEC_matrix&test_lab_mat,2)./sum(test_lab_mat,2); %%% Recall %%%
cnptRES(~isfinite(cnptRES(:,2)),2) = 0;
cnptRES(:,3) = 2*(cnptRES(:,1).*cnptRES(:,2))./(cnptRES(:,1)+cnptRES(:,2)); %%% F-measure %%%
cnptRES(~isfinite(cnptRES(:,3)),3) = 0;     
learn_te_res = [learn_te_res; [mean(cnptRES, 1)*100, sum(cnptRES(:, 2)>0)]];
energy_list(it, 1) = fx;
cur_results = [mean(cnptRES, 1)*100, sum(cnptRES(:, 2)>0)]; 
fprintf('P: %f, R: %f, N: %d, ', cur_results(1), cur_results(2), cur_results(4));


