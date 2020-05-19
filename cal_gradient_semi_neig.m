function [deltaoutker, fx] = cal_gradient_semi_neig(out_kernel, svm_models, sup_flag, similarity, ...
                             neighbor_flag, lambta)

pair_size = size(out_kernel, 1);    

% superivsed part   
conNum = size(svm_models, 1); 
sup_tr_size = sqrt(sum(sum(sup_flag))); 
alpha_mat = zeros(sup_tr_size, conNum);
for nConcept = 1 : conNum
    alpha_mat(:, nConcept) = svm_models{nConcept, 1} ;
end
tmp = - alpha_mat * alpha_mat' / 2;
delta_sup = zeros(pair_size, 1, 'single'); 
delta_sup(sup_flag==1) = tmp(:);
sup_energy = sum( delta_sup .* out_kernel ) + sum(sum(abs(alpha_mat))); 

% unsupervised part
uns_energy = out_kernel .* similarity;
uns_energy(neighbor_flag==1) = -1 * uns_energy(neighbor_flag==1);
uns_energy(neighbor_flag~=1) = 0;
uns_energy = sum(uns_energy); 

delta_uns = zeros(pair_size, 1, 'single'); 
delta_uns(neighbor_flag==1) = -1 * similarity(neighbor_flag==1);
delta_uns(neighbor_flag~=1) = 0;
    
deltaoutker = lambta * delta_sup + (1-lambta) * delta_uns; 

fx = lambta * sup_energy + (1-lambta) * uns_energy; 