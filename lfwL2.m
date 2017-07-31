function [] = lfwL2()

load feature/lfw_feats_sphereface_iter_22000.mat
% load feature/lfw_feats_sphereface3_28000.mat
% load feature/lfw_feats_normface.mat
% load feature/lfw_feats_center_author.mat
% load feature/lfw_feats_casia10_0_200000_gray.mat
% load lfw_feats_nir10_0_10000_gray.mat
% load lfw_feats_casia10_0_600000_gray_lefteye0.6.mat
% load lfw_feats_casia14_0_460000_gray_leftmouth0.8.mat
% load lfw_feats_casia10_0_200000_gray.mat
% load lfw_feats_casia10_1_600000_gray.mat
% load  lfw_feats_casia10_center2_430000_gray.mat
% load lfw_feats_casia11_0_400000_gray.mat
% load lfw_feats_casia10_center2_10000_gray
% load lfw_feats_casia14_0_460000_gray_leftmouth0.8.mat
% load lfw_feats_casia12_0_320000_gray.mat
% load lfw_feats_casia9_1_340000.mat
% load coeff_casia7_0.mat
load lfw/lfw_MTCNN_pairs.mat

libsvm_path = './libsvm-3.21/matlab'
addpath(genpath(libsvm_path));

F1 = double(F1);
F2 = double(F2);
% Mirror trick
F1 = max(F1(:,1:512), F1(:, 513:end));
F2 = max(F2(:,1:512) , F2(:, 513:end));

%10-folders cross validation
same_label = ones(6000,1);
same_label(3001:6000) = 0;

% F1 = bsxfun(@rdivide, F1, sqrt(sum(F1.^2,2)));
% F2 = bsxfun(@rdivide, F2, sqrt(sum(F2.^2,2))); 

%% Plot the distribution of distance
thresh = zeros(size(F1,1),1);
for j = 1:size(F1,1)
%     thresh(j) = sqrt(sum((F1(j,:)-F2(j,:)).^2));
    thresh(j) =  F1(j,:)*F2(j,:)'/(norm(F1(j,:))*norm(F2(j,:)));
end
hist(thresh(1:3000),200);
hold on;
hist(thresh(3001:6000),200);
hold off;
title('The distribution of cosine distance ');

%% Plot ROC Curve
MAX = max(thresh);
MIN = min(thresh);
roc_x = [];
roc_y = [];
for t = MIN:0.001:MAX
    positive=find(thresh<=t);
    negtive = find(thresh>t);
    FP = find(positive>3000);
    TP = find(positive<=3000);
    FPR = length(FP)/3000;
    TPR = length(TP)/3000;
    roc_x = [roc_x FPR];
    roc_y = [roc_y TPR];
end
plot(roc_x,roc_y);
title('ROC');
xlabel('FPR');
ylabel('TPR');

accuracies = zeros(10,1);
accs = zeros(10,1);
for i = 1:10
    test_idx = [(i-1) * 300 + 1:i*300, (i-1) * 300 + 3001:i*300 + 3000];
    train_idx = 1:6000;
    train_idx(test_idx) = [];
    train = [F1(train_idx,:);F2(train_idx,:)];
    
    % PCA
    [coeff,score,latent,tsquared,explained, mu] = pca(train);
    F1_score = (F1 - repmat(mu, length(F1) ,1))*coeff;
    F2_score = (F2 - repmat(mu, length(F2) ,1))*coeff;
    sum_var = cumsum(explained);
    dims = find(sum_var>99.5, 1, 'first')
    F1_pca = F1_score(:,1:dims);
    F2_pca = F2_score(:,1:dims);
    for j = 1:size(F1,1)
        thresh(j) = 1- F1_pca(j,:)*F2_pca(j,:)'/(norm(F1_pca(j,:))*norm(F2_pca(j,:)));
    end
    
    
    thr = getThreshold(thresh(train_idx), same_label(train_idx), 0.001);
    accs(i)  = getAccuracy(thresh(test_idx), same_label(test_idx), thr)
    


%     hist(thresh(1:3000),200);
%     hold on;
%     hist(thresh(3001:6000),200);
%     train_labels = [lfw_labels(train_idx,1);lfw_labels(train_idx,2)];
%     [mappedx, mapping] = JointBayesian(train, train_labels)
    

    cmd = [' -t 0 -h 0 -b 1'];
    model = svmtrain(same_label(train_idx), thresh(train_idx), cmd);
    [class] = svmpredict(same_label(train_idx), thresh(train_idx), model);
    [class, accuracy, deci] = svmpredict(same_label(test_idx), thresh(test_idx), model,'-b 1');
    accuracies(i) = accuracy(1);
%    roc_label = same_label(test_idx);
%    roc_label = [roc_label, 1-roc_label];
% %     plotroc(roc_label',deci');
%     
%     fp_idx = test_idx(find(class(301:600)>0) + 300);
%     fn_idx = test_idx(find(class(1:300)==0));
%     same_pair(fn_idx);
%     diff_pair(fp_idx);
%     for k = 1:size(fn_idx,2)
%         same_pair{fn_idx(k),1}
%         same_pair{fn_idx(k),2}
%         I1 = imread(same_pair{fn_idx(k),1});
%         I2 = imread(same_pair{fn_idx(k),2});
%         subplot(1,2,1);
%         imshow(I1);
%         subplot(1,2,2);
%         imshow(I2);
%         text(1,1,['Similarity: ', num2str(thresh(fn_idx(k) ) )],'FontSize',16,'color','r');
%         
%     end
%     for k = 1:size(fp_idx,2)
%         diff_pair{fp_idx(k)-3000,1}
%         diff_pair{fp_idx(k)-3000,2}
%         I1 = imread(diff_pair{fp_idx(k)-3000,1});
%         I2 = imread(diff_pair{fp_idx(k)-3000,2});
%         subplot(1,2,1);
%         imshow(I1);
%         subplot(1,2,2);
%         imshow(I2);
%         text(1,1,['Similarity: ', num2str(thresh(fp_idx(k) ) )],'FontSize',16,'color','r'); 
%     end
end

mean(accuracies)
mean(accs)

end

function [thr] = pcaSearch(F1,F2, step)
    test_idx = [randperm(300) 2700+randperm(300)]';
    train_idx = 1:5400;
    train_idx(test_idx) = [];
    train = [F1(train_idx, :); F2(train_idx, :)];
    % pca
    [coeff,score,latent,tsquared,explained, mu] = pca(train);
    F1_score = (F1 - repmat(mu, size(F1, 1) ,1))*coeff;
    F2_score = (F2 - repmat(mu, size(F2, 1) ,1))*coeff;
    sum_var = cumsum(explained);
    same_label = [ones(2700, 1); zeros(2700,1)];
    accuracies = [];
    a = 95;
    b = 100;
    
    for pca_t = a:step:b
        dims = find(sum_var>pca_t,1,'first')
        F1_pca = F1_score(:,1:dims);
        F2_pca = F2_score(:,1:dims);
        thresh = zeros(size(F1_pca, 1), 1);
        for j = 1:size(F1_pca,1)
%         thresh(j) = sqrt(sum((F1_pca(j,:)-F2_pca(j,:)).^2));
            thresh(j) = 1- F1_pca(j,:)*F2_pca(j,:)'/(norm(F1_pca(j,:))*norm(F2_pca(j,:)));
        end
        cmd = [' -t 0 -h 0 -b 1'];
        model = svmtrain(same_label(train_idx), thresh(train_idx), cmd);
        [class] = svmpredict(same_label(train_idx), thresh(train_idx), model);
        [class, accuracy, deci] = svmpredict(same_label(test_idx), thresh(test_idx), model,'-b 1');
        accuracies = [accuracies accuracy(1)];
    end
    [acc, ind] = max(accuracies);
    thr = a + (step-1)*ind;
    ['Optimal pca ratio: ' num2str(thr) ' accuracy:' num2str(acc)]
end

function bestThreshold = getThreshold(scores, positive, thrstep)
    a = min(scores);
    b = max(scores);
    thrs = a:thrstep:b;
    accs = zeros(length(thrs),1);
    for i = 1:length(thrs)
        accs(i) = getAccuracy(scores, positive, thrs(i));        
    end
    [~, indx]     = max(accs);
    bestThreshold = thrs(indx);
end

function acc = getAccuracy(scores, positive, threshold)
    acc = (length(find(scores(positive==1)<threshold)) + ...
           length(find(scores(positive==0)>threshold))) / length(scores);
end
