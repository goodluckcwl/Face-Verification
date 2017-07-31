clc;clear all;

load lfw/lfw_MTCNN_pairs.mat

%path of toolbox
caffe_path='/home/chenweiliang/caffe-windows-ms/matlab';
caffe_model_path='/home/chenweiliang/IDCard-Sphereface (another copy)/model'
addpath(genpath(caffe_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%load caffe models
prototxt_dir = '/home/chenweiliang/IDCard-Sphereface (another copy)/prototxt/sphereface_deploy.prototxt';
model_dir = strcat(caffe_model_path,'/sphereface_iter_28000.caffemodel');
net = caffe.Net(prototxt_dir,model_dir,'test');

%process LFW dataset
FEATURE_LENGTH = 512;
F1 = zeros(6000, FEATURE_LENGTH*2);
F2 = zeros(6000, FEATURE_LENGTH*2);
pairs = [same_pair; diff_pair];
for i=1:size(pairs,1)
    if mod(i, 100) ==0
        ['Process ' num2str(i) '/6000']
    end
    im1 = imread(pairs{i,1});
    cropImg = single(im1);
    cropImg = (cropImg - 127.5)/128;
    cropImg = permute(cropImg, [2,1,3]); % h,w,c-> w,h,c
    cropImg = cropImg(:,:,[3,2,1]); % rgb->bgr
    
    cropImg_(:,:,1) = flipud(cropImg(:,:,1));
    cropImg_(:,:,2) = flipud(cropImg(:,:,2));
    cropImg_(:,:,3) = flipud(cropImg(:,:,3));
    
    % extract deep feature
    res = net.forward({cropImg});
    res_ = net.forward({cropImg_});
    deepfeature = [res{1}; res_{1}];
    F1(i, :) = deepfeature';
    
    % F2
    im2 = imread(pairs{i,2});
    cropImg = single(im2);
    cropImg = (cropImg - 127.5)/128;
    cropImg = permute(cropImg, [2,1,3]);
    cropImg = cropImg(:,:,[3,2,1]);
    
    cropImg_(:,:,1) = flipud(cropImg(:,:,1));
    cropImg_(:,:,2) = flipud(cropImg(:,:,2));
    cropImg_(:,:,3) = flipud(cropImg(:,:,3));
    
    % extract deep feature
    res = net.forward({cropImg});
    res_ = net.forward({cropImg_});
    deepfeature = [res{1}; res_{1}];
    F2(i, :) = deepfeature';
end

save feature/lfw_feats_sphereface_iter_28000.mat F1 F2