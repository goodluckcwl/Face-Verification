clear;clc;


%minimum size of face
minsize=30;

%path of toolbox
caffe_path='/home/chenweiliang/caffe/matlab';
pdollar_toolbox_path='/home/chenweiliang/opt/Matlab-tools/toolbox-master';
mtcnn_path = '/home/chenweiliang/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2';
caffe_model_path= fullfile(mtcnn_path, 'model');
addpath(genpath(caffe_path));
addpath(genpath(mtcnn_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.5 0.6 0.6]

%scale factor
factor=0.709;


%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	

imgSize = [112, 96];
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
output_dir = '/home/chenweiliang/face-lfw/lfw-aligned-96';
folders = dir('/home/chenweiliang/face-lfw/lfw');
folders(1:2) = [];
mkdir(output_dir);
fid = fopen('undetected.txt','w');
for i = 1:length(folders)
    i
    folder_dir = fullfile('/home/chenweiliang/face-lfw/lfw', folders(i).name);
    names = dir(folder_dir);
    names(1:2) = [];
    for j = 1:length(names)
        if ~length(strfind(names(j).name,'.jpg'))&&~length(strfind(names(j).name,'.png'))
            continue;
        end
        im_dir = fullfile(folder_dir, names(j).name);
%         im_dir = '/home/chenweiliang/face-lfw/lfw/John_McEnroe/John_McEnroe_0001.jpg';
        im = imread(im_dir);
        
        %% for grayscale image
        [h,w,c] = size(im);
        if c==1
            im_tmp = zeros(h,w,3);
            im_tmp(:,:,1) = im;
            im_tmp(:,:,2) = im;
            im_tmp(:,:,3) = im;
            im = uint8(im_tmp);
        end
        

        [boudingboxes points]=detect_face(im,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
        numbox=size(boudingboxes,1);


        
        if ~numbox
            [im_dir ' undetected.']
            fprintf(fid,'%s\n',im_dir);
            continue;
        end
        %% select the true boudingbox
        [height, width, channels] = size(im);
        max_idx = 1;
        max_size = 0;
        for k = 1:numbox
            center = [(boudingboxes(k,1)+boudingboxes(k,3))/2,...
                (boudingboxes(k,2)+boudingboxes(k,4))/2];
            d = sum((center-[width/2, height/2]).^2);
            s = (min(boudingboxes(k,3), width) - max(boudingboxes(k,1), 1))*...
                (min(boudingboxes(k,4), height) - max(boudingboxes(k,2), 1)) -d ;
            if max_size < s
                max_idx = k;
                max_size = s;
            end
        end
        
        
        %% plot
%         figure(1);
%         imshow(im);hold on;
%         for k = max_idx:max_idx
%             plot(points(1:5,k),points(6:10,k),'g.','MarkerSize',10);
%             r=rectangle('Position',[boudingboxes(k,1:2) boudingboxes(k,3:4)-boudingboxes(k,1:2)],'Edgecolor','g','LineWidth',3);
%         end
%         pause(0.1);
        
        
        facial5points = double(reshape(points(:,max_idx), [],2));
        %% crop face
        Tfm =  cp2tform(facial5points, coord5points', 'similarity');
        cropImg = imtransform(im, Tfm, 'XData', [1 imgSize(2)],...
                                  'YData', [1 imgSize(1)], 'Size', imgSize);
        %% save result
        output_folder_dir = fullfile(output_dir, folders(i).name);
        mkdir(output_folder_dir);
        output_im_dir = fullfile(output_folder_dir, names(j).name);
        imwrite( cropImg,output_im_dir);
        
    end
    
    
end


