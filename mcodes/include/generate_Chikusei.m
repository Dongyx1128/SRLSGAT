%% This is a demo code to show how to generate training and testing samples from the HSI %%
clc
clear
close all

addpath('/home/shiyanshi/dyx/SRLSGAT/mcodes/include');

%% Step 1: generate the training and testing images from the original HSI
% Please down the Chikusei/Pavia/Wdc dataset.

wdc = imread('/home/shiyanshi/dyx/datasets/Wdc/dc.tif');
wdc = im2double(wdc);
save('/home/shiyanshi/dyx/datasets/Wdc/Wdc.mat', 'wdc');

% load('/home/shiyanshi/dyx/datasets/Chikusei/HyperspecVNIR_Chikusei_20140729.mat');
% load('/home/shiyanshi/dyx/datasets/Pavia/Pavia.mat');
load('/home/shiyanshi/dyx/datasets/Wdc/Wdc.mat');

%% center crop this image to size 2304 x 2048
% a = chikusei(107:2410,144:2191,50:80);
% a = pavia(:,:,35:65);
a = wdc(:,:,81:111);
% clear chikusei;
% clear pavia;
clear wdc;
% normalization
a = a ./ max(max(max(a)));
a = single(a);
% save('/home/shiyanshi/dyx/datasets/Pavia/all/Pavia.mat', 'a');
save('/home/shiyanshi/dyx/datasets/Wdc/all/Wdc.mat', 'a');

%% select first row as test images
[H, W, C] = size(a);
% test_img_size = 512;
test_img_size = 256; % wdc
% test_pic_num = floor(W / test_img_size);
mkdir ('/home/shiyanshi/dyx/datasets/Wdc/test');
test = a(1:test_img_size,:,:);
% test = a(:,(test_img_size+1):end,:);
% save('/home/shiyanshi/dyx/datasets/Chikusei/test/Chikusei_test.mat', 'test');
% save('/home/shiyanshi/dyx/datasets/Pavia/test/Pavia_test.mat', 'test');
save('/home/shiyanshi/dyx/datasets/Wdc/test/Wdc_test.mat', 'test');


% for i = 1:test_pic_num
    % left = (i - 1) * test_img_size + 1;
    % right = left + test_img_size - 1;
    % test = a(1:test_img_size,left:right,:);
    % save(strcat('/home/shiyanshi/dyx/datasets/Chikusei/test/Chikusei_test_', int2str(i), '.mat'),'a');
% end

%% the rest left for training
mkdir ('/home/shiyanshi/dyx/datasets/Pavia/train');
train = a((test_img_size+1):end,:,:);
% train = a(:,1:test_img_size,:);
% save('/home/shiyanshi/dyx/datasets/Chikusei/train/Chikusei_train.mat', 'train');
% save('/home/shiyanshi/dyx/datasets/Pavia/train/Pavia_train.mat', 'train');
save('/home/shiyanshi/dyx/datasets/Wdc/train/Wdc_train.mat', 'train');

%% Step 2: generate the testing images used in mains.py
generate_test_data;

%% Step 3: generate the training samples (patches) cropped from the training images
generate_train_data;

%% Step 4: Please manually remove 10% of the samples to the folder of evals