clc;
clear; 
close all;

% addpath('include');
% Convert HS dataset to patches

% List all '.mat' file in folder
file_folder=fullfile('/home/shiyanshi/dyx/datasets/Wdc/train/');
% file_folder=fullfile('/home/shiyanshi/dyx/datasets/Wdc/test/');
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};

% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    load(strcat('/home/shiyanshi/dyx/datasets/Wdc/train/',file_names{i}));
    % load(strcat('/home/shiyanshi/dyx/datasets/Wdc/test/',file_names{i}));
    crop_image(train, 64, 32, 0.5, name);
    % crop_image(test, 128, 64, 0.5, name);
    
    % crop_image(train, 64, 32, 0.25, name);
    % crop_image(test, 128, 64, 0.25, name);
    
    % crop_image(train, 128, 64, 0.125, name);
    % crop_image(test, 128, 64, 0.125, name);
end
