clear all
close all
%%
im_path ='../data/';

%%
label_im = imread([im_path 'seg_bird.ppm']);
% figure;imagesc(label_im);axis equal
%%
label_list = [];
for i = 1:3
    cur_channel = label_im(:,:,i);
    cur_channel = cur_channel(:);
    label_list = [label_list cur_channel];
end
r_c = label_im(:,:,1);
g_c = label_im(:,:,2);
b_c = label_im(:,:,3);

color_list = unique(label_list, 'rows');
%%
gray_label = zeros(size(label_im, 1), size(label_im, 2));
label_cnt = 0;
for i = 1:size(color_list, 1)
   cur_c = color_list(i, :);
   idx = find(r_c == cur_c(1) & g_c == cur_c(2) & b_c == cur_c(3));
   tmp_im = zeros(size(gray_label));
   tmp_im(idx) = 1;
   CC = bwconncomp(tmp_im);
   for k = 1:CC.NumObjects
       gray_label(CC.PixelIdxList{k}) = label_cnt;
       label_cnt = label_cnt + 1;
   end
end
%%
save([im_path 'seg_bird.mat'], 'gray_label')
% figure;imagesc(gray_label);axis equal;colormap gray
