clear all
close all
clc

%% sec 1.1
boxImage = imread('stapleRemover.jpg');
figure(1)
imshow(boxImage);
title('box image');
%% sec 1.2
sceneImage = imread('clutteredDesk.jpg');
figure(2)
imshow(sceneImage);
title('scene image');
%% sec 1.3
point_box = detectSURFFeatures(boxImage);
point_scene = detectSURFFeatures(sceneImage);

[feature_box, valid_points_box] = extractFeatures(boxImage, point_box);
[feature_scene, valid_points_scene] = extractFeatures(sceneImage, point_scene);

[indexPairs, matchmetric] = matchFeatures(feature_box,feature_scene);

matched_p_box = valid_points_box(indexPairs(:,1));
matched_p_scene = valid_points_scene(indexPairs(:,2));
figure(3);
ax=axes;
showMatchedFeatures(boxImage,sceneImage,matched_p_box,matched_p_scene,'montage','Parent',ax);
title('Match points');

[tform, inlier_origin,inlier_dist]= estimateGeometricTransform(matched_p_box,matched_p_scene,'affine');
%% sec 1.4
boxPolygon = [1, 1;... % top-left
size(boxImage, 2), 1;... % top-right
size(boxImage, 2), size(boxImage, 1);... % bottom-right
1, size(boxImage, 1);... % bottom-left
1, 1]; % top-left again to close the polygon
%% sec 1.5
scenePolygon = transformPointsForward(tform, boxPolygon);
scenePolygon_T = scenePolygon';

%% sec 1.6
GR_scene = insertShape(sceneImage,'Polygon',(scenePolygon_T(:))');

figure();
imshow(GR_scene,[]);
title('scene with polygon');

%% sec 2.1
im1 = rgb2gray(imread('view1.png'));
im2 = rgb2gray(imread('view2.png'));
figure();
imshow(im1);
figure();
imshow(im2);

%% sec 2.2
point_im1 = detectSURFFeatures(im1);
point_im2 = detectSURFFeatures(im2);

[feature_im1, valid_points_im1] = extractFeatures(im1, point_im1);
[feature_im2, valid_points_im2] = extractFeatures(im2, point_im2);

[indexPairs_im, matchmetric_im] = matchFeatures(feature_im1,feature_im2);

matched_p_im1 = valid_points_im1(indexPairs_im(:,1));
matched_p_im2 = valid_points_im2(indexPairs_im(:,2));
figure();
ax=axes;
showMatchedFeatures(im1,im2,matched_p_im1,matched_p_im2,'montage','Parent',ax);
title('Match points');

%% sec 2.3
[tform_im, inlier_im1, inlier_im2]= estimateGeometricTransform(matched_p_im2,matched_p_im1,'affine');

%% sec 2.4
x_min = 1;
y_min = 1;
x_max = size(im1,2);
y_max = size(im1,1);

%% sec 2.5
%im2_2 = imwarp(im2,tform_im);
%figure();
%imshow(im2_2);

[x_lim,y_lim] = outputLimits(tform_im,[1 size(im2,2)],[1 size(im2,1)]);

%% sec 2.6

x_min_t = min([x_min, x_lim(1)]);
x_max_t = max([x_max, x_lim(2)]);
y_min_t = min([y_min, y_lim(1)]);
y_max_t = max([y_max, y_lim(2)]);


width = round(x_max_t-x_min_t+1);

height = round(y_max_t-y_min_t+1);

panorama = zeros(height,width);

%% sec 2.7
panoramaView = imref2d([height width],[x_min_t,x_max_t],[y_min_t,y_max_t]);

%% sec 2.8
close all; clc;
i = [1 0 0; 0 1 0; 0 0 1];
panorama1 = zeros(size(panorama));
panorama1 = imwarp(im1, affine2d(i), 'OutputView', panoramaView);


%% sec 2.9
panorama2 = zeros(size(panorama));
panorama2 = imwarp(im2, tform_im, 'OutputView', panoramaView);
panorama2(:,1:x_max) = 0;

%% sec 2.10
panorama = panorama1 + panorama2;
panorama = panorama(y_min:(y_max+10),x_min:round(x_lim(2)));
figure();
imshow(panorama);
title('panorama- affine');

%% sec 2.11

[tform_im, inlier_im1, inlier_im2]= estimateGeometricTransform(matched_p_im2,matched_p_im1,'projective');

x_min = 1;
y_min = 1;
x_max = size(im1,2);
y_max = size(im1,1);


[x_lim,y_lim] = outputLimits(tform_im,[1 size(im2,2)],[1 size(im2,1)]);


x_min_t = min([x_min, x_lim(1)]);
x_max_t = max([x_max, x_lim(2)]);
y_min_t = min([y_min, y_lim(1)]);
y_max_t = max([y_max, y_lim(2)]);

width = round(x_max_t-x_min_t+1);

height = round(y_max_t-y_min_t+1);

panorama = zeros(height,width);

panoramaView = imref2d([height width],[x_min_t,x_max_t],[y_min_t,y_max_t]);

i = [1 0 0; 0 1 0; 0 0 1];
panorama1 = zeros(size(panorama));
panorama1 = imwarp(im1, affine2d(i), 'OutputView', panoramaView);

panorama2 = zeros(size(panorama));
panorama2 = imwarp(im2, tform_im, 'OutputView', panoramaView);
panorama2(:,1:x_max) = 0;

panorama = panorama1 + panorama2;
figure();
imshow(panorama(150:580, 1:1150));
title('panorama- projective');

%% sec 2.12

im1_rgb = imread('view1.png');
im2_rgb = imread('view2.png');

I = [1 0 0; 0 1 0; 0 0 1];
for i= 1:3
panorama1_rgb(:,:,i) = imwarp(im1_rgb(:,:,i),affine2d(I),'OutputView',panoramaView);
panorama2_rgb(:,:,i) = imwarp(im2_rgb(:,:,i),tform_im,'OutputView',panoramaView);

panorama2_rgb(:,1:577,i) = 0;
end

panorama_rgb = panorama1_rgb + panorama2_rgb ;
figure()
imshow(panorama_rgb(164:(end-19),1:(end-112),:));
title('panorama- rgb');
