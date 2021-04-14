clear all
close all
clc

%% sec 1
original = imread('cameraman.tif');
figure(1);
imshow(original);
title('original image');

%% sec 2
rotated_im = imrotate(original,20);
distorted = imresize(rotated_im,0.75);
figure(2);
imshow(distorted);
title('distorted');

%% sec 3
point_original = detectSURFFeatures(original);
point_distorted = detectSURFFeatures(distorted);

[feature_original, valid_points_original] = extractFeatures(original, point_original);
[feature_distorted, valid_points_distorted] = extractFeatures(distorted, point_distorted);

%% sec 4
[indexPairs, matchmetric] = matchFeatures(feature_original,feature_distorted);

%% sec 5
matched_p_original = valid_points_original(indexPairs(:,1));
matched_p_dist = valid_points_distorted(indexPairs(:,2));
figure(3);
ax=axes;
showMatchedFeatures(original,distorted,matched_p_original,matched_p_dist,'montage','Parent',ax);
title('Match points');

%% sec 7
[tform, inlier_origin,inlier_dist]= estimateGeometricTransform(matched_p_original,matched_p_dist,'similarity');

figure(4);
showMatchedFeatures(original,distorted,inlier_origin,inlier_dist,'montage')
title('Matched Inlier Points')

%% sec 8
scale_rot = norm([tform.T(1,1) tform.T(2,1)])
tetha_rot = acos(tform.T(1,1)/scale_rot)*180/pi

%% sec 9
outputView = imref2d(size(original));
recovered = imwarp(distorted,tform.invert,'OutputView',outputView);
figure(5)
imshow(recovered)
title('recovered image')