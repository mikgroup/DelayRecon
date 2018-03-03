%% True Scan
clc;clear;close all
%% load raw data and nominal trajectory
data = readcfl('PR_data');
k_traj = readcfl('PR_traj');

k_traj(3,:,:) = 0;
k_traj = k_traj/400;
%%
ksize = [6,6];                  % SPIRiT Kernel size
CalibSize = [36,36];            % size of the calibration region
wnRank = 1.8;
eigThresh_k = 0.02; % threshold of eigenvectors in k-space
Nc = size(data,4);
%% image size
N = [400 400];
k_nominal = squeeze(k_traj(1,:,:)) + 1j*squeeze(k_traj(2,:,:));

%% iterative gridding: avoid density compensation
im_recon = bart('bart nufft -i -l 0.1',k_traj*N(1),data);
figure,imshow(sos(im_recon),[])
%% Coil Compression -- Optional
% data = squeeze(data);
% D = reshape(data,size(data,1)*size(data,2),Nc);
% [U,S,V] = svd(D,'econ');
% Nc = max(find(diag(S)/S(1)>0.05));
% data = reshape(D*V(:,1:Nc),size(data,1),size(data,2),Nc);
%% Calibration Region

% choose low resolution part  (central part)
range = find(abs(k_nominal(:,1))* N(1) <= CalibSize(1)/2);
M1 = range(1);
M2 = range(end);
data = squeeze(data);
% subsample for computation, center is still fully sampled
k_calib = k_nominal(M1:M2,1:3:end);  % attention : need to be normalized!!!!

data_calib = data(M1:M2,1:3:end,:);

k_calib = k_calib/max(abs(k_calib(:)))*CalibSize(1)/2;

kx = real(k_calib);
ky = imag(k_calib);

k_traj = zeros(3,size(k_calib,1),size(k_calib,2));
k_traj(1,:,:) = kx;
k_traj(2,:,:) = ky;

im_calib = bart('bart nufft -i -l 0.01',k_traj,reshape(data_calib,[1 size(k_calib,1) size(k_calib,2) Nc]));

figure,imagesc(sos(im_calib)),colormap(gray),axis off
%%
% Initialization
index = 0;
incre = 10;
Kcalib = fft2c(im_calib);
tmp = im2row(Kcalib,ksize); 
[tsx,tsy,tsz] = size(tmp);
wnRank = 1.8;
rank = floor(wnRank*prod(ksize));
[sx,sy,Nc] = size(Kcalib);
% d is the vector of estimate delay - x,y 
d = zeros(2,1);

d_total = d;
Y = data_calib; % data consistency

X_old_Car = Kcalib;

%% stop 
 while  (index<300)  && (incre>0.001) 

       index = index+1;
       % part 1 solve for X
       X_new_Car = lowrank_thresh(X_old_Car,ksize,floor(wnRank*prod(ksize)));
       %rank = rank+0.2
       im_new = ifft2c(X_new_Car);
      % NUFFT to get updated k-space data
  
       data_calib = bart('bart nufft',k_traj,reshape(im_new,[CalibSize 1 Nc]));
       X_new  = reshape(data_calib,[M2-M1+1 size(k_calib,2) Nc]);
           
       % part 2 solve for delta t  
      
       % partial derivative 
       [dydtx,dydty]= partial_derivative(k_traj,X_new_Car,CalibSize);
       % direct solver     
       dydt = [real(vec(dydtx)) real(vec(dydty));imag(vec(dydtx)) imag(vec(dydty))]; 
       stepsize =  inv((dydt)'*dydt);
       d = 0.1*stepsize * dydt'*[real(vec(X_new - Y));imag(vec(X_new - Y))];    
       d(isnan(d)) = 0;
  
        d_total = d_total + real(d)  % the accumalated delay
        incre = norm(real(d));  % stop critiria
         
        k_whole_update = ksp_interp(k_nominal(:,1:3:end)*N(1),d_total);
        k_update = k_whole_update(M1:M2,:);
        kx = real(k_update);
        ky = imag(k_update);
         
        k_traj(1,:,:) = kx/max(kx(:))*CalibSize(1)/2;
        k_traj(2,:,:) = ky/max(ky(:))*CalibSize(2)/2;
         
        im_calib = bart('bart nufft -i -l 0.01 -d 36:36:1',k_traj,reshape(Y,[1 M2-M1+1 size(k_calib,2) Nc]));
        X_old_Car = fft2c(squeeze(im_calib));        
 end
%% updated trajectory for gridding recon
 k_corrected = ksp_interp(k_nominal(:,:),d_total);
 k_traj = zeros(3,size(k_corrected,1),size(k_corrected,2));
 k_traj(3,:,:) = 0;
 k_traj(1,:,:) = real(k_corrected);
 k_traj(2,:,:) = imag(k_corrected);
 im_corrected = bart('bart nufft -i -l 0.1', k_traj*N(1),reshape(data,[1 size(data)]));
 figure,imshow(sos(im_corrected),[])