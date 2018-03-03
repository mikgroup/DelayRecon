
function res = delay_SNR(scale);
%clear;clc;close all
addpath data/
%% real data:
%% load image and sensitivity maps to create multi-channel images
load im1.mat
load smaps.mat
%%
ksize = [6,6];                  % SPIRiT Kernel size
CalibSize = [36,36];            % size of the calibration region
wnRank = 1.8;
eigThresh_k = 0.02; % threshold of eigenvectors in k-space
%%
Nc = size(smaps,3);
channel_im = smaps .* repmat(im1, [1 1 Nc]);

%figure,imshow3(abs(channel_im),[],[1 Nc])
%% radial trajectory data
Nacq = 256;
Tramp = 15
kshift = 0;
Nramp = floor(Tramp - kshift) + 1;
k = [ max([0:Nramp-1]+kshift , 0).^2, ...
        (Tramp^2 + 2*Tramp*([Nramp:Nacq-1] + kshift - Tramp)) ];    
kr = k/max(k)/2;

kx = kr'*sin([0:804]/805*2*pi);
ky = kr'*cos([0:804]/805*2*pi);
k = kx+1j*ky;
% 
N = [size(channel_im,1),size(channel_im,2)];                  % size of the target image

%% change to spiral trajectory
%load spiral.mat
k_traj = zeros(3,size(k,1),size(k,2));
k_traj(1,:,:) = real(k);
k_traj(2,:,:) = imag(k);

%% forward model
 data = bart('bart nufft',k_traj*N(1),reshape(channel_im,[N 1 Nc]));
 % iterative
 %%
% 
 im_recon = bart('bart nufft -i -l 0.1',k_traj*N(1),data);
 figure,imshow(sos(im_recon),[])

%% To create delay on x,y direction respectively
kx = real(k);
ky = imag(k);
kx_true = zeros(size(k));
ky_true = zeros(size(k));
% delay in x = 1
kx_true(2:end,:) = kx(1:end-1,:); 
kx_true(1,:) = kx(1,:); 

% delay in x = -2
% kx_true(1:end-2,:) = kx(3:end,:); 
% kx_true(end-1:end,:) = repmat(kx(end,:),[2 1]);

% delay in y = 2
% ky_true(3:end,:) = ky(1:end-2,:); 
% ky_true(1:2,:) = repmat(ky(1,:),[2 1]);
% 
% delay in y = -1
ky_true(1:end-1,:) = ky(2:end,:); 
ky_true(end,:) = ky(end,:); 
% 
k_true = zeros(3,size(k,1),size(k,2));
k_true(1,:,:) = kx_true(:,:);
k_true(2,:,:) = ky_true(:,:);

data = bart('bart nufft',k_true*N(1),reshape(channel_im,[N 1 Nc]));
data = reshape(data,[size(k) Nc]);

%% add noise
noise = randn(size(data)) + 1j*randn(size(data));
data = data + 5000* noise*scale;
%%
% data = squeeze(data);
% D = reshape(data,size(data,1)*size(data,2),Nc);
% [U,S,V] = svd(D,'econ');
% Nc = max(find(diag(S)/S(1)>0.05));
% data = reshape(D*V(:,1:Nc),size(data,1),size(data,2),Nc);
%% Calibration Region
% image size

% choose low resolution part 
M = find(abs(k)* N(1) >= CalibSize(1)/2,1)

% subsample for computation, center is still fully sampled
k_calib = k(1:M,1:1:end)*N(1);  % attention : need to be normalized!!!!
k_tem = k(1:(M+10),1:1:end)*N(1);

data_calib = data(1:M,1:1:end,:);

k_calib = k_calib/max(abs(k_calib(:)))*CalibSize(1)/2;


kx = real(k_calib);
ky = imag(k_calib);

k_traj = zeros(3,M,size(k_calib,2));
k_traj(1,:) = kx(:);
k_traj(2,:) = ky(:);

im_calib = bart('bart nufft -i -l 0.01',k_traj,reshape(data_calib,[1 M size(k_calib,2) Nc]));

%figure,imagesc(sos(im_calib)),colormap(gray),axis off
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
       % part 2 solve for X 
       X_new_Car = lowrank_thresh(X_old_Car,ksize,floor(wnRank*prod(ksize)));
       %rank = rank+0.2
       im_new = ifft2c(X_new_Car);
      % NUFFT to get updated k-space data
  
       data_calib = bart('bart nufft',k_traj,reshape(im_new,[CalibSize 1 Nc]));
       X_new  = reshape(data_calib,[M size(k_calib,2) Nc]);
           
       % part 2 solve for delta t  
      
       % partial derivative 
       [dydtx,dydty]= partial_derivative(k_traj,X_new_Car,CalibSize);
       % direct solver     
       dydt = [real(vec(dydtx)) real(vec(dydty));imag(vec(dydtx)) imag(vec(dydty))]; 
       stepsize =  inv((dydt)'*dydt);
       d = stepsize * dydt'*[real(vec(X_new - Y));imag(vec(X_new - Y))];    
       d(isnan(d)) = 0;
%   
        d_total = d_total + real(d)  % the accumalated delay
        incre = norm(real(d));  % stop critiria
    
%       Do interpolation to update k-space
        k_update = ksp_interp(k_tem,d_total);   
   
        k_update = k_update(1:M,:);
        
        kx = real(k_update);
        ky = imag(k_update);
         
        k_traj(1,:,:) = kx/max(kx(:))*CalibSize(1)/2;
        k_traj(2,:,:) = ky/max(ky(:))*CalibSize(2)/2;
         
        im_calib = bart('bart nufft -i -l 0.01 -d 36:36:1',k_traj,reshape(Y,[1 M size(k_calib,2) Nc]));
        X_old_Car = fft2c(squeeze(im_calib));        
 end

 res = d_total;