% Experment Delay Verification

addpath ~/bart/matlab
addpath ../data
addpath ../code
%% Parameter
clear;clc;
close all
%%
wnRank = 1.8;
eigThresh_k = 0.05; % threshold of eigenvectors in k-space
eigThresh_im = 0.95; % threshold of eigenvectors in image space
ksize = [6,6]; 
CalibSize = [36,36];
%% Load the data
coils = [];
[data, header, rhuser] = rawloadX('0825_3T_inivo_x4y8',[],[],[],coils);
%% Gradient infomration
Nramp = rhuser(12);
frsize = size(data,1);
spres = [rhuser(2), rhuser(3)];
FOV = rhuser(17:18).';
Nprojections = rhuser(10);
shift = [rhuser(21:22)].'./spres;
Nc = size(data,5)
N = FOV./spres*10;
[theta, kmax, dcf] = calc_2dpr_ellipse(FOV*10, spres);

x = cos(theta) .* kmax;
y = sin(theta) .* kmax;
% Generate Trajectory
kscale = 0.5 / max(abs(kmax(:)));
x = kscale * x;
y = kscale * y;
[ksp, dcf_all] = calc_pr_ksp_dcf([x(:),y(:)],Nramp,frsize,dcf,1);

kx = reshape((ksp(:,1)),size(data,1),size(data,2));
ky = reshape((ksp(:,2)),size(data,1),size(data,2)); 
%%
k = kx+1j*ky;
%% 
k_traj = zeros(3,size(kx,1),size(kx,2));
k_traj(1,:,:) = kx*N(1);
k_traj(2,:,:) = ky*N(2);
%%
% Optional: coil compression
data = squeeze(data);
D = reshape(data,size(data,1)*size(data,2),Nc);
[U,S,V] = svd(D,'econ');
Nc = max(find(diag(S)/S(1)>0.05));
data = reshape(D*V(:,1:Nc),size(data,1),size(data,2),Nc);
%% Calibration Region
% image size
N = [320 320];
%%
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
       % part 2 solve for delta t  
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
         
        im_calib = bart('bart nufft -i -l 0.01',k_traj,reshape(Y,[1 M size(k_calib,2) Nc]));
        X_old_Car = fft2c(squeeze(im_calib));        
end

%% Theo following code is for estimate ESPIRiT sensitivity maps, you need to download ESPIRIT code 
Kcalib = crop(squeeze(X_old_Car),[30 30 Nc]);
[kernel,s] = dat2Kernel(Kcalib,ksize);
idx = max(find(s >= s(1)*eigThresh_k));
%%
% This shows that the calibration matrix has a null space as shown in the
% paper. 
kdisp = reshape(kernel,[ksize(1)*ksize(2)*Nc,size(kernel,4)]);
figure, subplot(211), plot([1:length(s)],s,'LineWidth',2);
hold on, 
plot([1:ksize(1)*ksize(2)*Nc],s(1)*eigThresh_k,'g-','LineWidth',2);
plot([idx,idx],[0,s(1)],'g--','LineWidth',2)
legend('signular vector value','threshold')
title('Singular Vectors')
subplot(212), imagesc(abs(kdisp)), colormap(gray(256));
xlabel('Singular value #');
title('Singular vectors')
%%
[M,W] = kernelEig(kernel(:,:,:,1:idx),N);
%%
% show eigen-values and eigen-vectors. The last set of eigen-vectors
% corresponding to eigen-values 1 look like sensitivity maps
figure, imshow3(abs(W),[],[1,Nc]); 
title('Eigen Values in Image space');
colormap((gray(256))); colorbar;
%%
figure, imshow3(abs(M),[],[Nc,Nc]); 
title('Magnitude of Eigen Vectors');
colormap(gray(256)); colorbar;

figure, imshow3(angle(M),[],[Nc,Nc]); 
title('Phase of Eigen Vectors');
colormap(jet(256)); colorbar;