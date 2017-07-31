clc;close all,clear
%%
fid = fopen('spiralGRAPPA_RT.ks','r');
traj = fread(fid,10000000,'float' );
fclose(fid);
traj_x = traj(1:3:end);
traj_x = reshape(traj_x,[3996 3]);
traj_y = traj(2:3:end);
traj_y = reshape(traj_y,[3996 3]);
% 1182 *60
k_traj = zeros(3,size(traj_x,1),size(traj_y,2));
k_traj(1,:,:) = traj_x;
k_traj(2,:,:) = traj_y;

k = traj_x+1j*traj_y;


%% heart data:
filename = 'heart6.raw'
sview = 0;
eview = 99999;

fid = fopen(filename,'r','l');

%substrview = 8;
%noviews = numinter;
%[magic, count] = fread(fid, 1, 'char');

[magic, count] = fread(fid, 4, 'char');
  if (count ~= 4)
	      return;
  end;
	if (sum(magic' ~= ['H' 'R' 'A' 'W']))
	fprintf(1, 'Invalid file format\n');
	return;
	end;

[data, count] = fread(fid, 1, 'int');
  if (count ~= 1)
         return;
  end;

  if (data ~= 50331648)
  	fprintf(1,'Unsupported raw format (%d). Use the raw file converter to update the file to the latest format\n', data);
	return;
  end;
  
version = data;

[data, count] = fread(fid, 1, 'int');
  if (count ~= 1)
         return;
  end;

endianess = data;


viewcount = zeros(1,16384);

interleave = 5;

while (interleave >0)

      [data, count] = fread(fid, 3, 'int');
      if (count ~= 3)
	break;
      end;

      d_samples = data(1);
      d_channels = data(2);
      d_size = data(3);

      [data2, count] = fread(fid, 2, 'int64');
      if (count ~= 2)
	break;
      end;
      tst_sec = data2(1);
      tst_usec = data2(2);

      [data, count] = fread(fid, 1, 'int');
      if (count ~= 1)
	break;
      end;

      d_nTags = data;

      [tags, count] = fread(fid, d_nTags, 'int');
      if (count ~= d_nTags)
	break;
      end;
  
      d_view = tags(2);
      d_slice = 0;
      d_echo = 0;
      d_id = tags(1);

  [data, count] = fread(fid, 1, 'int');
  if (count ~= 1)
    break;
  end;
  d_vector_size = data(1);    

  fprintf(1, 'slice: %03d, echo: %02d, view: %03d, id: %02ld, ch: %02d, samples: %04d, size: %04d, sec: %d, usec: %d\n', ...
  d_slice, d_echo, d_view, d_id, d_channels, d_samples, d_vector_size , tst_sec, tst_usec);
  
  [raw, count] = fread(fid, d_vector_size/2, 'int16');  
  if (count ~= (d_vector_size/2))
    break;
  end;

%  rawsw = int16(bitor(uint16(raw(2:2:end)) * 256, uint16(raw(1:2:end))));

%  raw = double(rawsw);


  if (and(d_view <= eview,d_view >=sview))
  viewcount(d_view+1) = viewcount(d_view+1) + 1;
  [d_view+1 viewcount(d_view+1)]
  if (d_channels > 1)
  PFILE(:, d_view+1, viewcount(d_view+1)) = raw(1:2:(d_samples*2*d_channels)) + i*raw(2:2:(d_samples*2*d_channels));
  	%PFILE(:, d_view+1 - substr_view, viewcount(d_view+1)) = raw(1:2:(d_samples*2*d_channels)) + i*raw(2:2:(d_samples*2*d_channels));
  else
  PFILE(:, d_view+1, viewcount(d_view+1)) = raw(1:2:(d_samples*2)) + i*raw(2:2:(d_samples*2));
  	%PFILE(:, d_view+1 - substr_view, viewcount(d_view+1)) = raw(1:2:(d_samples*2)) + i*raw(2:2:(d_samples*2));
  end;
  end;
  interleave = interleave-1;
end;

rawdata = PFILE(:,2:end,1);
rawdata = reshape(rawdata,[3996 8 3]);
data = permute(rawdata,[1 3 2]);

%% DELAY ESTIMATION
wnRank = 1.8;
eigThresh_k = 0.02; % threshold of eigenvectors in k-space
eigThresh_im = 0.6; % threshold of eigenvectors in image space
ksize = [6,6]; 
CalibSize = [36,36];
N = [320 320];
%%
M = find(abs(k)* N(1) >= CalibSize(1)/2,1)

% subsample for computation, center is still fully sampled
k_calib = k(1:M,1:1:end)*N(1);  % attention : need to be normalized!!!!
k_tem = k(1:(M+10),1:1:end)*N(1);

data_calib = data(1:M,1:1:end,:);
Nc = size(data,3);
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
         
        im_calib = bart('bart nufft -i -l 0.01 -d 36:36:1',k_traj,reshape(Y,[1 M size(k_calib,2) Nc]));
        X_old_Car = fft2c(squeeze(im_calib));        
 end
%%
im_dc = squeeze(bart('bart nufft -i -l 0.1 -d 320:320:1',k_traj*N(1),reshape(data,[1  3996  3  8])));
figure,imshow(sos(im_dc),[])
%%
k_true = ksp_interp(k,d_total);   

%%
%%ESPIRIT
Kcalib = crop(squeeze(X_old_Car),[30 30 Nc]);
%Kcalib = crop(squeeze(fft2c(im_calib)),[30 30 Nc]);
%subplot(122),imshow(sos(ifft2c(Kcalib)),[]),title('crop the edge')
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
%% parallel imaging recon

maps = M(:,:,:,end);
weights = double(W(:,:,end) >  eigThresh_im);

ESP = ESPIRiT(maps,weights);

disp('Done Calibrating');

%%
%k_true = k;

accel = 1;
idx = (1:accel:size(k,2));
w = repmat(ones(size(k_true(:,1))),[1 size(k_true,2)]);
nCoil = size(data,3);
k_u = k_true(:,idx);
w_u = w(:,idx);  % use w_u
kData_u = data(:,idx,:);
%%
disp('generating nufft object for recon')
GFFT_u = NUFFT(k_u,w_u, [0,0], N);
nIterCG = 30;
%im_dc = GFFT_u'*(kData_u.*repmat(sqrt(w_u),[1,1,nCoil]))*accel;
res = im_dc;
disp('initiating reconstruction')
tic
%%
XOP = Wavelet('Daubechies',4,6);
[res] = cgL1ESPIRiT(kData_u.*repmat(sqrt(w_u),[1,1,nCoil]), ESP'*im_dc, GFFT_u, ESP, nIterCG,XOP,1,0.5,20);
toc
disp('done!');

figure(100), imshow(cat(2,sos(im_dc),abs(res)),[]), 
