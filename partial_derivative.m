function [dydtx,dydty]= partial_derivative(k_traj,X_new_Car,CalibSize);
       Nc = size(X_new_Car,3);
       
       kx = squeeze(k_traj(1,:,:));
       ky = squeeze(k_traj(2,:,:));
        
       dkx = zeros(size(kx));
       dky = zeros(size(ky));
       
       dkx(2:end,:) = kx(2:end,:)- kx(1:end-1,:);
       dky(2:end,:) = ky(2:end,:)- ky(1:end-1,:);
       
       tmp =(1j*ifft2c(X_new_Car).*repmat([0:(CalibSize(1)/2-1), 0, -CalibSize(1)/2+1:-1]'/CalibSize(1),[1 CalibSize(2) Nc]));
       tmp_calib = bart('bart nufft',k_traj,reshape(tmp,[CalibSize 1 Nc]));
       dydkx = reshape(tmp_calib,[size(kx) Nc]);

       tmp = (1j*ifft2c(X_new_Car).*repmat([0:(CalibSize(1)/2-1), 0, -CalibSize(1)/2+1:-1]/CalibSize(1),[CalibSize(1) 1 Nc]));  
       tmp_calib = bart('bart nufft',k_traj,reshape(tmp,[CalibSize 1 Nc]));
       dydky = reshape(tmp_calib,[size(kx) Nc]);
        
       dydtx = dydkx.*repmat(dkx,[1 1 Nc]); 
       dydty = dydky.*repmat(dky,[1 1 Nc]);
       dydtx(isnan(dydtx)) = 0;
       dydty(isnan(dydty)) = 0;
end