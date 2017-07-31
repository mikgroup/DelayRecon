function k_update = ksp_interp(k_tem,d_total);
k_update = zeros(size(k_tem));
      
       for ind = 1: size(k_tem,2)    
          % tmp1 = interp1([1:size(k_calib,1)]+d_total(1),real(k_calib(:,ind)),([1:size(k_calib,1)]),'linear'); %Kx
          % tmp2 = interp1([1:size(k_calib,1)]+d_total(2),imag(k_calib(:,ind)),([1:size(k_calib,1)]),'linear'); %Ky
           tmp1 = interp1([1:size(k_tem,1)]+d_total(1),real(k_tem(:,ind)),[1:size(k_tem,1)],'linear'); %Kx
           tmp2 = interp1([1:size(k_tem,1)]+d_total(2),imag(k_tem(:,ind)),[1:size(k_tem,1)],'linear'); %Ky
          
           if (d_total(1) > 0 )
           tmp1(isnan(tmp1)) = 0;  % could be a problem
           else
           tmp1(isnan(tmp1)) = real(k_tem(isnan(tmp1),ind)); 
           end
           if (d_total(2) > 0 )
           tmp2(isnan(tmp2)) = 0;
           else
             tmp2(isnan(tmp2)) = imag(k_tem(isnan(tmp2),ind));
           end
            k_update(:,ind) = tmp1(:) + 1j*tmp2(:); 
       end    
end
