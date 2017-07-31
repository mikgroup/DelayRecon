function X_new = lowrank_thresh(X_old,ksize,thresh);
       keep = 1:thresh;
       [sx,sy,Nc] = size(X_old);
       tmp = im2row(X_old,ksize); 
       [tsx,tsy,tsz] = size(tmp);
       A = reshape(im2row(X_old,ksize),tsx,tsy*tsz);
      [U,S,V] = svd(A,'econ');
       A = U(:,keep)*S(keep,keep)*V(:,keep)';
      %A = U * SoftThresh( S, thresh ) * V';
      %S(S<thresh) = 0;
       A = reshape(A,tsx,tsy,tsz);       
      X_new = row2im(A,[sx,sy,Nc],ksize);

end