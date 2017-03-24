function Y = svdpca(X, k, method)

if ~exist('method','var')
    method = 'svd';
end

X = bsxfun(@minus, X, mean(X));

switch method
    case 'svd'
        disp 'PCA using SVD'
        [U,~,~] = svds(X', k);
    case 'random'
        disp 'PCA using random SVD'
        [U,~,~] = randPCA(X', k);
end

Y = X * U;