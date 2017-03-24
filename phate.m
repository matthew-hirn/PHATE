function [Y, DiffOp, DiffOp_t] = phate(data, varargin)

k = 5;
a = 10;
npca = 100;
t = 12;
ndim = 2;
mds_method = 'cmds';
symm = 'pdist';
distfun = 'euclidean';
pca_method = 'svd';

for i=1:length(varargin)
        if(strcmp(varargin{i},'k'))
           k =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'a'))
           a =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'t'))
           t =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'npca'))
           npca =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'ndim'))
           ndim =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'mds_method'))
           mds_method =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'symm'))
           symm =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'distfun'))
           distfun =  lower(varargin{i+1});
        end
        if(strcmp(varargin{i},'pca_method'))
           pca_method =  lower(varargin{i+1});
        end
end

disp '======= PHATE ======='

M = svdpca(data, npca, pca_method);

disp 'computing distances'
PDX = squareform(pdist(M, distfun));
[~, knnDST] = knnsearch(M,M,'K',k+1,'dist',distfun);

disp 'computing kernel and operator'
epsilon = knnDST(:,k+1); % bandwidth(x) = distance to k-th neighbor of x
PDX = bsxfun(@rdivide,PDX,epsilon); % autotuning d(x,:) using epsilon(x)
GsKer = exp(-PDX.^a); % not really Gaussian kernel
GsKer=GsKer+GsKer';
DiffDeg = diag(sum(GsKer,2)); % degrees
DiffOp = DiffDeg^(-1)*GsKer; % row stochastic

disp 'diffusing operator'
DiffOp_t = DiffOp^t;
X = DiffOp_t;

disp(['symm: ' symm])
switch symm
    case 'x*xt'
        X = X * X';
        disp 'potential recovery'
        X(X<=eps)=eps;
        X = -log(X);
        disp 'setting diag to zero'
        X = X - diag(diag(X));
    case 'x+xt'
        X = (X + X') ./ 2;
        disp 'potential recovery'
        X(X<=eps)=eps;
        X = -log(X);
        disp 'setting diag to zero'
        X = X - diag(diag(X));
    case 'pdist'
        disp 'potential recovery'
        X(X<=eps)=eps;
        X = -log(X);
        X = svdpca(X, npca, pca_method);
        X = squareform(pdist(X));
end

switch mds_method
    case 'cmds_fast'
        Y = randmds(X, ndim);
    case 'cmds'
        disp 'CMDS'
        Y = cmdscale(X, ndim);
    case 'nmmds'
        disp 'NMMDS'
        opt = statset('display', 'iter');
        Y = mdscale(X, ndim, 'options', opt);
end

disp 'done.'
