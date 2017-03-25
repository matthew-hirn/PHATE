% Test the embedding on the diffusion limited aggregation tree
% branching data.

%% Generate the data
n_dim = 100;
n_branch = 20;
n_steps = 100;
n_drop=0;
seed = 37;
rng(seed); % random seed for reproducibility (only necessaty fpr random pca and fast (random) mds)

%% generate random fractal tree via DLA
M = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
for I=1:n_branch-1
    ind = randsample(size(M,1), 1);
    M2 = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
    M = [M; repmat(M(ind,:),n_steps,1) + M2];
end
C = repmat(1:n_branch,n_steps,1);
C = C(:); % Color coding of branches
fprintf(1,'%u data points by %u features\n',size(M,1),size(M,2));

%% add noise
sigma = 4;
M = M + normrnd(0, sigma, size(M,1), size(M,2));

%% Run PHATE using CMDS (faster but less accurate)
t=30;
a=13;
k=5;
pca_method='random';
log_transform=0;
mds_method='cmds_fast';
ndim = 2;
Y_cmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'log',log_transform,'mds_method',mds_method);

% Plot the embedding colored by branch
scatter(Y_cmds(:,1),Y_cmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using CMDS'

%% Run PHATE using NMMDS (slower but more accurate)
t=30;
a=13;
k=5;
pca_method='random';
log_transform=0;
mds_method='nmmds';
ndim = 2;
Y_nmmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'log',log_transform,'mds_method',mds_method);

% Plot the embedding colored by branch
scatter(Y_nmmds(:,1),Y_nmmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS'
