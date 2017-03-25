%% load mESC data Klein et al. 2015 (doi:10.1016/j.cell.2015.04.044)
file_name = 'data/GSM1599499_ES_d7_LIFminus.csv';
fid = fopen(file_name);
line1 = strsplit(fgetl(fid),',');
ncol = length(line1); % get number of cols
fclose(fid);
fid = fopen(file_name);
format = ['%s' repmat('%f',1,ncol-1)];
file_data = textscan(fid, format, 'delimiter', ',');
fclose(fid);
genes = file_data{1};
data = cell2mat(file_data(2:end));
M = data';
size(M)

%% library size normalization
libsize  = sum(M,2);
M = bsxfun(@rdivide, M, libsize) * median(libsize);

%% log transform (some data requires log transform)
%M = log(M + 0.1); % 0.1 pseudocount

%% gene to color embeddings by
ind = ismember(genes, 'Actb');
C = log(M(:,ind) + 0.1);
%C = M(:,ind)

%% PCA
Y_pca = svdpca(M, 2, 'random');

%% Plot the embedding colored by gene
figure;
scatter(Y_pca(:,1),Y_pca(:,2),20,C,'filled')
axis tight
xlabel('PCA 1')
ylabel('PCA 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PCA'

%% Run PHATE using CMDS (faster but less accurate)
t = 25;
a = 50;
k = 4;
pca_method = 'random'; % fast PCA
mds_method = 'cmds_fast'; % fast CMDS using random PCA
symm = 'x*xt';
%symm = 'pdist';
ndim = 2;
Y_cmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'log',log_transform,'mds_method',mds_method,'ndim',ndim,'symm',symm);

% Plot the embedding colored by gene
figure;
scatter(Y_cmds(:,1),Y_cmds(:,2),20,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using CMDS X*Xt'

%% Run PHATE using NMMDS with pdist symmetrization (slower but more accurate)
t = 25;
a = 50;
k = 4;
pca_method = 'random'; % fast PCA
mds_method = 'nmmds';
symm = 'pdist';
ndim = 2;
Y_nmmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'log',log_transform,'mds_method',mds_method,'ndim',ndim,'symm',symm);

% Plot the embedding colored by gene
figure;
scatter(Y_nmmds(:,1),Y_nmmds(:,2),20,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS pdist'

%% Run PHATE using NMMDS and X*Xt symmetrization
t = 25;
a = 50;
k = 4;
pca_method = 'random'; % fast PCA
mds_method = 'nmmds';
symm = 'x*xt'; % <--- different symmetrization
ndim = 2;
Y_nmmds2 = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'log',log_transform,'mds_method',mds_method,'ndim',ndim,'symm',symm);

% Plot the embedding colored by gene
figure;
scatter(Y_nmmds2(:,1),Y_nmmds2(:,2),20,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS with X*Xt'

%% plot all embeddings combined
figure;

subplot(2,2,1);
scatter(Y_pca(:,1),Y_pca(:,2),10,C,'filled')
axis tight
xlabel('PCA 1')
ylabel('PCA 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PCA'

subplot(2,2,2);
scatter(Y_cmds(:,1),Y_cmds(:,2),10,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using CMDS (X*Xt)'

subplot(2,2,3);
scatter(Y_nmmds(:,1),Y_nmmds(:,2),10,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS (pdist)'

subplot(2,2,4);
scatter(Y_nmmds2(:,1),Y_nmmds2(:,2),10,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS (X*Xt)'

