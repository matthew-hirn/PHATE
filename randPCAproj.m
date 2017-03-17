function Y = randPCAproj(X, k)

X = X';
X = bsxfun(@minus,X,mean(X,2));
[U,~,~] = randPCA(X,k);
Y = U'*X;
Y = Y';