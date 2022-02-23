
% X=rand(15,20);
% labels=[-ones(4,1);2*ones(5,1);3*ones(6,1)];
% Y=rand(10,20);
% %A=sparse_represent(Y,X,0.3);
% [predictions,src_scores]=src(X,labels,Y,0.3)
%src(Traindata,Trainlabels,Testdata,sp_level)
clear
rng(1)
load YaleB_32x32.mat
load PCA_YB.csv

YaleB = [PCA_YB, gnd];
K = 38;
m = 10;
train_10 = [];
test_10 = [];
for i = 1:K
    G = YaleB(YaleB(:,6) == i, 1:6); % extract data of class i
    idx = randperm(size(G,1),m); %randomly generate q integers 1 to size(G,1)
    vec = [1: size(G,1)]; % Need this to extract test set
    test_idx = setdiff(vec,idx);

    test_set = G(test_idx,:);
    test_10 = [test_10; test_set];


    train = G(idx,:);
    train_10 = [train_10;train];
end
X_train = train_10(:,1:5);
Y_train = train_10(:,6);
X_test = test_10(:,1:5);
Y_test = test_10(:,6);

[predictions,src_scores]=src(X_train,Y_train,X_test,0.3);