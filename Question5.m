clear all;
clc;
 
load MNIST_digit_data.mat
split = 1:500;        %splitting in half for 2 labels
n=1000;
L = 0.01 ; 


X = images_train;
y = labels_train;
 
X_test = images_test;
y_test = labels_test;
 
 
%----- Convert the Labels for 1--> -1 and 6 --> 1(Training and Test)
%----------Training------------

%test
[rows_y_test,~]=size(y_test);
for i= 1:rows_y_test
    if(y_test(i) ==6)
        y_test(i) = 1;
    elseif(y_test(i)==1)
        y_test(i) = -1;
    else
        y_test(i) = 0;
    end
end

%----------Test Starts--------
 
indexsix_test = find(y_test>0);
indexone_test = find(y_test<0);
 
X_new_test_6 = X_test(indexsix_test(split),:);
y_new_test_6 = y_test(indexsix_test(split),:);
X_new_test_1 = X_test(indexone_test(split),:);
y_new_test_1 = y_test(indexone_test(split),:);
 
X_new_test = vertcat(X_new_test_1,X_new_test_6);
y_new_test = vertcat(y_new_test_1,y_new_test_6);
 
 
 
%---Pick random data---------------
rand('seed',1);
random_data = randperm(n)';
 
[x_test_rows,x_test_cols]=size(X_new_test);
[y_test_rows,y_test_cols]=size(y_new_test);


correct =0;

for count=1:10
    [a_weight(count,:),a_b(count,1)] = one_vs_all(X,y,split,L,count,random_data);
end
   
 


%------End of Test---------------------


