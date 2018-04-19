clear all;
clc;
 
load MNIST_digit_data.mat
split = 1:500;        %splitting in half for 2 labels
n=1000;
iterations =1000;
L = 0.01 ; 


X = images_train;
y = labels_train;
 
X_test = images_test;
y_test = labels_test;
 
 
%----- Convert the Labels for 1--> -1 and 6 --> 1(Training and Test)
%----------Training------------
[rows_y,~]=size(y);
for i= 1:rows_y
    if(y(i) ==6)
        y(i) = 1;
    elseif(y(i)==1)
        y(i) = -1;
    else
        y(i) = 0;
    end
end
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
 
indexsix = find(y>0);
indexone = find(y<0);
 
X_new_train_6 = X(indexsix(split),:);
y_new_train_6 = y(indexsix(split),:);
X_new_train_1 = X(indexone(split),:);
y_new_train_1 = y(indexone(split),:);
 
X_new_train = vertcat(X_new_train_1,X_new_train_6);
y_new_train = vertcat(y_new_train_1,y_new_train_6);
 
%------Training ends--------------
 
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
 

[x_train_rows,x_train_cols]=size(X_new_train);
[y_train_rows,y_train_cols]=size(y_new_train);

[x_test_rows,x_test_cols]=size(X_new_test);
[y_test_rows,y_test_cols]=size(y_new_test);

%----- Initialize Weights and Bias to Zero------
w = zeros(1,x_train_cols);
b=0;


%---------Training Model-------

 
for o = 1:x_train_rows
    g_wts =zeros(1,x_train_cols);
    g = 0;
        if y(o,1)*(dot(w(1,:),X_new_train(random_data(o),:)) + b) <= 1
            g_wts = g_wts + y_new_train(random_data(o),1) * X_new_train(random_data(o),:);
            g = g + y_new_train(random_data(o),1);
        end
    
    eta = 1/o;
    g_wts = g_wts - L*(w);
    w = w + eta*g_wts;
    b = b + eta*g;


%---------------end of training model--------
%--------Testing model---------

correct =0;
    for i=1:1000
        %--------a_test = predictn(X_new_test(i,:),w,b);
        %-----(y_new_train(random_data(o),1))
         %--- if(y_new_test(i) == a_test)



        if y_new_test(i,1)*(dot(X_new_test(i,:),w(1,:))+b) >= 1

            correct = correct + 1;
        end
    end
    accuracy(o) = (correct/1000)*100;
end
%------End of Test---------------------

plot(1:1000,accuracy);
