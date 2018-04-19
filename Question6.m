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

[rows_y,~]=size(y);

rand('seed',1);
random_data = randperm(n)';

[rows_y_test,~]=size(y_test);

matrix = zeros(10,10);
for incr = 1 : 10
   [a_weight(incr,:),a_b(incr,1)] = one_vs_all(X,y,split,L,incr,random_data);
end
for check = 1 : rows_y_test
    for a = 1:10
        up(a,1) = dot(X_test(check,:),a_weight(a,:)) + a_b(a,1);
    end
    [~,up_index] = max(up);
    matrix(up_index,y_test(check)+1) = matrix(up_index,y_test(check)+1) + 1;    
end
for a = 1:10
    matrix_updated(a,:) =  matrix(a,:) / norm(matrix(a,:));
end
temp =trace(matrix_updated)/10;
fprintf('Accuracy for the data is %2.9f\n',temp);

