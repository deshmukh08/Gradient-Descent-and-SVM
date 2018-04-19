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
    [up_value,up_index] = max(up);
    if(up_index-1 ~= y_test(check))
        tab(check,1) = up_value;
        tab(check,2) = up_index -1;
        tab(check,3) = y_test(check);
        tab(check,4) = check;
    end
    %matrix(up_index,y_test(check)+1) = matrix(up_index,y_test(check)+1) + 1;    
end       
tab_sorted = sortrows(tab,'descend');
for s = 1:10
  fprintf('Predicted Truth=%d and Ground Truth=%d\n',tab_sorted(s,2),tab_sorted(s,3));
end
im = reshape(X_test(tab_sorted(1,4), :), [28 28]);
im = horzcat(im,reshape(X_test(tab_sorted(2,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(3,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(4,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(5,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(6,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(7,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(8,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(9,4), :), [28 28]));
im = horzcat(im,reshape(X_test(tab_sorted(10,4), :), [28 28]));

imshow(im);
