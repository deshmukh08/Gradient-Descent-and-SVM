function [ w,b ] = one_vs_all( X,y,split,L,count,random_data )


[rows_y,~]=size(y);
for i= 1:rows_y
    if(y(i) ==count-1)
        y(i) = 1;
    else
        y(i) = -1;
    end
end

indexone = find(y>0);
indexsix = find(y<0);

X_new_train_6 = X(indexsix(:),:);
y_new_train_6 = y(indexsix(:),:);
X_new_train_1 = X(indexone(:),:);
y_new_train_1 = y(indexone(:),:);
 
X_new_train = vertcat(X_new_train_1,X_new_train_6);
y_new_train = vertcat(y_new_train_1,y_new_train_6);

w = zeros(1,784);
b=0;
for o = 1:1000
    g_wts =zeros(1,784);
    g = 0;
        if y(o,1)*(dot(w(1,:),X_new_train(random_data(o),:)) + b) <= 1
            g_wts = g_wts + y_new_train(random_data(o),1) * X_new_train(random_data(o),:);
            g = g + y_new_train(random_data(o),1);
        end
    
    eta = 1/o;
    g_wts = g_wts - L*(w);
    w = w + eta*g_wts;
    b = b + eta*g;
end


end
