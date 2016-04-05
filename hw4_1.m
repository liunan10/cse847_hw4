clear;
% read spam_email data
data =  dlmread('spam_email/data.txt');
labels = dlmread('spam_email/labels.txt');
data = [data, ones(size(data,1),1)];

% create a separate test data set
data_test = data(2001:4601,:);
labels_test = labels(2001:4601,:);

% choose the first n rows of the training data
n = [200 500 800 1000 1500 2000];
accuracy = zeros(length(n),1);

for i=1:length(n)
    data_train = data(1:n(i),:);
    labels_train = labels(1:n(i),:);
    % training
    w = logistic_train(data_train, labels_train);
    % testing
    y_hat = logsig(data_test*w);
    y_hat(y_hat>=0.5) = 1;
    y_hat(y_hat<0.5) = 0;
    accuracy(i) = mean(y_hat==labels_test);
end

% plot accuracy
figure;
plot(n, accuracy,'x-');
xlabel('Training size');
ylabel('Accuracy');
box on;
