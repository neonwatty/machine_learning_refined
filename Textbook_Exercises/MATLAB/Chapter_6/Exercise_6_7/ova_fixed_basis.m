function ova_fixed_basis()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

red =  [ 1 0 .4];
blue =  [ 0 .4 1];
green = [0 1 0.5];
cyan = [1 0.7 0.5];
grey = [.7 .6 .5];
blah = [0.7 0.2 0.7];
colors = [red;blue;green;cyan;blah;grey];     % 4 maximum classes
xmin = 0;
xmax = 1;

% parameters to play with
deg = 2;    % range of poly models to compare

% load data 
[X,y] = load_data();
num_classes = length(unique(y));  % number of classes = number of separators

% make individual classifiers for each class
ot = [xmin:0.002:xmax];
[t1,t2] = meshgrid(ot,ot);
t1 = reshape(t1,numel(t1),1);
t2 = reshape(t2,numel(t2),1);
X2 = [t1,t2];
M = [];
for q = 1:num_classes
    y_temp = y;
    ind = find(y_temp == q);
    ind2 = find(y_temp ~= q);
    y_temp(ind) = 1;
    y_temp(ind2) = -1;

    % do k-fold cross-validation
    F = poly_features(X,deg);
    w = log_loss_newton(F',y_temp);
    subplot(1,num_classes+1,q)
    plot_poly(w,deg,colors(q,:))

    % calculate val
    F = poly_features(X2,deg);
    u = F*w;
    M = [M,u];
end
[f,z] = max(M,[],2);

% plot max separator on the whole thing
t1 = reshape(t1,[length(ot),length(ot)]);
t2 = reshape(t2,[length(ot),length(ot)]);   
z = reshape(z,[length(ot),length(ot)]);   

subplot(1,num_classes + 1,num_classes + 1)
for i = 1:num_classes - 1
   hold on
   contour(t1,t2,z,[i + 0.5,i + 0.5],'Color','k','LineWidth',2)
end
axis([xmin xmax xmin xmax])
axis square
xlabel('x_1','Fontsize',18,'FontName','cmmi9')
ylabel('x_2  ','Fontsize',18,'FontName','cmmi9')
set(gca,'XTick',[xmin xmax])
set(gca,'YTick',[xmin xmax])
set(get(gca,'YLabel'),'Rotation',0)
set(gcf,'color','w');   
box off

%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%
    
%%% builds (poly) features based on input data %%%
function F = poly_features(data,deg)

% ------>  YOUR CODE GOES HERE

end

%%% plots learned model %%%
function plot_poly(w,deg,color)
    % Generate poly seperator
    o = [xmin:0.01:xmax];
    [s,t] = meshgrid(o,o);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    D = poly_features([s,t],deg);
    f = D*w;

    s = reshape(s,[length(o),length(o)]);
    t = reshape(t,[length(o),length(o)]);
    f = reshape(f,[length(o),length(o)]);  
    % plot contour in original space
    hold on
    contour(s,t,f,[0,0],'Color',color,'LineWidth',2)
    axis([xmin xmax xmin xmax])
    axis square
    
    xlabel('x_1','Fontsize',18,'FontName','cmmi9')
    ylabel('x_2  ','Fontsize',18,'FontName','cmmi9')
    set(gca,'XTick',[xmin xmax])
    set(gca,'YTick',[xmin xmax])
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');   
    box off

end

%%% plots points for each fold %%%
function plot_pts(X,y,c,j)
    
    % plot training set
    ind = find(c ~= j);
    ind2 = find(y(ind) == 1);
    ind3 = ind(ind2);
    red =  [ 1 0 .4];

    plot(X(ind3,1),X(ind3,2),'o','Linewidth',2.5,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)
    hold on
    ind2 = find(y(ind) == -1);
    ind3 = ind(ind2);
    blue =  [ 0 .4 1];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',2.5,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
    
    % plot test set?
    ind = find(c == j);
    ind2 = find(y(ind) == 1);
    ind3 = ind(ind2);
    red =  [ 1 0 .4];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',1,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)

    hold on
    ind2 = find(y(ind) == -1);
    ind3 = ind(ind2);
    blue =  [ 0 .4 1];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',1,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
end

%%% plots (mean) training/testing errors %%%
function plot_errors(poly_degs, test_errors, train_errors)
    % plot training and testing errors
    % plot mean errors
    plot(1:max(poly_degs),mean(test_errors'),'--','Color',[1 0.7 0])
    hold on
    plot(1:max(poly_degs),mean(test_errors'),'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    plot(1:max(poly_degs),mean(train_errors'),'--','Color',[0.1 0.8 1])
    hold on
    plot(1:max(poly_degs),mean(train_errors'),'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])

    % clean up plot
    set(gcf,'color','w');
    box off
    s = mean(test_errors');
    axis([0.5 6 0 max(s(1:6))])
    axis square
    xlabel('M','Fontsize',18,'FontName','cmr10')
    ylabel('average error','Fontsize',18,'FontName','cmr10')
    
    set(get(gca,'YLabel'),'Rotation',90)
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
end

%%% load data %%%
function [X,y] = load_data()
    % load data from file
    data = csvread('bullseye_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);

     % how many classes in the data?  4 maximum for this toy.
    class_labels = unique(y);           % class labels
    num_classes = length(class_labels);
    for j = 1:num_classes
        % plot data
        subplot(1,num_classes + 1,j)
        ind = find(y ~= j);
        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',grey,'markerFacecolor','none');

        ind = find(y == j);
        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',colors(j,:),'markerFacecolor','none');
       
        subplot(1,num_classes + 1,num_classes + 1)
        hold on
        scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',colors(j,:),'markerFacecolor','none');
        blah = 0;
    end
    blah = 0;
end

%%% newton's method for log-loss classifier %%%
function w = log_loss_newton(D,b)
    % initialize
    w = randn(size(D,1),1);

    % precomputations
    H = diag(b)*D';
    grad = 1;
    n = 1;

    %%% main %%%
    while n <= 30 && norm(grad) > 10^-5

        % prep gradient for logistic objective
        r = sigmoid(-H*w);
        g = r.*(1 - r);
        grad = -H'*r;
        hess = D*diag(g)*D';

        % take Newton step
        s = hess*w - grad;
        w = pinv(hess)*s;
        n = n + 1;
    end

end

%%% sigmoid function for use with log_loss_newton %%%
function t = sigmoid(z)
    t = 1./(1+exp(-z));
end

%%% evaluates error of a learned model %%%
function score = evaluate(A,b,w)
    s = A*w;
    ind = find(s > 0);
    s(ind) = 1;
    ind = find(s <= 0);
    s(ind) = -1;
    t = s.*b;
    ind = find(t < 0);
    t(ind) = 0;
    score = 1 - sum(t)/numel(t);

end

end





