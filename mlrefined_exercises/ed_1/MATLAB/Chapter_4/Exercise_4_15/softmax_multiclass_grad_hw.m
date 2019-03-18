function softmax_multiclass_grad_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
[X,y] = load_data();


% initializations
N = size(X,2);
C = length(unique(y));
X = [ones(size(X,1),1), X]';
W0 = randn(N+1,C);
alpha = 0.1;


% find separators via multiclass softmax classifier
W = softmax_multiclass_grad(X,y,W0,alpha);


% plot the separators as well as final classification
plot_separators(W, X, y)


%%%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%%%%%%%
function W = softmax_multiclass_grad(X,y,W0,alpha)
  
    % initialize
    max_its = 10000; 
    [N,P] = size(X);
    C = length(unique(y));
    W = W0;
    k = 1;

    %%% main %%%
    while k <= max_its
   

% ----> grad = 
        W = W - alpha*grad;
        
        % update counter
        k = k + 1;
        
    end
end

function [X,y] = load_data()
    data = csvread('4class_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);       
end

function plot_separators(W,X,y,deg)
    
    red =  [ 1 0 .4];
    blue =  [ 0 .4 1];
    green = [0 1 0.5];
    cyan = [1 0.7 0.5];
    grey = [.7 .6 .5];
    colors = [red;blue;green;cyan;grey]; 
    % plot data
    subplot(1,3,1)
    plot_data(X,y)
    subplot(1,3,2)
    plot_data(X,y)
    subplot(1,3,3)
    plot_data(X,y)

    %%% plot all linear separators %%%
    subplot(1,3,2)
    num_classes = length(unique(y));
    x = [0:0.01:1];
    for j = 1:num_classes
        hold on
        w = W(:,j);    
        plot (x,(-w(1)-w(2)*x)/w(3),'Color',colors(j,:),'linewidth',2);
    end

    %%% generate max-separator surface %%%
    s = [0:0.005:1];
    [s1,s2] = meshgrid(s,s);
    s1 = reshape(s1,numel(s1),1);
    s2 = reshape(s2,numel(s2),1);
    
    % compute criteria for each point in the range [0,1] x [0,1]
    square = [s1(:), s2(:)];
    p = [ones(size(s1(:),1),1),s1(:), s2(:)];
    f = W'*p';         
    [f,z] = max(f,[],1);
    
    % fill in appropriate regions with class colors
    subplot(1,3,3)
    ind = find(z == 1);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',red,'FaceAlpha',0.3)

    ind = find(z == 2);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 2*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',blue,'FaceAlpha',0.3)
    
    ind = find(z == 3);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 3*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',green,'FaceAlpha',0.3)
    
    ind = find(z == 4);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 4*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',cyan,'FaceAlpha',0.3)
    
    ind = find(z == 5);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 4*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',grey,'FaceAlpha',0.3)
    
    % produce decision boundary
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);   
    z = reshape(z,[length(s),length(s)]);   
    num_classes = length(unique(z));
    subplot(1,3,3)
    for i = 1:num_classes - 1
       hold on
       contour(s1,s2,z,[i + 0.5,i + 0.5],'Color','k','LineWidth',2)
    end
    
    % make plot real nice lookin'
    for i = 1:3
        subplot(1,3,i)
        axis([0 1 0 1])
        axis square
        xlabel('x_1','FontName','cmmi9','Fontsize',18)
        ylabel('x_2','FontName','cmmi9','Fontsize',18)
        set(get(gca,'YLabel'),'Rotation',0)
        zlabel('y','FontName','cmmi9','Fontsize',18)
        set(get(gca,'ZLabel'),'Rotation',0)

        set(gca,'XTick',[0,1])
        set(gca,'YTick',[0,1])
        set(gca,'ZTick',[0:1:num_classes])
        set(gcf,'color','w');
    end
end

function plot_data(X,y)
    red =  [ 1 0 .4];
    blue =  [ 0 .4 1];
    green = [0 1 0.5];
    cyan = [1 0.7 0.5];
    grey = [.7 .6 .5];
    colors = [red;blue;green;cyan;grey]; 
    % how many classes in the data? maximum 4 here.
    class_labels = unique(y);           % class labels
    num_classes = length(class_labels);

    % plot data
    for i = 1:num_classes
        class = class_labels(i);
        ind = find(y == class);
        hold on
        scatter3(X(2,ind),X(3,ind),class*ones(length(ind),1),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');
        hold on
        scatter3(X(2,ind),X(3,ind),class*ones(length(ind),1),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');
    end
    axis([0 1 0 1])
    axis square
    box on
end
end