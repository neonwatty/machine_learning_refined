function one_versus_all_demo_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data
red =  [ 1 0 .4];
blue =  [ 0 .4 1];
green = [0 1 0.5];
cyan = [1 0.7 0.5];
grey = [.7 .6 .5];
colors = [red;blue;green;cyan;grey];     % 4 maximum classes
[X,y] = load_data();

% find separators
W = learn_separators(X,y);

% % plot separators one-at-a-time
figure(1)
plot_one_ata_time(W,X,y)

% % plot the max(..) separator
% figure(2)
% plot_max_separator(W,X,y)
    
%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%

%%% learn separator for each class %%%
function W = learn_separators(X,y)  
    class_labels = unique(y);           % class labels
    num_classes = length(unique(y));
    W = [];         % container for all the weights to learn
    for i = 1:num_classes
        % setup temporary labels for one-vs-all classification       
        y_temp = y;
        class = class_labels(i);
        ind = find(y_temp == class);
        ind2 = find(y_temp ~= class);
        y_temp(ind) = 1;
        y_temp(ind2) = -1;
        
        % run newton's method and store resulting weights
        w = softmax_newtons_method(X,y_temp);
        W = [W, w];
    end

end

function w = softmax_newtons_method(X,y)
% ----->  your code goes here
end

%%% plot full dataset %%%
function plot_data(X,y)
    % how many classes in the data?  4 maximum for this toy.
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

%%% plot max separator %%%
function plot_max_separator(W,X,y,deg)
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

%%% plot learned separators one at a time
function plot_one_ata_time(W,X,y)

    %%% plot one separator at a time %%%
    class_labels = unique(y);           % class labels
    num_classes = length(unique(y));
    x = [0:0.01:1];

    for i = 1:num_classes
        subplot(1,num_classes+1,i)
        w = W(:,i);    
        plot (x,(-w(1)-w(2)*x)/w(3),'Color',colors(i,:),'linewidth',2);

        class = class_labels(i);
        ind = find(y == class);
        hold on
        scatter(X(2,ind),X(3,ind),'Linewidth',2,'Markeredgecolor',colors(i,:),'markerFacecolor','none');
        hold on
        ind = find(y ~= class);
        scatter(X(2,ind),X(3,ind),'Linewidth',2,'Markeredgecolor',[0.5 0.5 0.5],'markerFacecolor','none');
        axis([0 1 0 1])
        axis square
        box on

    end
    
    %%% plot fused separator %%%
    subplot(1,num_classes+1,num_classes+1)
    plot_data(X,y)
    
    % generate max-separator surface 
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
    ind = find(z == 1);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',red,'FaceAlpha',0.2)

    ind = find(z == 2);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 2*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',blue,'FaceAlpha',0.2)
    
    ind = find(z == 3);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 3*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',green,'FaceAlpha',0.2)
    
    ind = find(z == 4);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 4*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',cyan,'FaceAlpha',0.2)
    
    ind = find(z == 5);
    k = boundary(square(ind,:));
    v = [square(ind(k),:), 4*ones(length(k),1)];
    f = 1:length(k);
    patch('Faces',f,'Vertices',v,'FaceColor',grey,'FaceAlpha',0.2)
    
    
    % produce decision boundary
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);   
    z = reshape(z,[length(s),length(s)]);   
    num_classes = length(unique(z));
    for i = 1:num_classes - 1
       hold on
       contour(s1,s2,z,[i + 0.5,i + 0.5],'Color','k','LineWidth',2)
    end

    % make plot real nice lookin'
    for i = 1:num_classes + 1
        subplot(1,num_classes + 1,i)
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
    set(gcf,'color','w');
end

%%% load data %%%
function [X,y] = load_data()
    % load data from file
    data = csvread('four_class_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);   
    X = [ones(size(X,1),1), X];
    X = X';
end
end