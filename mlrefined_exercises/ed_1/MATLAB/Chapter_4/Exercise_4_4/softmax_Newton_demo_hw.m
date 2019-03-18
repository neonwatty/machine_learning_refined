function softmax_Newton_demo_hw()

% softmax_Newton_demo_hw is a simple linear classification demo with softmax-loss
% on simulated data, with tanh surface (logistic regression interpretation)
% plotted

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% get data 
xmin = 0;
xmax = 1;
ymin = 0;
ymax = 1;
[X,y] = load_data();

% get weights
w0 = [3;-1;40];    % initial point
w = softmax_newton(X',y,w0);

% plot separator
plot_separator(w);

%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%
%%% newton's method algorithm %%%
function w = softmax_newton(X,y,w)

    grad = 1;
    k = 1;

    %%% main %%%
    while k <= 30 && norm(grad) > 10^-5

        % take Newton step
% ----> w = 
        k = k + 1;
    end

end

%%% sigmoid for softmax loss optimization %%%
function t = sigmoid(z)
    t = 1./(1+exp(-z));
end

%%% plot the seprator + surface %%%
function plot_separator(w)
    
    % plot determined surface in 3d space
    s = [xmin:0.01:xmax];
    [s1,s2] = meshgrid(s,s);
    s1 = reshape(s1,numel(s1),1);
    s2 = reshape(s2,numel(s2),1);
    
    f = zeros(length(s1),1);
    for i = 1:length(s1)
        p = [s1(i);s2(i)];
        m = [1 ; p]'*w;
        f(i) = 2*sigmoid(m) - 1;
    end
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);
    f = reshape(f,[length(s),length(s)]);  % divide by # for visualization purposes only!
   
    subplot(1,2,2)
    hold on
    surf(s1, s2, f, 'LineStyle', 'none', 'FaceColor', 'interp')    
    colormap gray  
    alpha(0.4)
    camproj('perspective') % for some reason this removes axis labels!  But does make the surface look a bit nicer.
    light;
    axis square
    view(55,22)
    
    % plot contour in original space
    subplot(1,2,1)
    hold on
    contour(s1,s2,f,[0,0],'Color','k','LineWidth',2)
    box off

end

%%% load data and plot %%%
function [X,y] = load_data()
    % generating simulated data set
    data = csvread('overlapping_2class.csv');
    X = data(:,1:end-1);
    y = data(:,end); 
    
    % plot data 
    subplot(1,2,1)
    hold on
    ind = find(y == 1);
    red =  [ 1 0 .4];
    scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none')
    hold on
    ind = find(y == -1);
    blue =  [ 0 .4 1];
    scatter(X(ind,1),X(ind,2),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none')
    set(gcf,'color','w');
  
    % graph info labels
    xlabel('x_1','Fontsize',18,'FontName','cmmi9')
    ylabel('x_2  ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    axis([xmin xmax ymin ymax]);
    axis square
    box on
    
    subplot(1,2,2)
    ind = find(y == 1);
    red =  [ 1 0 .4];
    plot3(X(ind,1),X(ind,2),y(ind),'o','MarkerEdgeColor',red,'MarkerFaceColor','none','LineWidth',2)
    hold on
    ind = find(y == -1);
    blue =  [ 0 .4 1];
    plot3(X(ind,1),X(ind,2),y(ind),'o','MarkerEdgeColor',blue,'MarkerFaceColor','none','LineWidth',2)
    set(gcf,'color','w');
    xlabel('x_1','Fontsize',18,'FontName','cmmi9')
    ylabel('x_2','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(get(gca,'YLabel'),'Rotation',0)
    zlabel('y   ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'ZLabel'),'Rotation',0)
    X = [ones(size(X,1),1) X];

end

end
