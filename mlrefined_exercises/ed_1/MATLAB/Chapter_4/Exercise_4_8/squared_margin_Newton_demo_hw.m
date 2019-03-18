function squared_margin_Newton_demo_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


%%% load data %%%
[X,y] = load_data();

w0 = randn(3,1);     % initial point
w = squared_margin_newton(X,y,w0);

%%% plot everything, pts and lines %%%
plot_all(X(2:3,:)',y,w);


%%% Newton's method for squared hinge loss %%%
function w = squared_margin_newton(X,y,w)
    % Initializations 
    iter = 1;
    max_its = 30;
    grad = 1;

    while  norm(grad) > 10^-8 && iter < max_its
        
        % take Newton step
% ----> w =
        
        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(data,y,w)
    red = [1 0 .4];
    blue =  [ 0 .4 1];

    % plot points 
    ind = find(y == 1);
    scatter(data(ind,1),data(ind,2),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none');
    hold on
    ind = find(y == -1);
    scatter(data(ind,1),data(ind,2),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none');
    hold on

    % plot separator
    s =[0:0.01:1];
    plot (s,(-w(1)-w(2)*s)/w(3),'m','linewidth',2);
    hold on

    % make plot nice looking
    set(gcf,'color','w');
    axis square
    box off
    
    % graph info labels
    xlabel('w_1','Fontsize',14)
    ylabel('w_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
    axis([(min(X(2,:)) - 0.05) (max(X(2,:)) + 0.05) (min(X(3,:)) - 0.05) (max(X(3,:)) + 0.05)]);
end

%%% loads data %%%
function [A,b] = load_data()
    data = load('overlapping_2class.csv');
    A = data(:,1:end-1);
    A = [ones(size(A,1),1) A]';
    b = data(:,end);
end

end
