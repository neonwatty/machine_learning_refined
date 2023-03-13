function exp_vs_log_demo_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


%%% load data %%%
[X,y] = load_data();

X_tilde = [ones(size(X,1),1) X]';   % use compact notation
w0 = randn(3,1);                    % random initial point
alpha = .01;                        % fixed steplength

%%% run gradient descent for h1 %%%
w1 = grad_descent_soft_cost(X_tilde,y,w0,alpha);

%%% run gradient descent for h2 %%%
w2 = grad_descent_exp_cost(X_tilde,y,w0,alpha);

%%% plot everything, pts and lines %%%
plot_all(X',y,w1,'k');
plot_all(X',y,w2,'m');


%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%
%%% gradient descent function for h1 %%%
function w = grad_descent_soft_cost(X,y,w,alpha)
  
    % Initializations 
    iter = 1;
    max_its = 30000;
    grad = 1;
    
    while  norm(grad) > 10^-12 && iter < max_its
        % compute gradient
% --->  grad = 
        w = w - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% gradient descent function for h2 %%%
function w = grad_descent_exp_cost(X,y,w,alpha)
  
    % Initializations 
    iter = 1;
    max_its = 30000;
    grad = 1;
    
    while  norm(grad) > 10^-12 && iter < max_its
        % compute gradient
% --->  grad = 
        w = w - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(X,y,w,color)
    red = [1 0 .4];
    blue =  [ 0 .4 1];

    % plot points 
    ind = find(y == 1);
    scatter(X(1,ind),X(2,ind),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none');
    hold on
    ind = find(y == -1);
    scatter(X(1,ind),X(2,ind),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none');
    hold on

    % plot separator
    s =[0:0.01:1 ];
    plot (s,(-w(1)-w(2)*s)/w(3),color,'linewidth',2);
    
    % clean up plot and add info labels
    set(gcf,'color','w');
    axis square
    box off
    axis([0.3 1 0.3 1])
    xlabel('x_1','Fontsize',14)
    ylabel('x_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
end

%%% loads data %%%
function [X,y] = load_data()
    data = csvread('exp_vs_log_data.csv');
    X = data(:,1:end-1);
    y = data(:,end);
end

end
