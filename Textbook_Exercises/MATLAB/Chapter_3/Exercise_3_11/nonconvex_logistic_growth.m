function nonconvex_logistic_growth()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data, plot data and surfaces
[X,y] = load_data();
[s,t,non_obj] = plot_surface(X,y);
plot_pts(X,y);

%%% run grad descent with 2 starting points and plot descent path %%%

% run grad descent with first starting point
w0 = [0;2];
[in,out] = grad_descent(X,y,w0);

% plot results
plot_sigmoid(X,y,in(end,:),1);
plot_descent_path(in,out,1,s,t,non_obj);

% run grad descent with second starting point
w0 = [0;-2];
[in,out] = grad_descent(X,y,w0);

% plot results
plot_sigmoid(X,y,in(:,end),2)
plot_descent_path(in,out,2,s,t,non_obj)

% perform grad descent
function [in,out] = grad_descent(X,y,w)
    % step length
    alpha = 10^-2;

    % Initializations
    in = [w];
    out = [norm(1./(1 + exp(-X*w)) - y)^2];
    grad = 1;
    iter = 1;
    max_its = 10000;

    % main loop
    while  norm(grad) > 10^-5 && iter < max_its

        % fixed steplength
% ----> grad = 
        w = w - alpha*grad;

        % update containers
        in = [in w];
        out = [out ; norm(1./(1 + exp(-X*w)) - y)^2];
        iter = iter + 1;
    end
    in = in';
end

function [s,t,non_obj] = plot_surface(X,y)
    % setup surface
    range = 3;                     % range over which to view surfaces
    [s,t] = meshgrid(-range:0.2:range);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    non_obj = zeros(length(s),1);   % nonconvex surface

    % build surface
    for i = 1:length(y)
        non_obj = non_obj + non_convex(X(i,:),y(i),s,t)';
    end

    % plot surface
    figure(2)
    subplot(1,2,1)
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    set(gcf,'color','w');
    r = sqrt(numel(s));
    s = reshape(s,r,r);
    t = reshape(t,r,r);
    non_obj = reshape(non_obj,r,r);
    surf(s,t,non_obj)
    box on
    xlabel('w','Fontsize',18,'FontName','cmmi9')
    ylabel('b','Fontsize',18,'FontName','cmmi9')
    zlabel('g','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'ZLabel'),'Rotation',0)

    % plot contour
    subplot(1,2,2)
    set(gcf,'color','w');
    r = sqrt(numel(s));
    s = reshape(s,r,r);
    t = reshape(t,r,r);
    non_obj = reshape(non_obj,r,r);
    contourf(s,t,non_obj,10)
    box on
    xlabel('w','Fontsize',18,'FontName','cmmi9')
    ylabel('b','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
end

function plot_pts(X,y)
    % plot labeled points
    figure(1)
    scatter(X(:,2),y,'fill','k')
    set(gcf,'color','w');
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box on
    set(gca,'FontSize',12);
end

function plot_sigmoid(X,y,w,i)
    % plot
    figure(1)
    hold on
    u = [0:0.1:max(X(:,2))];
    w = 1./(1+exp(-(u*w(2) + w(1))));
    if i == 1
        plot(u,w,'m','LineWidth',2);
    else
        plot(u,w,'g','LineWidth',2);
    end
end

function plot_descent_path(in,out,i,s,t,non_obj)

    % plot nonconvex-output path
    figure(2)
    subplot(1,2,1)
    hold on
    if i == 1
       plot3(in(:,1),in(:,2),out,'m','LineWidth',3);
    else
       plot3(in(:,1),in(:,2),out,'g','LineWidth',3);
    end
    axis([min(min(t)) max(max(t)) min(min(s)) max(max(s)) min(min(non_obj)) max(max(non_obj))])
    subplot(1,2,2)
    hold on
    if i == 1
       plot3(in(:,1),in(:,2),out,'m','LineWidth',3);
    else
       plot3(in(:,1),in(:,2),out,'g','LineWidth',3);
    end
end

% loads data for processing
function [X,y] = load_data()
    % load bacteria data
    data = csvread('bacteria_data.csv');
    x = data(:,1);
    y = data(:,2);
    X = [ones(length(x),1) x];
end

function s = non_convex(c,z,s,t)    % objective function for nonconvex problem
    s = (sigmoid(c*[s,t]') - z).^2;
end

function y = sigmoid(z)
y = 1./(1+exp(-z));
end

end
