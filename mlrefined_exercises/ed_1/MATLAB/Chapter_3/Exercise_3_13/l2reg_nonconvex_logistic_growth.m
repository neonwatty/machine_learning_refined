function l2reg_nonconvex_logistic_growth()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data, plot data and surfaces
[X,y] = load_data();

%%% run grad descent with 2 starting points and plot descent path %%%
lams = [0.1];

for q = 1:length(lams)
    % plot pts and surface
    figure(q)
    
    [s,t,non_obj] = plot_surface(X,y,lams(q));

    % run grad descent with first starting point
    w0 = [0;2];
    [in1,out1] = grad_descent(X,y,w0,lams(q));
     
    % run grad descent with second starting point
    w0 = [0;-2];
    [in2,out2] = grad_descent(X,y,w0,lams(q));
    
    % plot results
    subplot(1,3,1)
    plot_pts(X,y);
    hold on
    plot_sigmoid(X,y,in1(end,:),1);
    hold on
    plot_sigmoid(X,y,in2(end,:),2);
    axis([0 max(X(:,2)) 0 1])
    axis square
    box on
    
    subplot(1,3,2)
    plot_descent_path(in1,out1,1,s,t,non_obj);
    hold on
    plot_descent_path(in2,out2,2,s,t,non_obj);
    axis square
    box on
      
    subplot(1,3,3)
    [s,t,non_obj] = plot_surface(X,y,lams(q));
    hold on
    plot3(in1(:,1),in1(:,2),out1,'m','LineWidth',3);
    hold on
    plot3(in2(:,1),in2(:,2),out2,'g','LineWidth',3);
    axis square
    box on

end



%%%%%%%%%% subfunctions %%%%%%%%%
% perform grad descent 
function [in,out] = grad_descent(X,y,w,lam)
    
    % Initializations 
    in = [w];
    d = obj(w);
    out = [d];
    grad = 1;
    iter = 1;
    max_its = 5000;
    
    % main loop
    while  norm(grad) > 10^-5 && iter < max_its

        % fixed steplength
  
% ----> grad = ...
    
        d = obj(w);
        alpha = 1e-2;
        w = w - alpha*grad;

        % update containers
        in = [in w];
        out = [out ; d];
        iter = iter + 1;
    end
    in = in';
    
    function d = obj(s)
         d = norm(1./(1 + exp(-X*s)) - y)^2 + lam*s(2)^2;
    end
end

function [s,t,non_obj] = plot_surface(X,y,lam)
    % setup surface
    range = 3;                     % range over which to view surfaces
    [s,t] = meshgrid(-range:0.25:range);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    non_obj = zeros(length(s),1);   % nonconvex surface

    % build surface
    for i = 1:length(y)
        non_obj = non_obj + non_convex(X(i,:),y(i),s,t)';
    end
    non_obj = non_obj + lam*t.^2;
    
    % plot surface
    subplot(1,3,2)
    set(gcf,'color','w');
    r = sqrt(numel(s));
    s = reshape(s,r,r);
    t = reshape(t,r,r);
    non_obj = reshape(non_obj,r,r);
    surf(s,t,non_obj)
    box off
    view(-100,10)
    zlabel('g ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'ZLabel'),'Rotation',0)

    % plot contour
    subplot(1,3,3)
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
    
    xlabel('w ','Fontsize',18,'FontName','cmr10')
    ylabel('b ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',0)
    set(get(gca,'XLabel'),'Rotation',0)
    hold on

    
    function s = non_convex(c,z,s,t)    % objective function for nonconvex problem
         s = (sigmoid(c*[s,t]') - z).^2;
    end
end

function plot_pts(A,b)
    % plot labeled points
    scatter(A(:,2),b,'fill','k')
    set(gcf,'color','w');
    xlabel('x','Fontsize',18,'FontName','cmr10')
    ylabel('y  ','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box on
end

function plot_sigmoid(X,y,w,i)
    % plot
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
    hold on
    if i == 1
       plot3(in(:,1),in(:,2),out,'m','LineWidth',3);
    else
       plot3(in(:,1),in(:,2),out,'g','LineWidth',3);
    end
    axis([min(min(t)) max(max(t)) min(min(s)) max(max(s)) min(min(non_obj)) max(max(non_obj))])

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
    a = data(:,1);
    y = data(:,2);
    X = [ones(length(a),1) a];

end

function sig = sigmoid(z)
    sig = 1./(1+exp(-z));
end

end
