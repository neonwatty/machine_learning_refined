function compare_maps_regression_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
data = csvread('noisy_sin_samples.csv');
x = data(:,1);
y = data(:,2);

% true underlying data-generating function
x_true = [0:0.01:1]';
y_true = sin(2*pi*x_true);

% parameters to play with
k = 3;                  % # of folds to use

% split points into k equal sized sets
c = split_data(x,y,k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do k-fold cross-validation using polynomial basis
poly_degs = 1:10;           % range of poly models to compare
deg = cross_validate_poly(x,y,c,poly_degs,k);

% plot it
subplot(1,3,1)
plot_poly(x,y,deg,k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do k-fold cross-validation using fourier basis
fourier_degs = 1:10;           % range of fourier models to compare
deg = cross_validate_fourier(x,y,c,fourier_degs,k);

% plot it
subplot(1,3,2)
plot_fourier(x,y,deg,k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%do k-fold cross-validation using tanh basis
tanh_degs = 1:6;           % range of NN models to compare
deg = cross_validate_tanh(x,y,c,tanh_degs,k);

% plot it
subplot(1,3,3)
plot_tanh(x,y,deg,k);


%%% modules to complete %%%

function c = split_data(x,y,k)
  
    % --->  YOUR CODE GOES HERE

end

function deg = cross_validate_poly(x,y,c,poly_degs,k)  

      % --->  YOUR CODE GOES HERE
  
end

function deg = cross_validate_fourier(x,y,c,fourier_degs,k)  

      % --->  YOUR CODE GOES HERE
  
end

function deg = cross_validate_tanh(x,y,split,tanh_degs,k)  

      % --->  YOUR CODE GOES HERE
  
end


%%% already completed modules %%%

function plot_poly(x,y,deg,k)

    % calculate weights
    X = build_poly(x,deg);
    w = linsolve(X,y);
    % output model
    in = [0:0.01:1]';
    out = zeros(length(in),1);
    for n = 0:deg
        out = out + w(n+1)*in.^n;
    end

    % plot 
    plot(x_true,y_true,['.','k'],'LineWidth',1.5)
    hold on
    scatter(x,y,40,'k','fill')
    hold on
    plot(in,out,'b','LineWidth',2)   

    % clean up plot
    %axis square
    axis([0 1 (min([y;y_true]) - 0.1) (max([y;y_true]) + 0.1)])
    axis square
    box off
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y  ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    title(['k = ', num2str(k), ',   deg = ', num2str(deg)]);
    
end

function plot_fourier(x,y,deg,k)
      
    X = build_fourier(x,deg);
    w = pinv(X'*X)*X'*y;

    % output model
    period = 1;
    in = [0:0.01:1]';
    out = w(1)*ones(length(in),1);
    for n = 1:deg
        out = out + w(2*n)*cos((1/period)*2*pi*n*in) + w(2*n + 1)*sin((1/period)*2*pi*n*in);
    end

    % plot
    plot(x_true,y_true,['.','k'],'LineWidth',1.5)
    hold on
    scatter(x,y,40,'k','fill')
    hold on
    plot(in,out,'r','LineWidth',2)
    
    % clean up plot 
    axis([0 1 (min([y;y_true;out]) - 0.1) (max([y;y_true;out]) + 0.1)])
    axis square
    box off
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y  ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    title(['k = ', num2str(k), ',   deg = ', num2str(deg)]);

end

function plot_tanh(x,y,deg,k)

    %%% Main: perform gradient descent to fit tanh basis sum %%%
    num_inits = 2;
    for foo = 1:num_inits
        [b,w,c,v] = tanh_grad_descent(x,y,deg); 

        % plot resulting fit
        hold on
        plot_approx(b,w,c,v,rand(3,1))
    end
    hold on
    plot(x_true,y_true,['.','k'],'LineWidth',1.5)
    hold on
    scatter(x,y,40,'k','fill')
    set(gcf,'color','w');

    % clean up plot 
    box off
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y  ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    title(['k = ', num2str(k), ',   deg = ', num2str(deg)]);
    
    function plot_approx(b,w,c,v,color)
    
        M = length(c);
        s = [0:0.01:1]';
        t = b;
        for m = 1:M
            t = t + w(m)*tanh(c(m) + v(m)*s);
        end
        hold on
        plot(s,t,'color',color,'LineWidth',2)
        axis([0 1 (min([y;y_true;t]) - 0.1) (max([y;y_true;t]) + 0.1)])
        axis square
    end
end
       
function F = build_poly(data, deg)
    F = [];
    for j = 0:deg
        F = [F  data.^j];
    end
end

function F = build_fourier(data, deg)
    F = [];
    for m = 1:deg
        F = [F, cos(2*pi*m*data), sin(2*pi*m*data)];
    end
    F = [ones(size(F,1),1) F];
end

function [b,w,c,v] = tanh_grad_descent(x,y,M)
    % initializations
    P = length(x);
    b = M*randn(1);
    w = M*randn(M,1);
    c = M*randn(M,1);
    v = M*randn(M,1);
    l_P = ones(P,1);
    
    % stoppers
    max_its = 10000;
    grad = 1;
    count = 1;
    
    %%% main %%%
    while norm(grad) > 10^-5 && count < max_its        
        % update gradients 
        q = obj([b;w;c;v]);
        grad_b = l_P'*q;
        grad_w = zeros(M,1);
        grad_c = zeros(M,1);
        grad_v = zeros(M,1);
        for n = 1:M
            t = tanh(c(n) + x*v(n));
            s = sech(c(n) + x*v(n)).^2;
            grad_w(n) = 2*l_P'*(q.*t);
            grad_c(n) = 2*l_P'*(q.*s)*w(n);
            grad_v(n) = 2*l_P'*(q.*x.*s)*w(n);
        end
        
        % determine steplength 
        grad = [grad_b; grad_w; grad_c; grad_v];
        alpha = 10^-4;

        % take gradient steps 
        b = b - alpha*grad_b;
        w = w - alpha*grad_w;
        c = c - alpha*grad_c;   
        v = v - alpha*grad_v;
        
        % update stoppers 
        count = count + 1;   
    end
   
    function s = obj(z)
        s = zeros(P,1);
        for p = 1:P
            s(p) = z(1) + z(2:M+1)'*tanh(z(M + 2:2*M+1) + x(p)*z(2*M + 2:end)) - y(p);
        end    
    end
    
end

end





