function single_layer_classification_hw()


% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


minx = -1;
maxx = 1;
% load/make function to approximate
num_its = 1;
[X,y] = load_data(num_its);
M = 4;     % number of hidden units

%%% Main: perform gradient descent to fit tanh basis sum %%%
for j = 1:num_its
    subplot(1,num_its,j)
    [b,w,c,V] = tanh_softmax(X',y,M); 

    % plot resulting fit
    hold on
    plot_separator(b,w,c,V,X,y);
end

%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%

%%% gradient descent for single layer tanh nn %%%
function [b,w,c,V] = tanh_softmax(X,y,M)

 % initializations
    [N,P] = size(X);   
    b = randn(1);
    w = randn(M,1);
    c = randn(M,1);
    V = randn(N,M);
    l_P = ones(P,1);
        
    % stoppers
    max_its = 10000;
    grad = 1;
    count = 1;
    
    %%% main %%%
    while count <= max_its && norm(grad) > 10^-5   
       
        F = obj(c,V,X);
        
        % calculate gradients
% --->  grad_b = 
% --->  grad_w = 
% --->  grad_c = 
% --->  grad_V = 
        
        % determine steplength 
%       alpha = adaptive_step();
        alpha = 10^-2;
        
        % take gradient steps 
        b = b - alpha*grad_b;
        w = w - alpha*grad_w;
        c = c - alpha*grad_c;   
        V = V - alpha*grad_V;
        
        % update stoppers 
        count = count + 1;   
    end
    norm(grad)
    
    function p = adaptive_step()
        g_n = norm(grad)^2;
        step_l = 1;
        step_r = 0;
        u = 1;
        p = 1;
        
        while step_l > step_r && u < 30
            p = p*0.7;
            
            % left
            F = obj(c - p*grad_c,V - p*grad_V,X);
            o2 = sum(log(1 + exp(-y.*((b - p*grad_b) + F'*(w - p*grad_w)))));
            step_l = o2 - o;
            
            % right 
            step_r = -(p*g_n)/2;
            u = u + 1;
        end
    end
      
end

function y = sigmoid(z)
    y = 1./(1+exp(-z));
end

function F = obj(z,H,A)
    F = zeros(M,size(A,2));
    for p = 1:size(A,2)
        F(:,p) = tanh(z + H'*A(:,p));
    end    
    
end

% load data
function [A,b] = load_data(num_its)
    data = csvread('genreg_data.csv');
    A = data(:,1:end - 1);
    b = data(:,end);

   
    for j = 1:num_its
        subplot(1,num_its,j)
        % plot data 
        hold on
        ind = find(b == 1);
        red =  [ 1 0 .4];
        scatter(A(ind,1),A(ind,2),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none')
        hold on
        ind = find(b == -1);
        blue =  [ 0 .4 1];
        scatter(A(ind,1),A(ind,2),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none')
    end

end

% plot the seprator + surface
function plot_separator(b,w,c,V,X,y)
    
    % plot determined surface in 3d space
    s = [minx:0.01:maxx];
    [s1,s2] = meshgrid(s,s);
    s1 = reshape(s1,numel(s1),1);
    s2 = reshape(s2,numel(s2),1);
    
    g = zeros(length(s1),1);
    for i = 1:length(s1)
        t = [s1(i);s2(i)];
        F = obj(c,V,t);
        g(i) = tanh(b + F'*w);
    end
    s1 = reshape(s1,[length(s),length(s)]);
    s2 = reshape(s2,[length(s),length(s)]);
    g = reshape(g,[length(s),length(s)]);  % divide by # for visualization purposes only!
    alpha(0.4)
    
    % plot contour in original space
    hold on
    contour(s1,s2,g,[0,0],'Color','k','LineWidth',2)
    axis([0 1 0 1])
    
    % graph info labels
    xlabel('x_1','Fontsize',16)
    ylabel('x_2      ','Fontsize',16)
    set(get(gca,'YLabel'),'Rotation',0)
    axis square
    set(gcf,'color','w');
end

end
