function poly_classification_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


xmin = 0;   % viewing area minimum 
xmax = 1;   % viewing area maximum

% parameters to play with
poly_degs = 1:8;    % range of poly models to compare

% load data 
[X,y] = load_data();

% perform feature transformation + classification
classify(X,y,poly_degs);


%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%
function classify(X,y,poly_degs)  

    errors = [];
    % solve for weights and collect errors
    for i = 1:length(poly_degs)
        % generate features
        poly_deg = poly_degs(i);         
        F = poly_features(X,poly_deg);
        
        % run logistic regression
        w = log_loss_newton(F',y);

        % output model
        figure(1)
        subplot(2,4,i) 
        hold on
        plot_poly(w,poly_deg,'b')
        str = ['D = ' num2str(i)];
        title (str,'Fontsize',12,'FontName','cmr10')
        hold on
        plot_pts(X,y)
        box on
        
        % calculate training errors
        resid = evaluate(F,y,w);    
        errors = [errors; resid];
    end
    
    % plot training errors for visualization
    figure(2)
    plot_errors(poly_degs, errors)
end

    
%%% builds (poly) features based on input data %%%
function F = poly_features(data,deg)

% ----->  YOUR CODE GOES HERE

end


%%% plots learned model %%%
function plot_poly(w,deg,color)
    % Generate poly seperator
    o = [xmin:0.01:xmax];
    [s,t] = meshgrid(o,o);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    f = zeros(length(s),1);
    count = 1;
    for n = 0:deg
        for m = 0:deg
            if n + m <= deg
                f = f + w(count)*s.^(n).*t.^(m);
                count = count + 1;
            end
        end
    end

    s = reshape(s,[length(o),length(o)]);
    t = reshape(t,[length(o),length(o)]);
    f = reshape(f,[length(o),length(o)]);  
    % plot contour in original space
    hold on
    contour(s,t,f,[0,0],'Color','k','LineWidth',2)
    axis([xmin xmax xmin xmax])
    axis square
    
    xlabel('x_1','Fontsize',18,'FontName','cmmi9')
    ylabel('x_2   ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box off
end

%%% plots points for each fold %%%
function plot_pts(X,y)
    
    % plot training set
    ind = find(y == 1);
    red =  [ 1 0 .4];

    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)
    hold on
    ind = find(y == -1);
    blue =  [ 0 .4 1];
    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
   
end

%%% plots training errors %%%
function plot_errors(poly_degs, errors)

    h2 = plot(1:max(poly_degs), errors,'--','Color',[0.1 0.8 1]);
    hold on
    plot(1:max(poly_degs),errors,'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
    legend([h2],{'training error'}); 

    % clean up plot
    set(gcf,'color','w');
    box on
    xlabel('D','Fontsize',18,'FontName','cmr10')
    ylabel('error','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    set(gcf,'color','w');
    box off
    axis square
end

%%% loads and plots labeled data %%%
function [X,y] = load_data()
      
    % load data
    data =  csvread('2eggs_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);
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





