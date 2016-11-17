function poly_regression_hw()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% load data 
[x,y] = load_data();
deg = [1,3,5,7,15,20];           % degree polys to try

% plot data
plot_data(x,y,deg)

% generate nonlinear features
mses = [];
for D = 1:6
    % generate poly feature transformation
    F = poly_features(x,deg(D));
    
    % get weights
    w = pinv(F*F')*F*y;
    mses = [mses, norm(F'*w - y)/length(y)];
    
    % plot fit to data
    subplot(2,3,D)
    plot_model(w,deg(D));
end

% make plot of mse's
plot_mse(mses,deg)


%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%
    
%%%% takes poly features of the input %%%
function F = poly_features(x,D)
    
% ----> YOUR CODE GOES HERE

end

%%% plot the D-fit %%%
function plot_model(w,D)
    
    % plot determined surface in 3d space
    f = 0;
    s = [0:0.01:1]';
    f = [];
    for m = 1:D;
        f = [f, s.^m];
    end
    f = sum(f*w(2:end),2) + w(1);
    
    % plot contour in original space
    hold on
    plot(s,f,'Color','r','LineWidth',2)
    axis([0 1 (min(y) - 0.5) (max(y) + 0.5)])
end

%%% plot mse's over all D tested %%%
function plot_mse(mses,deg)
    figure(2)
    plot(1:length(mses),mses,'ro--')
    title('MSE on entire dataset in D','Fontsize',18)
    xlabel('D','Fontsize',18)
    set(gca,'XTick',[1:length(deg)],'Fontsize',13)
    s = {};
    for i = 1:length(deg)
        s{i} = num2str(deg(i));
    end
    set(gca,'XTickLabel',s)
    ylabel('MSE       ','Fontsize',18)
    set(get(gca,'YLabel'),'Rotation',0)
    axis square
    set(gcf,'color','w');
end

%%% plot data %%%
function plot_data(x,y,deg)
    for i = 1:6
        subplot(2,3,i)
        scatter(x,y,30,'k','fill');
        hold on

        % graph info labels
        s = ['D = ',num2str(deg(i))];
        title(s,'FontSize', 15)
        xlabel('x','Fontsize',18)
        ylabel('y   ','Fontsize',18)
        set(get(gca,'YLabel'),'Rotation',0)
        axis square
        set(gcf,'color','w');
    end

end

%%% load data %%%
function [x,y] = load_data()
    data = csvread('noisy_sin_samples.csv');
    x = data(:,1);
    y = data(:,2);
end

end


