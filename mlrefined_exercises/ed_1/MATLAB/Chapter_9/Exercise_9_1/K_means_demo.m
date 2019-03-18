function K_means_demo()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data
X = csvread('kmeans_demo_data.csv');

C0 = [0,0;-.5,.5];     % initial centroid locations

% run K-means
K = size(C0,2);
[C, W] = your_K_means(X, K, C0);

% plot results
plot_results(X, C, W, C0);


function [C, W] = your_K_means(X, K, C0)

% ----->  YOUR CODE GOES HERE    
   
end


function plot_results(X, C, W, C0)
    
    K = size(C,2);
    
    % plot original data 
    subplot(1,2,1)
    scatter(X(1,:),X(2,:),'fill','k');
    title('original data')
    axis([0 1 0 1]);
    set(gcf,'color','w');
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    axis square
    axis([-.5 .5 -.5 .5]);
    box on
    hold on
    colors = [0 0 1;
              1 0 0;
              0 1 0;
              1 0 1
              1 1 0
              0 1 1];
          
    for k = 1:K  
        scatter(C0(1,k),C0(2,k),100,'x','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:))
        hold on
    end      
          
                 
    % plot clustered data 
    subplot(1,2,2)
    for k = 1:K
        ind = find(W(k,:) == 1);
        scatter(X(1,ind),X(2,ind),'fill','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:));
        hold on
    end
    
    for k = 1:K  
        scatter(C(1,k),C(2,k),100,'x','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:))
        hold on
    end  
    title('clustered data')
    axis([-.5 .5 -.5 .5]);
    set(gcf,'color','w');
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    axis square
    box on
end

end



