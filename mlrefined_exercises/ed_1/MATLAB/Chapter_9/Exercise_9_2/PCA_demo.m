function PCA_demo()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
X = csvread('PCA_demo_data.csv');
n = size(X,1);
means = repmat(mean(X),n,1);
X = X - means;  % center the data
X = X';

K = 1;  

% run PCA 
[C, W] = your_PCA(X, K);

% plot_results
plot_results(X, C)


function [C, W] = your_PCA(X, K)

% ---->  YOUR CODE GOES HERE    
    
end


function plot_results(X, C)
        
    % Print points and pcs
    subplot(1,2,1)
    for j = 1:n
        hold on
        scatter(X(1,:),X(2,:),'fill','b')
    end
    hold on
    s = C(1,1):0.001:-C(1,1);
    m = C(2,1)/C(1,1);
    plot(s,m*s,'k','LineWidth',2.5)
    xlabel('b_1','Fontsize',14,'FontName','cmr10')
    ylabel('b_2','Fontsize',14,'FontName','cmr10')

    axis([-0.5 0.5 -0.5 0.5])    % Set viewing axes
    axis square


    % Plot projected data
    subplot(1,2,2)

    X_proj = C*((C'*C)\(C'*X));
    for j = 1:n
        hold on
        scatter(X_proj(1,:),X_proj(2,:),'fill','b')
    end
    axis([-0.5 0.5 -0.5 0.5])    % Set viewing axes
    xlabel('b_1','Fontsize',14,'FontName','cmr10')
    ylabel('b_2','Fontsize',14,'FontName','cmr10')

    axis square
    set(gcf,'color','w');

end

end


    
    








