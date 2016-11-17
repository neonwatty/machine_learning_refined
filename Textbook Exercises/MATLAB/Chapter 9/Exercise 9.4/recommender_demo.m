function recommender_demo()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
X = csvread('recommender_demo_data_true_matrix.csv');
X_corrupt = csvread('recommender_demo_data_dissolved_matrix.csv');

K = rank(X);

% run ALS for matrix completion
[C, W] = matrix_complete(X_corrupt, K); 

% plot results
plot_results(X, X_corrupt, C, W)


function [C, W] = matrix_complete(X, K)
    
% ---->  YOUR CODE GOES HERE   

end

function plot_results(X, X_corrupt, C, W)

    gaps_x = [1:size(X,2)];
    gaps_y = [1:size(X,1)];
    
    % plot original matrix
    subplot(1,3,1)
    imshow(X,[])
    colormap hot
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    title('original')
    set(gcf,'color','w');

    % plot corrupted matrix
    subplot(1,3,2)
    imshow(X_corrupt,[])
    colormap hot
    colorbar
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    set(gca,'CLim',[0, max(max(X))])
    title('corrupted')
    set(gcf,'color','w');

    % plot reconstructed matrix
    hold on
    subplot(1,3,3)
    imshow(C*W,[])
    colormap('hot');
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    RMSE_mat = sqrt(norm(C*W - X,'fro')/prod(size(X)));
    f = ['RMSE-ALS = ',num2str(RMSE_mat),'  rank = ', num2str(rank(C*W))];
    title(f)
    set(gcf,'color','w');

end


end

