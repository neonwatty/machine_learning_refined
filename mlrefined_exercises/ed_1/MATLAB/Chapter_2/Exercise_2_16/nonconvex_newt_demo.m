function nonconvex_newt_demo()
% nonconvex_newt_demo.m is a toy wrapper to illustrate the path
% taken by Hessian descent (or Newton's method).  The steps are evaluated 
% at the objective, and then plotted.  For the first 5 iterations the
% quadratic surrogate used to transition from point to point is also plotted.
% The plotted points on the objective turn from green to red as the 
% algorithm converges (or reaches a maximum iteration count, preset to 50).
%
% The (nonconvex) function here is
%
% f(x) = exp(x)*cos(2pi*sin(pi*x))
%
% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

%%% create function, choose initial point, to perform hessian descent on %%%
range = 1.1;    % symmetric range over which to plot the function
[a,b] = make_fun(range);
x0 = choose_starter(a,b,range);
disp(['You picked starting point x0 = ',num2str(x0)])

%%% perform hessian descent %%%
[x,in,out] = newton_descent(x0);

%%% plot function with hessian descent objective evaluations %%%
plot_it_all(in,out,range)

% performs hessian descent
function [x,in,out] = newton_descent(x)

    % initializations
    grad_stop = 10^-3;
    max_its = 50;
    iter = 1;
    in = [x];    
    out = [obj(x)];
    grad_eval = 1;
  
    % main loop
    while abs(grad_eval) > grad_stop && iter <= max_its
        % take gradient step
        grad_eval = grad(x);
        hess_eval = hess(x);
        if abs(hess_eval) < 10^-6
            hess_eval = sign(hess_eval)*10^-5;
        end
        x = x - grad_eval/hess_eval;
        
        % update containers
        in = [in x];
        out = [out obj(x)];
        
        % update stopers
        iter = iter + 1;
    end  
end

% evaluate the objective
function z = obj(y)
    z = exp(y)*cos(2*pi*sin(pi*y));
end 

% evaluate the gradient
function z = grad(y)
    z = exp(y)*cos(2*pi*sin(pi*y)) - 2*pi^2*exp(y)*sin(2*pi*sin(pi*y))*cos(pi*y);
end 

% evaluate the hessian
function z = hess(y)
    z = exp(y)*(2*(pi^3)*sin(pi*y)*sin(2*pi*sin(pi*y)) - (4*pi^4)*(cos(pi*y)^2)*cos(2*pi*sin(pi*y)) + cos(2*pi*sin(pi*y)) - (4*pi^2)*sin(2*pi*sin(pi*y))*cos(pi*y));
end 
       
% evaluate surrogate
function z = surrogate(y,x)
    z = obj(y) + grad(y)*(x - y) + 1/2*hess(y)*(x - y).^2;
end

% plot evaluation of descent steps 
function plot_steps(in,out)
    
    % colors for points
    s = (1/length(out):1/length(out):1)';
    colorspec = [s.^(1),flipud(s), zeros(length(out),1)];
    
    % plot initial point
    hold on
    plot(in(1),out(1),'o','Color',colorspec(1,:),'MarkerFaceColor',colorspec(1,:),'MarkerSize',6)
    u = text(in(1),out(1),num2str(0),'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',15);
    pause(1)

    % plot first surrogate
    s_range = 0.3;
    s = in(2) - s_range:0.001:in(2) + s_range;
    t = surrogate(in(1),s);
    h = plot(s,t,'--','Color','m');
    
    for i = 1:length(out) - 1
            if i < 5
                pause(1)
                
                % plot point 
                hold on
                plot(in(i+1),out(i+1),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',7)
               
                % plot iter number
                hold on
                delete(u)
                u = text(in(i+1),out(i+1),num2str(i),'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',15);
                
                % plot surrogate
                if i < length(out) - 2
                    pause(0.5)
                    delete(h)
                    s_range = 0.3;
                    s = in(i+2) - s_range:0.01:in(i+2) + s_range;
                    t = surrogate(in(i+1),s);
                    hold on
                    pause(0.5)
                    h = plot(s,t,'-.','Color','m');
                end
                
            end
        if i == 5
            delete(h);
        end
        if i > 5 && i < 15 % just plot point so things don't get too cluttered
            pause(0.3)
            
            % show connector on obj function
            hold on
            plot([in(i),in(i + 1)],[out(i),out(i + 1)],'--','Color','b')

            % plot point
            hold on
            plot(in(i+1),out(i+1),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',7)
        end
        if i >= 15 % just plot point so things don't get too cluttered
            pause(0.1)
            
            % show connector on obj function
            hold on
            plot([in(i),in(i + 1)],[out(i),out(i + 1)],'--','Color','b')
            
            % plot point
            plot(in(i+1),out(i+1),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',7)
        end 
        if i == length(out) - 1
            hold on
            delete(u)
            u = text(in(i+1),out(i+1),num2str(i),'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',15);
        end
    end
end

% makes desired function over -range*10 to range*10
function [a,b] = make_fun(range)
    a = [-range*10:0.001:range*10];
    b = exp(a).*cos(2*pi*sin(pi*a));
end

% plots everything
function plot_it_all(in,out,range)
    % plot function first
    plot(a,b,'k','LineWidth',1.5)

    % adjust window for best visualization,remove possible infinity values
    % obtained from too big of learning rate grad descent steps
    ind = isnan(in);
    in(ind) = [];
    out(ind) = [];
    ind = isinf(out);
    in(ind) = [];
    out(ind) = [];
    axis([min([0,in]) max([range,in]) (min([out,-3]) - 0.1) (max([out,3]) + 0.1)])
    box on
    xlabel('w','Fontsize',18,'FontName','cmmi9')
    ylabel('g','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    set(gca,'FontSize',12);

    % plot grad descent steps evaluated at the objective
    plot_steps(in,out)
end

% allows selection of initial point
function z = choose_starter(a,b,r)
    % make window for point picking
    plot(a,b,'k','LineWidth',1.5)
    axis([0 r -3.1 3.1])
    title('Pick starting point!')
    box on
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('f','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    set(gca,'FontSize',12);

    % pick a point
    [z,w]=ginput(1);
    close gcf
end

end