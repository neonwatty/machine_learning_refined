# add stumps together to get 1-d approximation
def add_stumps(x,y,max_depth):
    residual = y.copy()
    xmin = min(x)
    xmax = max(x)
            
    # loop over number of depths
    splits = []
    heights = []
    for d in range(max_depth):
        # loop over elements of input
        costs = []
        temp_splits = []
        temp_heights = []
        for s in range(len(x)-1):
            # determine points on each side of split
            split = (x[s]+x[s+1])/float(2)
            temp_splits.append(split)
            resid_left  = [t[0] for t in x if t <= split]
            resid_left = residual[:len(resid_left)]
            resid_right = residual[len(resid_left):]

            # compute average on each side
            ave_left = np.mean(resid_left)
            ave_right = np.mean(resid_right)
            temp_heights.append([ave_left,ave_right])

            # compute least squares error on each side
            cost_left = sum((ave_left - resid_left)**2)
            cost_right = sum((ave_right - resid_right)**2)
            total_cost = cost_left + cost_right
            costs.append(total_cost)
        
        # find best split
        ind = np.argmin(costs)
        split_pt = temp_splits[ind]
        split_heights = temp_heights[ind]
        splits.append(split_pt)
        heights.append(split_heights)
        # update residuals 
        y_hat = []
        for pt in range(len(x)):
            if x[pt] <= split_pt:
                y_hat.append(split_heights[0])
            else:
                y_hat.append(split_heights[1])
        residual -= y_hat

    return splits,heights,residual

# run stump adder
splits,heights,residual = add_stumps(x,y,max_depth = 100)

plt.scatter(x,y,linewidth = 3,color = 'k')
a = np.linspace(0,1,100)
model = np.zeros((len(a),1))
for s in range(len(splits)):
    current_split = splits[s]
    left = [t for t in a if t < current_split]
    model[0:len(left)] += heights[s][0]
    model[len(left):] += heights[s][1]
plt.plot(a,model,linewidth = 3) 


this still needs work - needs to be made recursive

do we actually need this though?  I guess..?  Isn't gradient  boosting 
just a tree grower?  What is the one where you just add stumps?


def binary_make_tree(x,y):
    max_depth = 2
    splits = []
    heights = []
    
    xmin = min(x)[0]
    xmax = max(x)[0]
    
    intervals = [[xmin,xmax]]
    
    for d in range(max_depth):
        for i in intervals:
            # grab limits
            xmin = i[0]
            xmax = i[1]

            # get all points in interval
            x_interval = [pt for pt in x (if pt <= xmax and pt >= xmin)]
            
            # loop over grid, find max
            best_split_cost = 10**5
            best_split_pt = 0
            for s in x_interval:
                # determine points on each side of split
                xleft = x_interval[0:s]
                xright = x[s+1:-1]

                # compute average on each side
                ave_l = np.mean(xleft)
                ave_r = np.mean(xright)

                # compute least squares error on each side
                cost = (xleft - ave_l)**2 + (xright - ave_r)**2

                # update split point
                if cost < best_split_cost:
                    best_split_pt = s

            # update split container
            splits.append(best_split_pt)