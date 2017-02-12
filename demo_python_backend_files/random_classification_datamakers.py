import numpy as np                                        # a fundamental numerical linear algebra library
import matplotlib.pyplot as plt                           # a basic plotting library
import pandas as pd

class Random_Classification_Datamaker:
    def __init__(self):
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
        self.num_seps = 0
        self.labels = 0
        self.data = 0

    # make a toy circle dataset
    def make_linear_classification_dataset(self,num_pts,num_seps):
        '''
        This function generates a random line dataset with N classes. 
        You can run this a couple times to get a distribution you like visually.  
        You can also adjust the num_pts parameter to change the total number of points in the dataset.
        '''

        # generate points
        data_x = 4*np.random.rand(num_pts) - 2
        data_y = 4*np.random.rand(num_pts) - 2
        data_x.shape = (len(data_x),1)
        data_y.shape = (len(data_y),1)
        data = np.concatenate((data_x,data_y),axis = 1)

        # make separators
        x_f = 4*np.linspace(0,1,100) - 2
        x_f.shape = (len(x_f),1)
        # loop over and assign labels
        labels = []
        seps = []
        for n in range(num_seps):
            m,b = np.random.randn(2,1)
            y_f = m*x_f + b

            # make labels and flip a few to show some misclassifications
            one_labels = np.sign(data_y - (m*data_x + b))
            one_labels = [int(v[0]) for v in one_labels]
            labels.append(one_labels)

            sep = np.concatenate((x_f,y_f),axis = 1)
            seps.append(sep)

        # determine true labels based on individual classifier labels for all points
        labels = np.asarray(labels)
        unique_vals = np.vstack({tuple(row) for row in labels.T})
        new_labels = np.zeros((len(data)))

        for i in range(len(unique_vals)):
            val = unique_vals[i]
            yo = np.argwhere((labels.T == val).all(axis=1))
            yo = [v[0] for v in yo]
            new_labels[yo] = int(i+1)
            
        # if two-class switch labels to -1/+1
        if len(unique_vals) == 2:
            ind = np.argwhere(new_labels > 1)
            ind = [v[0] for v in ind]
            new_labels[ind] = -1
            
        # return datapoints and labels for further 
        self.labels = new_labels
        self.seps = seps
        self.data = data
        
        # plot dataset
        self.plot_data()

    # make a toy nonlinear dataset
    def make_nonlinear_classification_dataset(self,num_pts,num_seps):
        '''
        This function generates a random line dataset with N classes. 
        You can run this a couple times to get a distribution you like visually.  
        You can also adjust the num_pts parameter to change the total number of points in the dataset.
        '''

        # generate points
        data_x = 4*np.random.rand(num_pts) - 2
        data_y = 4*np.random.rand(num_pts) - 2
        data_x.shape = (len(data_x),1)
        data_y.shape = (len(data_y),1)
        data = np.concatenate((data_x,data_y),axis = 1)

        # make separators
        x_f = 4*np.linspace(0,1,100) - 2
        x_f.shape = (len(x_f),1)

        # loop over and assign labels
        labels = []
        seps = []
        for n in range(num_seps):       

            deg = np.random.randint(2,7)
            b = np.random.randn(1)
            m = np.random.randn(1)
            y_f = m*x_f**deg + b 

            # make labels and flip a few to show some misclassifications
            temp = data_y - (m*data_x**deg + b)
            one_labels = np.sign(temp)
            one_labels = [int(v[0]) for v in one_labels]
            labels.append(one_labels)

            sep = np.concatenate((x_f,y_f),axis = 1)
            seps.append(sep)

        # determine true labels based on individual classifier labels for all points
        labels = np.asarray(labels)
        unique_vals = np.vstack({tuple(row) for row in labels.T})

        new_labels = np.zeros((len(data)))
        for i in range(len(unique_vals)):
            val = unique_vals[i]
            yo = np.argwhere((labels.T == val).all(axis=1))
            yo = [v[0] for v in yo]
            new_labels[yo] = int(i+1)

        # if two-class switch labels to -1/+1
        if len(unique_vals) == 2:
            ind = np.argwhere(new_labels > 1)
            ind = [v[0] for v in ind]
            new_labels[ind] = -1
            
        # return datapoints and labels for further 
        self.data = data
        self.labels = new_labels
        self.seps = seps
        
        # plot dataset
        self.plot_data()

    # make a toy circle dataset
    def make_circle_classification_dataset(self,num_pts):
        '''
        This function generates a random circle dataset with two classes. 
        You can run this a couple times to get a distribution you like visually.  
        You can also adjust the num_pts parameter to change the total number of points in the dataset.
        '''

        # generate points
        num_misclass = 5                 # total number of misclassified points
        s = np.random.rand(num_pts)
        data_x = np.cos(2*np.pi*s)
        data_y = np.sin(2*np.pi*s)
        radi = 2*np.random.rand(num_pts)
        data_x = data_x*radi
        data_y = data_y*radi
        data_x.shape = (len(data_x),1)
        data_y.shape = (len(data_y),1)

        # make separator
        s = np.linspace(0,1,100)
        x_f = np.cos(2*np.pi*s)
        y_f = np.sin(2*np.pi*s)

        # make labels and flip a few to show some misclassifications
        labels = radi.copy()
        ind1 = np.argwhere(labels > 1)
        ind1 = [v[0] for v in ind1]
        ind2 = np.argwhere(labels <= 1)
        ind2 = [v[0] for v in ind2]
        labels[ind1] = -1
        labels[ind2] = +1

        flip = np.random.permutation(num_pts)
        flip = flip[:num_misclass]
        for i in flip:
            labels[i] = (-1)*labels[i]

        # return datapoints and labels for further 
        data_x = np.asarray(data_x)
        data_x.shape = (len(data_x),1)
        data_y = np.asarray(data_y)
        data_y.shape = (len(data_y),1)
        self.data = np.concatenate((data_x,data_y),axis = 1)
        self.labels = labels
        x_f = np.asarray(x_f)
        x_f.shape = (len(x_f),1)
        y_f = np.asarray(y_f)
        y_f.shape = (len(y_f),1)
        seps = np.concatenate((x_f,y_f),axis = 1)
        self.seps = [seps]
        
        # plot dataset
        self.plot_data()
     
    # function - plot data with underlying target function generated in the previous Python cell
    def plot_data(self):    
        classes = np.unique(self.labels)
        if len(classes) > len(self.colors):
            print 'add colors to color list, currently not enough colors loaded to show all classes generated'
            return

        # plot data 
        fig = plt.figure(figsize = (4,4))
        count = 0
        for num in classes:
            inds = np.argwhere(self.labels == num)
            inds = [s[0] for s in inds]
            plt.scatter(self.data[inds,0],self.data[inds,1],color = self.colors[int(count)],linewidth = 1,marker = 'o',edgecolor = 'k',s = 50)
            count+=1

        # plot separators
        for i in range(len(self.seps)):
            plt.plot(self.seps[i][:,0],self.seps[i][:,1],'--k',linewidth = 3)

        # clean up plot
        plt.yticks([],[])
        plt.xlim([-2.1,2.1])
        plt.ylim([-2.1,2.1])
        plt.axis('off') 
        
    # save multiclass dataset
    def save_data(self,data_csvname,seps_csvname):
        # save data and labels
        data = np.asarray(self.data)
        data.shape = (len(self.data),2)
        labels = np.asarray(self.labels)
        labels.shape = (len(self.labels),1)
        new_data = np.concatenate((data,labels),axis = 1)
        new_data = pd.DataFrame(new_data)
        new_data.to_csv(data_csvname,header = None, index = False)

        # save separators
        new_seps = []
        for i in range(np.shape(self.seps)[0]):
            new_seps.append(self.seps[i][:,0])
            new_seps.append(self.seps[i][:,1])
        new_seps = np.asarray(new_seps).T
        new_seps = pd.DataFrame(new_seps)
        new_seps.to_csv(seps_csvname,header = None, index = False)