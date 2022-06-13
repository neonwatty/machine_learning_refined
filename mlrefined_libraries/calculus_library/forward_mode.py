import numpy as np

class MyTuple:
    '''
    The basic object representing the input variable 'w'
    represents the core of our AD calculator.  An instance 
    of this class is a tuple containining one function/derivative
    evaluation of the variable 'w'.  Because it is meant to 
    represent the simple variable 'w' the derivative 'der' is
    preset to 1.  The value 'val' can be set to 0 by default.  
    '''
    def __init__(self,**kwargs):
        # variables for the value (val) and derivative (der) of our input function 
        self.val = 0
        self.der = 1    
        
        # re-assign these default values 
        if 'val' in kwargs:
            self.val = kwargs['val']
        if 'der' in kwargs:
            self.der = kwargs['der']
            
##### basic arithmetic functions #####         
### our implementation of the addition rules from Table 2 ###
def add(a,b):
    # Create output evaluation and derivative object
    c = MyTuple()
    
    # switch to determine if a or b is a constant
    if type(a) != MyTuple:
        c.val = a + b.val
        c.der = b.der
    elif type(b) != MyTuple:
        c.val = a.val + b
        c.der = a.der
    else: # both inputs are MyTuple objects, i.e., functions
        c.val = a.val + b.val
        c.der = a.der + b.der
    
    # Return updated object
    return c

# this next line overloads the addition operator for our MyTuple objects, or in other words adds the 'add' function to our MyTuple class definition on the fly
MyTuple.__add__ = add

# overload the reverse direction so that a + b = b + a
MyTuple.__radd__ = add

### our implementation of the addition rules from Table 2 ###
def multiply(a,b):
    # Create output evaluation and derivative object
    c = MyTuple()

    # switch to determine if a or b is a constant
    if type(a) != MyTuple:
        c.val = a*b.val
        c.der = a*b.der
    elif type(b) != MyTuple:
        c.val = a.val*b
        c.der = a.der*b

    else: # both inputs are MyTuple objects i.e., functions
        c.val = a.val*b.val
        c.der = a.der*b.val + a.val*b.der     # product rule
    
    # Return updated object
    return c

# create two MyTuple objects and try to use Python's built in function assigned to the * operator on them
MyTuple.__mul__ = multiply

# overload the 'reverse multiplication' so that a*b = b*a
MyTuple.__rmul__ = multiply    

### our implementation of the power rule from Table 1 ###
def power(a,n):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = a.val**n

    # Produce new derivative value - we need to use the chain rule here!
    b.der = n*(a.val**(n-1))*a.der
    
    # Return updated object
    return b

# create two MyTuple objects and try to use Python's built in function assigned to the ** operator on them
MyTuple.__pow__ = power

##### elementary functions #####         
# our implementation of the sinusoid rule from Table 1
def log(a):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = np.log(a.val)

    # Produce new derivative value
    b.der = (1/a.val)*a.der
    
    # Return updated object
    return b

# our implementation of the power rule from Table 1 
def tanh(a):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = np.tanh(a.val)

    # Produce new derivative value
    b.der = (1 - np.tanh(a.val)**2)*a.der
    
    # Return updated object
    return b

# our implementation of the cosine rule from Table 1
def cos(a):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = np.cos(a.val)

    # Produce new derivative value - we need to use the chain rule here!
    b.der = -np.sin(a.val)*a.der
    
    # Return updated object
    return b

# our implementation of the sinusoid rule from Table 1
def sin(a):
    # Create output evaluation and derivative object
    b = MyTuple()
    
    # Produce new function value
    b.val = np.sin(a.val)

    # Produce new derivative value - we need to use the chain rule here!
    b.der = np.cos(a.val)*a.der
    
    # Return updated object
    return b