import matplotlib.pyplot as plt
import numpy as np
from typing import Union

class Variable(np.ndarray):
    def __new__(cls, size: int = 0):
        if size > 0:
            obj = np.zeros(size).view(cls)
        else:
            obj = np.array([]).view(cls)
        obj.i = 0
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        # Always set i to current length for sliced/modified arrays
        self.i = len(self)
    
    def append(self, value):
        """Append single value"""
        if self.i < len(self):
            self[self.i] = value
            self.i += 1
        else:
            new_arr = np.append(self, value).view(Variable)
            new_arr.i = len(new_arr)
            return new_arr
        return self
    
    def extend(self, values):
        """Extend with multiple values"""
        values = np.array(values)
        available_space = len(self) - self.i
        
        if available_space >= len(values):
            # Fit in existing space
            self[self.i:self.i + len(values)] = values
            self.i += len(values)
            return self
        else:
            # Need to concatenate
            if available_space > 0:
                self[self.i:] = values[:available_space]
                remaining = values[available_space:]
            else:
                remaining = values
            
            new_arr = np.concatenate([self, remaining]).view(Variable)
            new_arr.i = len(new_arr)
            return new_arr
    
    def plot(self, /, title: str, xlabel: str = "Index", ylabel: str = "Value", plot_type: str = "auto", **kwargs):
        """
        Okay so what I wanna do here is that - I wanna make making plots fast and mistake proof. 
        It will not simply return the plot, it's gonna be something more useful. 
        I think I can use another class that's gotta extend plt.plot or variable, transformation function
        What I want to return is that class and then I wanna immediately use some new mechanics of that class 
        And What I wanna get is our_new_class.show() - plots and shows the figure
        out_new_class.savefig obviously plots and saves the figure (doesn't show)
        
        """
    
    def __repr__(self):
        return f"Variable({np.array(self).__repr__()}, i={self.i})"
