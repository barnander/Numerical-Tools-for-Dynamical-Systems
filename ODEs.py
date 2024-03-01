#Systems of ODEs represtenting normal forms of bifurcations
#%% Packages
import numpy as np

#%%

def Saddle_Node(x,t,p):
    """
    Parameters:
        -x (np array): state variable
        -t (float): time
        -p (float): parameter

    Returns:
        -dx (np array): derivative  
    """
    dx = p + x**2
    return dx

def Transcritical(x,t,p):
    """
    Parameters:
        -x (np array): state variable
        -t (float): time
        -p (float): parameter

    Returns:
        -dx (np array): derivative  
    """
    dx = p*x - x**2
    return dx

def Pitchfork_Super(x,t,p):
    """
    Parameters:
        -x (np array): state variable
        -t (float): time
        -p (float): parameter

    Returns:
        -dx (np array): derivative  
    """
    dx = p*x - x**3
    return dx

def Hopf_Super(x,t,p):
    """
    Parameters:
        -x (np array): state variables
        -t (float): time
        -p (float): parameter

    Returns:
        -deriv (np array): array of derivatives of state variables
    """
    dx = p*x[0] + x[1]
    dy = -x[0] + p*x[1] - x[0]**2 * x[1]
    deriv = np.append(dx,dy)
    return deriv