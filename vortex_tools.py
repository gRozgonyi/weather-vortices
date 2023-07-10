import numpy as np
import matplotlib.pyplot as plt

def generate_vortex(X, Y, x0, y0, gamma, surface="plane"):
    """
    Generate a vortex velocity field with a given circulation and center.

    Parameters
    ----------
    X : 2D array of floats
        x coordinates of the mesh.
    Y : 2D array of floats
        y coordinates of the mesh.
    x0 : float
        x coordinate of the vortex center.
    y0 : float 
        y coordinate of the vortex center.
    gamma : float
        Circulation of the vortex.

    Returns
    -------
    w : 2D array of floats
        Complex potential of the vortex.
    """
    z0 = x0 + y0*1j
    z = X + Y*1j
    if surface == "plane":
        w = -1j*gamma/(2*np.pi)*(np.log(z-z0+1e-16))
    return w

def generate_vortices(x, y, x0, y0, gamma):
    """
    Generate multiple vortex velocity fields with a given circulation and center.
    
    Parameters
    ----------
    x : 2D array of floats
        x coordinates of the mesh.
    y : 2D array of floats
        y coordinates of the mesh.
    x0 : 1D array of floats
        x coordinates of the vortex centers.
    y0 : 1D array of floats
        y coordinates of the vortex centers.
    gamma : 1D array of floats
        Circulation of the vortices.

    Returns
    -------
    w : 2D array of floats
        Complex potential of the vortices.
    """
    for i in range(len(x0)):
        if i == 0:
            w = generate_vortex(x, y, x0[i], y0[i], gamma[i])
        else:
            w += generate_vortex(x, y, x0[i], y0[i], gamma[i])
    return w

def velocity_field(w):
    """
    Compute the velocity field on a given mesh due to a complex potential.
    
    Parameters
    ----------
    w : 2D array of floats
        Complex potential.
    X : 2D array of floats
        x coordinates of the mesh.
    Y : 2D array of floats
        y coordinates of the mesh.

    Returns
    -------
    u : 2D array of floats
        x component of the velocity
    v : 2D array of floats
        y component of the velocity
    """
    psi = w.imag
    u, v = np.gradient(psi)
    return u, -v

def remove_self_induced(w, X, Y, x0, y0, gamma):
    """
    Remove the self-induced contribution of a vortex to the complex potential.
    
    Parameters
    ----------
    w : 2D array of floats
        Complex potential.
    x0 : float
        x coordinate of the vortex center.
    y0 : float
        y coordinate of the vortex center.
    gamma : float
        Circulation of the vortex.
        
    Returns
    -------
    W : 2D array of floats
        Non-self-induced complex potential.
    """
    W = w - generate_vortex(X, Y, x0, y0, gamma)
    return W


################################################################################
############################## PLOTTING TOOLS ##################################
################################################################################

cmap = plt.cm.PRGn

def plot_streamlines(w, X, Y, title="Streamlines of the flow", width=11, height=9, levels=20):
    """
    Plot streamlines of a given complex potential.
    
    Parameters
    ----------
    w : 2D array of floats
        Complex potential.
    """
    cmap = plt.cm.PRGn
    plt.figure(figsize=(width, height))
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    maxlevel = np.max([np.abs(w.imag.min()), np.abs(w.imag.max())])
    plt.contour(X, Y, w.imag, levels=np.linspace(-maxlevel, maxlevel, levels+1), linewidths=2, colors='k')
    plt.contourf(X, Y, w.imag, levels=np.linspace(-maxlevel, maxlevel, levels+1), cmap=cmap.resampled(levels))
    plt.colorbar()
    plt.title(title, fontsize=16)

def plot_velocities(w, X, Y, title="Streamlines of the flow", width=11, height=9, levels=20):
    """
    Plot the velocity field due to a complex potential.
    
    Parameters
    ----------
    w : 2D array of floats
        Complex potential.
    X : 2D array of floats
        x coordinates of the mesh.
    Y : 2D array of floats
        y coordinates of the mesh.
    """
    u, v = velocity_field(w)
    width = 10
    height = (Y.max()-Y.min())/(X.max()-X.min())*width
    plt.figure(figsize=(width, height))
    plt.axis('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.streamplot(X, Y, u, v, density=1, linewidth=1, arrowsize=1, arrowstyle='->', color='gray')
    maxlevel = np.max([np.abs(w.imag.min()), np.abs(w.imag.max())])
    plt.pcolormesh(X,Y,w.imag, cmap=cmap, vmin=-maxlevel, vmax=maxlevel)
    plt.title(title, fontsize=16)