import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def psi_direct(Nx,Ny):

    dx = 2.0/Nx
    dy = 2.0/Ny

    xx = np.linspace(-1 + dx/2,1-dx/2,Nx)
    yy = np.linspace(-1 + dy/2,1-dy/2,Ny)
    x, y= np.meshgrid(xx,yy,indexing="ij")

    tmp = -2*np.ones(Nx)
    tmp[0] = tmp[-1] = -1
    Ad = sp.diags(tmp, offsets=0, shape=(Nx,Nx))
    Au = sp.diags([np.ones(Nx-1)],[1], shape=(Nx,Nx))
    Al = sp.diags([np.ones(Nx-1)],[-1],shape=(Nx,Nx))
    Ax = Ad+Au+Al

    I = sp.eye(Nx)
    blocks = []

    for j in range(Ny):
        if j==0 or j==Ny-1:
            block = Ax/dx**2 - I/dy**2
        else:
            block = Ax/dx**2 - 2*I/dy**2
        blocks.append(block)

    Abig = sp.block_diag(blocks, format="csr")

    off = sp.eye(Nx*(Ny-1),format="csr")
    Abig[:-Nx,Nx:] += off/dy**2
    Abig[Nx:,:-Nx] += off/dy**2

    f = -2*np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y)
    f = f.ravel()

    Abig = Abig.tolil()
    Abig[0,:] = 0
    Abig[0,0] = 1.0
    f[0] = 0.0
    Abig = Abig.tocsr()

    u = spla.spsolve(Abig, f)
    u = u.reshape((Nx,Ny))

    return x,y,u


#Nx,Ny = 100,100
#x,y,u = psi(Nx,Ny)

#plt.figure()
#plt.pcolormesh(x,y,u,shading="auto", cmap="coolwarm")
#plt.colorbar(label="u(x,y)")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.show()
