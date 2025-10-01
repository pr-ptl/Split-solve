import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from scipy.fft import dstn, idstn

def psi_dct(Nx, Ny):

    dx = 2.0/Nx
    dy = 2.0/Ny

    xx = np.linspace(-1+dx/2, 1-dx/2, Nx)
    yy = np.linspace(-1+dy/2, 1-dy/2, Ny)
    x,y = np.meshgrid(xx, yy, indexing="ij")

    kx = np.arange(Nx)
    ky = np.arange(Ny)
    mwx = 2*(np.cos(np.pi*kx/Nx)-1)/dx**2
    mwy = 2*(np.cos(np.pi*ky/Ny)-1)/dy**2
    MWX,MWY = np.meshgrid(mwx, mwy, indexing="ij")

    f = -2 * np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y)
    fhat = dctn(f, type=2, norm="ortho")
    denom = MWX+MWY
    uhat = np.zeros_like(fhat)
    mask = denom !=0
    uhat[mask] = fhat[mask]/denom[mask]
    uhat[0,0] = 0.0

    u = idctn(uhat, type=2, norm="ortho")

    return x, y, u

def psi_dst(Nx, Ny):

    dx = 2.0/Nx
    dy = 2.0/Ny
    #x1d = -1.0 + np.arange(1,Nx+1) * dx
    #y1d = -1.0 + np.arange(1,Ny+1) * dy
    xx = np.linspace(-1+dx/2, 1-dx/2, Nx)
    yy = np.linspace(-1+dy/2, 1-dy/2, Ny)

    x,y = np.meshgrid(xx, yy, indexing="ij")

    f= -2 * np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y)
    fhat = dstn(f, type=2, norm="ortho")

    kx = np.arange(1, Nx+1)
    ky = np.arange(1, Ny+1)
    lamx = 2.0 * (1.0 - np.cos(np.pi*kx/(Nx+1)))/dx**2
    lamy = 2.0 * (1.0 - np.cos(np.pi*ky/(Ny+1)))/dy**2

    denom = lamx[:, None] + lamy[None, :]
    uhat = fhat/denom
    u = idstn(uhat, type=2, norm="ortho")

    return x, y, u
    

Nx,Ny = 50,50
x,y,u = psi_dst(Nx,Ny)

u_exact = np.cos(np.pi * x) * np.cos(np.pi * y)

# relative error
rel_err = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
print("relative error:", rel_err)

plt.pcolormesh(x, y, u, shading='auto')
plt.colorbar(); plt.title("u (DST solver)")
plt.show()

plt.pcolormesh(x,y,u,shading="auto",cmap="coolwarm")
plt.colorbar(label="u(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
