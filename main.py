import numpy as np
import matplotlib.pyplot as plt
import time

from Direct_p import psi_direct
from DCT import psi_dct

Ns = [16,32,64,94,128]
err_dir=[]
err_dct=[]
time_dir=[]
time_dct=[]

for N in Ns:
    print(f"At case Nx=Ny={N}")
    dx = 2.0/N
    dy = 2.0/N
    xx = np.linspace(-1+dx/2,1-dx/2,N)
    yy = np.linspace(-1+dy/2,1-dy/2,N)
    X,Y = np.meshgrid(xx,yy,indexing="ij")
    u_ex = np.cos(np.pi*X)*np.cos(np.pi*Y)

    t0 = time.perf_counter()
    _,_,u_dir = psi_direct(N,N)
    t1 = time.perf_counter()
    time_dir.append(t1-t0)

    t0 = time.perf_counter()
    _,_,u_dct = psi_dct(N,N)
    t1 = time.perf_counter()
    time_dct.append(t1-t0)

    u_dir -= np.mean(u_dir)
    u_dct -= np.mean(u_dct)
    u_ex -= np.mean(u_ex)

    err_dir.append(np.linalg.norm(u_dir-u_ex)/np.linalg.norm(u_ex))
    err_dct.append(np.linalg.norm(u_dct-u_ex)/np.linalg.norm(u_ex))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(Ns, err_dir, "-o", label="Direct solver error")
plt.loglog(Ns, err_dct, "-s", label="DCT solver error")
plt.xlabel("Grid size N (Nx=Ny)")
plt.ylabel("Relative error vs exact")
plt.title("Error vs exact solution")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Ns, time_dir, "-o", label="Direct sparse solve")
plt.plot(Ns, time_dct, "-s", label="DCT solver")
plt.xlabel("Grid size N (Nx=Ny)")
plt.ylabel("Runtime [s]")
plt.title("Performance comparison")
plt.legend()

plt.tight_layout()
plt.show()
