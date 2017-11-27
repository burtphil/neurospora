import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def ztan(t,T,kappa,z0, s=10):
    pi = np.pi
    om = 2*pi/T
    mu = pi/(om*np.sin(kappa*pi))
    cos1 = np.cos(om*t)
    cos2 = np.cos(kappa*pi)
    out = 1+2*z0*((1/pi)*np.arctan(s*mu*(cos1-cos2)))
    return out

t= np.linspace(0,48,1000)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex='col', sharey='row')

ax1.plot(t, ztan(t=t, T = 24, kappa = 0.5, z0= 0.05), label = "z0 = 0.05")
ax1.plot(t, ztan(t=t, T = 24, kappa = 0.5, z0= 0.2), label = "z0 = 0.2")
ax1.legend(loc = 1)

ax2.plot(t, ztan(t=t, T = 8, kappa = 0.5, z0= 0.2), label = "T = 8")
ax2.plot(t, ztan(t=t, T = 16, kappa = 0.5, z0= 0.2), label = "T = 16")
ax2.legend(loc = 1)

ax3.plot(t, ztan(t=t, T = 24, kappa = 0.5, z0= 0.2, s=0.1), label = "s = 0.1")
ax3.plot(t, ztan(t=t, T = 24, kappa = 0.5, z0= 0.2), label = "s = 10")
ax3.legend(loc = 1)

ax4.plot(t, ztan(t=t, T = 24, kappa = 0.25, z0= 0.2), label = r"$\kappa = 0.25$")
ax4.plot(t, ztan(t=t, T = 24, kappa = 0.5, z0= 0.2), label = r"$\kappa = 0.05$")
ax4.legend(loc = 1)

fig.text(0.5, 0.01, 'Time [a.u.]', ha='center', fontsize = 'xx-large')
fig.text(0.01, 0.5, 'Z(t) [a.u]', va='center', rotation='vertical', fontsize = 'xx-large')

#fig.savefig("zeitgeber_function.pdf",dpi=1200)

plt.show()
