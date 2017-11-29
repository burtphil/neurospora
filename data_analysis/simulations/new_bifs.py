import numpy as np
import matplotlib.pyplot as plt
import string

path = '/home/burt/neurospora/figures/bifurcations/frq_tot/bif_new/'

bif_k2 = np.load(path+"K2_bif.npz")
bif_k9 = np.load(path+"k9_bif.npz")
bif_k11 = np.load(path+"k11_bif.npz")
bif_k13 = np.load(path+"k13_bif.npz")
bif_k14 = np.load(path+"k14_bif.npz")

bif_data = [bif_k2,bif_k9,bif_k11,bif_k13,bif_k14]

t = []
frq_max = []
frq_min = []
frq_per = []

label = ["$K_2$","$k_9$","$k_{11}$","$k_{13}$","$k_{14}$"]

for dat in bif_data:
    t.append(dat["bif_array"])
    frq_max.append(dat["frq_tot_max_array"])
    frq_min.append(dat["frq_tot_min_array"])
    frq_per.append(dat["period_frq_tot_array"])
    
fig, axes = plt.subplots(5,2,figsize = (12,18))

axes = axes.flatten()

amps = [axes[0],axes[2],axes[4],axes[6],axes[8]]
per = [axes[1],axes[3],axes[5],axes[7],axes[9]]


#for idx,ax in enumerate(amps):
#    ax.plot(t[idx], frq_max[idx], 'k', t[idx], frq_min[idx], 'k')
for idx,ax in enumerate(amps):

    ax.plot(t[idx], frq_max[idx], 'k', t[idx], frq_min[idx], 'k')
    ax.set_xlabel(label[idx], fontsize = "xx-large")
    ax.set_ylabel("$FRQ_c$", fontsize = "xx-large")
    ax.tick_params(labelsize = 'x-large')
    ax.set_yticks([15,20,25,30,35])
    
for idx,ax in enumerate(per):
    ax.plot(t[idx], frq_per[idx], 'k')
    ax.set_xlabel(label[idx], fontsize = "xx-large")
    ax.set_ylabel("Period (h)", fontsize = "xx-large")
    ax.set_ylim(14,30)
    ax.set_yticks([18,22,26])
    ax.tick_params(labelsize = 'x-large')

for n, ax in enumerate(axes): 
    ax.text(-0.13, .97, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')

plt.tight_layout()
fig.savefig("new_bif.pdf", dpi = 1200)