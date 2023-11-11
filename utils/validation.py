import numpy as np
from utils.metric import curl, vector_norm, divergence, normalized_divergence, weighted_theta, energy, normalized_divergence_2, weighted_theta_2

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def valid(b, b_mm = [-2500, 2500], j_mm = [1e1, 1e3], jxb_mm = [1e2, 1e6], e_mm = [0, 1e6]):
    b_z0 = b[:, :, 0, :]
    j = curl(b)
    j_map = vector_norm(j).sum(2)

    nu = vector_norm(np.cross(j, b, -1))
    de =  vector_norm(b)
    nu = (np.divide(nu, de, where=de!=0)).sum() 
    de = vector_norm(j).sum()
    sig_J = np.divide(nu, de, where=de!=0) * 1e2

    nu = vector_norm(np.cross(j, b, -1)) ** 2
    de = vector_norm(b) ** 2
    Lf = (np.divide(nu, de, where=de!=0)).mean()
    Ld = (divergence(b) ** 2).mean()

    total_div = normalized_divergence(b).mean()
    total_div_2 = normalized_divergence_2(b).mean()
    theta = weighted_theta(b, j)
    theta_2 = weighted_theta_2(b, j)

    jxb = np.cross(j, b, axis=-1)
    jxb_map = vector_norm(jxb).sum(2)

    me = energy(b)
    energy_map = me.sum(2)

    print('sig_J * 1e2: %.04f; Lf: %.04f; Ld: %.04f' % (sig_J, Lf, Ld))
    print('DIVERGENCE [1/pix]: %.04f; THETA [deg] %.04f' % (total_div, theta))
    print('DIVERGENCE [1/pix]: %.04f; THETA [deg] %.04f' % (total_div_2, theta_2))

    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plt.imshow(b_z0[..., 2].T, origin='lower', vmin=b_mm[0], vmax=b_mm[1], cmap='gray')
    plt.title('$B_z$')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(j_map.T, origin='lower', cmap='viridis', norm=LogNorm(vmin=j_mm[0], vmax=j_mm[1]))
    plt.title('$|J|$')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(jxb_map.T, origin='lower', cmap='inferno', norm=LogNorm(vmin=jxb_mm[0], vmax=jxb_mm[1]))
    plt.title('$|J x B|$')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(energy_map.T, origin='lower', vmin=e_mm[0], vmax=e_mm[1], cmap='jet')
    plt.title('Energy')
    plt.axis('off')
    plt.show()    