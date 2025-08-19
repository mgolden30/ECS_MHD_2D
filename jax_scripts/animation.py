"""
Generate a GIF visualizing a periodic orbit of a flow.
"""

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO

filename = "temp_data/newton/5.npz"

input_dict, param_dict = dictionaryIO.load_dicts(filename)

steps = param_dict['steps'] #number of timesteps

save_every = 32

assert( steps % save_every == 0)
print(f"Evolving this state with {steps} timesteps and saving every {save_every}. This gives {steps // save_every} frames...")


# Load in the initial data
f = input_dict['fields']
T = input_dict['T']
sx= input_dict['sx']

#frames per second
fps = 8

def main(f):
    # collect frames in memory
    frames = []
    nt = steps // save_every

    update = jax.jit( lambda f: mhd_jax.eark4(f, T/steps, save_every, param_dict) )


    for t in range(nt):
        print(t)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns
        plt.subplots_adjust(wspace=0.3)  # 0.3 gives some horizontal space
        for ax, field in zip(axs, [f[0,:,:], f[1,:,:]]):  # show two components
            ax.set_axis_off()
            im = ax.imshow(field.transpose(), cmap='bwr', origin="lower",
                           extent=[0, 2*jnp.pi, 0, 2*jnp.pi],
                           interpolation="none",
                           vmin=-10.0,vmax=10.0)  # pixel-wise
            fig.colorbar(mappable=im, ax=ax, fraction=0.046, pad=0.04)

        axs[0].set_title(r"$\nabla \times {\bf u}$", fontsize=12)
        axs[1].set_title(r"$\nabla \times {\bf B}$", fontsize=12)

        # save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=400, bbox_inches="tight", pad_inches=0)
        #plt.savefig(f"figures/{t}.png", format="png", dpi=400, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

        #Update the state after writing
        f = jnp.fft.rfft2(f)
        f = update(f)
        f = jnp.exp( -1j*sx/nt*param_dict['kx'] )*f #Spatially shift to comoving frame
        f = jnp.fft.irfft2(f)

    # write GIF
    #imageio.mimsave("figures/RPO.gif", frames, palettesize=256, subrectagles=True, duration=(1.0/fps), loop=0)  # duration in seconds per frame
    imageio.mimsave("figures/RPO.mp4", frames, fps=fps ) 

if __name__ == "__main__":
    main(f)

