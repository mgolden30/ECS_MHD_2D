"""
Generate a MP4 visualizing a periodic orbit of a flow.
"""

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

import lib.mhd_jax as mhd_jax
import lib.timestepping as timestepping
import lib.dictionaryIO as dictionaryIO


from matplotlib.colors import LinearSegmentedColormap
import numpy as np




####################
# PARAMETERS
####################

#State to animate
filename = "temp_data/newton/4.npz"
#filename = "solutions/Re50/1.npz"
#filename = "candidates/Re100/1.npz"
#filename = "temp_data/newton/3.npz"


save_every = 32 #How often do we save a frame?
fps = 8 #frames per second
vmin = -10.0 #colorbar limits
vmax = 10.0 #colorbar limits
comoving = False
double_domain = False
add_last_frame = True #Set to false for perfect periodic video

bg_color = "black"
font_color = "white"






input_dict, param_dict = dictionaryIO.load_dicts(filename)

print((input_dict["sx"], param_dict["shift_reflect_ny"], param_dict["rot"]))


steps = param_dict['steps'] #number of timesteps


assert( steps % save_every == 0)
print(f"Evolving this state with {steps} timesteps and saving every {save_every}. This gives {steps // save_every} frames...")

# Load in the initial data
f = input_dict['fields']
T = input_dict['T']
sx= input_dict['sx']



print(f"T = {input_dict['T']} and sx = {input_dict['sx']}")





# Your RGB points (values between 0 and 1)
bottom     = [0, 0, 0.5]
botmiddle  = [0, 0.5, 1]
middle     = [0, 0, 0]
topmiddle  = [1, 0, 0]
top        = [0.5, 0, 0]

colors = [bottom, botmiddle, middle, topmiddle, top]

# Create linear segmented colormap
my_cmap = LinearSegmentedColormap.from_list("custom_bkb", colors, N=256)





def main(f):
    # collect frames in memory
    frames = []
    nt = steps // save_every

    v_fn = lambda f: mhd_jax.state_vel(f,param_dict,include_dissipation=False)
    L_diag = mhd_jax.dissipation(param_dict)
    update = jax.jit( lambda f: timestepping.lawson_rk4(f, T/steps*save_every, save_every, v_fn, L_diag, param_dict['mask']) )

    num_frames = nt
    if add_last_frame:
        num_frames = nt + 1

    for t in range(num_frames):
        print(t)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), facecolor=bg_color)  # 1 row, 2 columns
        plt.subplots_adjust(wspace=0.3)  # 0.3 gives some horizontal space
        for ax, field in zip(axs, [f[0,:,:], f[1,:,:]]):  # show two components
            ax.set_axis_off()
            
            if double_domain:
                field = jnp.tile(field, (2,2))

            im = ax.imshow(field.transpose(), cmap=my_cmap, origin="lower",
                           extent=[0, 2*jnp.pi, 0, 2*jnp.pi],
                           interpolation="none",
                           vmin=vmin, vmax=vmax)  # pixel-wise
            cbar = fig.colorbar(mappable=im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([vmin, 0, vmax])
            cbar.ax.tick_params(colors=font_color)             # tick color
            plt.setp(cbar.ax.get_yticklabels(), color=font_color)  # tick labels
            ax.tick_params(colors=font_color)  # tick labels
            for spine in ax.spines.values():
                spine.set_edgecolor(font_color)  # axis border
            

        axs[0].set_title(r"$\nabla \times {\bf u}$", fontsize=12, color=font_color)
        axs[1].set_title(r"$\nabla \times {\bf B}$", fontsize=12, color=font_color)

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
        if comoving:
            f = jnp.exp( -1j*sx/nt*param_dict['kx'] )*f #Spatially shift to comoving frame
        f = jnp.fft.irfft2(f)

    # write GIF
    #imageio.mimsave("figures/RPO.gif", frames, palettesize=256, subrectagles=True, duration=(1.0/fps), loop=0)  # duration in seconds per frame
    imageio.mimsave("figures/RPO.mp4", frames, fps=fps ) 

if __name__ == "__main__":
    main(f)

