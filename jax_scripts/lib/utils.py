'''
You are entering the junkyard. This is for useful functions that need to live somewhere.
'''

import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.loss_functions as loss_functions
import lib.symmetry as symm
import lib.mhd_jax as mhd_jax
import lib.adam as adam
from lib.linalg import gmres




def initialize_system(n, nu, eta, b0, u0, seed, forcing_str, precision=jnp.float64):
    '''
    PURPOSE:
    Build everything needed to start running MHD from a small set of physical
    inputs: grid resolution, dissipation parameters, mean fields, a forcing
    function (as a string), and a random seed. This hides grid construction,
    dealiasing, and random initial-data generation from the caller.

    INPUT:
    n          - grid resolution (n x n)
    nu         - fluid dissipation
    eta        - magnetic dissipation
    b0         - mean magnetic field, e.g. [0.0, 0.1]
    u0         - mean flow field, e.g. [0.0, 0.0]
    seed       - integer seed for the random initial condition
    forcing_str - a string defining a lambda(x,y) -> vorticity forcing,
                  e.g. "lambda x,y : -4*jnp.cos(4*y)"
    precision  - jnp.float32 or jnp.float64 (default double precision)

    OUTPUT:
    f          - random, dealiased initial condition in real space, shape [2,n,n]
    param_dict - grid info plus nu, eta, b0, u0, forcing, forcing_str
    '''

    if precision == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    param_dict = mhd_jax.construct_domain(n, precision)

    x = param_dict['x']
    y = param_dict['y']

    forcing_fn = eval(forcing_str)
    forcing = forcing_fn(x, y)

    param_dict.update({
        'nu': nu, 'eta': eta, 'b0': b0, 'u0': u0,
        'forcing': forcing, 'forcing_str': forcing_str,
    })

    key = jax.random.PRNGKey(seed=seed)
    f = 10 * jax.random.normal(key, shape=[2, n, n])

    # Remove unphysical modes introduced by random initialization
    f = param_dict['mask'] * jnp.fft.rfft2(f)
    f = jnp.fft.irfft2(f)

    return f, param_dict


def evolve_transient(f, param_dict, dt, transient_time):
    '''
    PURPOSE:
    Run forward time integration to discard the initial transient, so that
    we are left with a state on (or near) the turbulent attractor.

    INPUT:
    f             - initial condition in real space, shape [2,n,n]
    param_dict    - output of initialize_system
    dt            - timestep
    transient_time - total time to integrate for

    OUTPUT:
    f - state in spectral space after the transient, shape [2,n,n//2+1]
    '''
    import lib.timestepping as timestepping

    transient_steps = int(round(transient_time / dt))

    f = jnp.fft.rfft2(f)
    v_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=True)
    f = timestepping.rk4(f, dt * transient_steps, transient_steps, v_fn)

    return f


def generate_turbulent_trajectory(f, param_dict, dt, steps, ministeps):
    '''
    PURPOSE:
    Integrate forward from f, saving "steps" snapshots spaced "ministeps"
    timesteps apart, and compute a pairwise-distance recurrence diagram
    between the saved snapshots. Bright/dark streaks away from the diagonal
    indicate near-recurrences, which make good initial guesses for RPOs.

    INPUT:
    f          - initial condition in spectral space, shape [2,n,n//2+1]
    param_dict - output of initialize_system
    dt         - timestep
    steps      - number of snapshots to save
    ministeps  - number of timesteps taken between each saved snapshot

    OUTPUT:
    fs   - trajectory snapshots in spectral space, shape [steps,2,n,n//2+1]
    dist - normalized recurrence diagram, shape [steps,steps]
    '''
    import time
    import lib.timestepping as timestepping

    v_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=True)
    one_step = jax.jit(lambda f: timestepping.rk4(f, dt * ministeps, ministeps, v_fn))

    start = time.time()

    def wrapper(f, _):
        f_new = one_step(f)
        return f_new, f_new

    _, fs = jax.lax.scan(wrapper, f, None, length=steps)
    stop = time.time()
    print(f"Generating {steps} steps of turbulence took {stop-start} seconds.")

    start = time.time()
    dist = jnp.zeros([steps, steps])
    for i in range(steps):
        diff = jnp.abs(fs) - jnp.abs(fs[i, :, :, :])
        diff = jnp.reshape(diff, [steps, -1])
        dist = dist.at[:, i].set(jnp.linalg.vector_norm(diff, axis=1))
    stop = time.time()
    print(f"Recurrence diagram computed in {stop - start} seconds")

    dist = dist / jnp.mean(jnp.abs(dist))

    return fs, dist


def _compute_observables(f, param_dict):
    '''
    Energy injection and dissipation for a spectral-space state f, shape [2,n,n//2+1].
    Used as a cheap 2D projection of the (very high dimensional) state space.
    '''
    u = jnp.fft.irfft2(1j * param_dict['to_u'] * f[0, :, :])
    v = jnp.fft.irfft2(1j * param_dict['to_v'] * f[0, :, :])

    force = jnp.fft.rfft2(param_dict['forcing'])
    fx = jnp.fft.irfft2(1j * param_dict['to_u'] * force)
    fy = jnp.fft.irfft2(1j * param_dict['to_v'] * force)

    injection = jnp.mean(u * fx + v * fy)

    dissipation = jnp.mean(jnp.square(jnp.fft.irfft2(f)), axis=[-1, -2])
    dissipation = param_dict['nu'] * dissipation[0] + param_dict['eta'] * dissipation[1]
    return jnp.array([injection, dissipation])


def animate_rpo(input_dict, param_dict, filename="figures/rpo.gif", steps=128, fps=8):
    '''
    PURPOSE:
    Animate a converged RPO over one period T, alongside a 2D projection of
    the trajectory onto (energy injection, dissipation) -- a cheap way to see
    the periodic orbit as a closed loop in a low-dimensional observable space.

    Hides: spectral-space time integration with TDRK4 over the period,
    collecting snapshots with jax.lax.scan, computing/normalizing the
    projection observables, and rendering with blitting (see make_gif).

    INPUT:
    input_dict - converged state: {'fields', 'T', 'sx'}
    param_dict - grid + physical parameters
    filename   - output path for the gif
    steps      - number of snapshots/frames to render over one period
    fps        - frames per second

    OUTPUT:
    filename - the path the gif was written to
    '''
    import os
    import numpy as np
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import lib.timestepping as timestepping

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    f = jnp.fft.rfft2(input_dict['fields'])
    dt = input_dict['T'] / steps
    v_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=True)
    one_step = jax.jit(lambda f: timestepping.tdrk4(f, dt, 1, v_fn))

    def body(f, _):
        obs = _compute_observables(f, param_dict)
        real = jnp.fft.irfft2(f)
        f_new = one_step(f)
        return f_new, (real, obs)

    _, (reals, obs_traj) = jax.lax.scan(body, f, None, length=steps)
    obs_traj = np.asarray(obs_traj)

    vmin, vmax = -10, 10
    extent = [0, 2*jnp.pi, 0, 2*jnp.pi]
    ticks = [0, jnp.pi, 2*jnp.pi]
    ticklabels = ["0", r"$\pi$", r"$2\pi$"]

    plt.ioff()
    figure, axis = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    field_axes = axis[:2]
    proj_ax = axis[2]

    images, counters = [], []
    for ax, name in zip(field_axes, ["vorticity", "current"]):
        im = ax.imshow(np.zeros(reals.shape[2:]), vmin=vmin, vmax=vmax, cmap='bwr',
                        origin='lower', extent=extent, aspect='equal')
        ax.set_title(name)
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
        figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[vmin, 0, vmax])
        images.append(im)
        counters.append(ax.text(0.02, 0.98, "", transform=ax.transAxes,
                                 ha='left', va='top', color='black'))

    proj_ax.plot(obs_traj[:, 0], obs_traj[:, 1], color='gray', lw=1)
    proj_ax.set_xlabel("energy injection")
    proj_ax.set_ylabel("dissipation")
    proj_ax.set_title("2D projection")
    marker, = proj_ax.plot([], [], 'ro', markersize=8)

    figure.canvas.draw()
    figure.set_constrained_layout(False)
    backgrounds = [figure.canvas.copy_from_bbox(ax.bbox) for ax in axis]

    frames = []
    for t in range(steps):
        for ax, im, counter, data, bg in zip(field_axes, images, counters, reals[t], backgrounds[:2]):
            figure.canvas.restore_region(bg)
            im.set_data(np.asarray(data).transpose())
            counter.set_text(f"frame {t}")
            ax.draw_artist(im)
            ax.draw_artist(counter)
            figure.canvas.blit(ax.bbox)

        figure.canvas.restore_region(backgrounds[2])
        marker.set_data([obs_traj[t, 0]], [obs_traj[t, 1]])
        proj_ax.draw_artist(proj_ax.lines[0])
        proj_ax.draw_artist(marker)
        figure.canvas.blit(proj_ax.bbox)

        frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)

    plt.close(figure)

    imageio.mimsave(filename, frames, fps=fps, loop=0)
    print(f"Wrote {len(frames)} frames to {filename}")

    return filename


def make_gif(fs, filename="figures/turbulence.gif", fps=8):
    '''
    PURPOSE:
    Render a trajectory of snapshots (as produced by generate_turbulent_trajectory)
    to an animated gif. Builds one figure and reuses it across frames (via set_data)
    instead of rebuilding + PNG-encoding a fresh figure per frame, which is ~10x faster.

    INPUT:
    fs       - trajectory snapshots in spectral space, shape [steps,2,n,n//2+1]
    filename - output path for the gif
    fps      - frames per second

    OUTPUT:
    filename - the path the gif was written to
    '''
    import os
    import numpy as np
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    real = jnp.fft.irfft2(fs)  # [steps,2,n,n]

    vmin, vmax = -10, 10
    extent = [0, 2*jnp.pi, 0, 2*jnp.pi]
    ticks = [0, jnp.pi, 2*jnp.pi]
    ticklabels = ["0", r"$\pi$", r"$2\pi$"]

    plt.ioff()
    figure, axis = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    images, counters = [], []
    for ax, name in zip(axis, ["vorticity", "current"]):
        im = ax.imshow(np.zeros(real.shape[2:]), vmin=vmin, vmax=vmax, cmap='bwr',
                        origin='lower', extent=extent, aspect='equal')
        ax.set_title(name)
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
        figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[vmin, 0, vmax])
        images.append(im)
        counters.append(ax.text(0.02, 0.98, "", transform=ax.transAxes,
                                 ha='left', va='top', color='black'))

    # Render once fully, then freeze layout so only the image/text artists
    # need to be redrawn (via blitting) on each subsequent frame.
    figure.canvas.draw()
    figure.set_constrained_layout(False)
    backgrounds = [figure.canvas.copy_from_bbox(ax.bbox) for ax in axis]

    frames = []
    for t in range(real.shape[0]):
        for ax, im, counter, data, bg in zip(axis, images, counters, real[t], backgrounds):
            figure.canvas.restore_region(bg)
            im.set_data(np.asarray(data).transpose())
            counter.set_text(f"frame {t}")
            ax.draw_artist(im)
            ax.draw_artist(counter)
            figure.canvas.blit(ax.bbox)
        frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)

    plt.close(figure)

    imageio.mimsave(filename, frames, fps=fps, loop=0)
    print(f"Wrote {len(frames)} frames to {filename}")

    return filename


def plot_recurrence_diagram(dist):
    '''
    PURPOSE:
    Plot the recurrence diagram interactively. Requires "%matplotlib widget"
    to have been run in the notebook. Click on a pixel to print and mark its
    (row, col) snapshot indices -- these are what you pass as `idx` to
    utils.create_state_from_turb().
    '''
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    im = ax.imshow(dist, origin='lower', cmap='jet', vmin=0, vmax=1)
    figure.colorbar(im, ax=ax)
    ax.set_title("recurrence diagram (click a pixel)")
    ax.set_xlabel("snapshot index")
    ax.set_ylabel("snapshot index")

    marker, = ax.plot([], [], 'wx', markersize=12, markeredgewidth=2)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        print(f"clicked pixel: idx = [{row}, {col}]  (dist={dist[row, col]:.3f})")
        marker.set_data([col], [row])
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    return figure


#This function in particular was generated with the help of ChatGPT
#Don't trust that it is written correctly
def line_search_unravel(x, step, obj, unravel_fn, f_vec, max_iters=20):
    """
    Perform a fixed-length JAX-friendly line search that compares candidates
    to the original f_vec (never overwrites it).
    Returns x_final (flat) after the line search.
    """
    norm_orig = jnp.linalg.norm(f_vec)

    # loop state: (x, damp, done)
    init_state = (x, 1.0, False)

    def body(i, state):
        x_curr, damp, done = state

        # If already done, keep state unchanged (but fori_loop still calls body)
        def compute_update(state):
            x_c, d, _ = state
            x_temp = x_c - d * step

            # Evaluate objective at x_temp
            temp_dict = unravel_fn(x_temp)
            f_temp = obj(temp_dict)
            f_temp_vec = jax.flatten_util.ravel_pytree(f_temp)[0]
            norm_temp = jnp.linalg.norm(f_temp_vec)

            improved = norm_temp < norm_orig
            do_update = jnp.logical_and(improved, jnp.logical_not(done))

            x_new = jnp.where(do_update, x_temp, x_c)
            d_new = jnp.where(do_update, d, d / 2.0)
            done_new = jnp.logical_or(done, improved)

            return (x_new, d_new, done_new)

        # Use lax.cond to skip expensive computation when done==True
        new_state = jax.lax.cond(done, lambda s: s, compute_update, state)
        return new_state

    final_state = jax.lax.fori_loop(0, max_iters, body, init_state)
    x_final = final_state[0]
    damp = final_state[1]
    return x_final, damp

def run_newton(input_dict, param_dict, use_transpose, maxit=32, mode="TDRK4",
               inner=16, outer=1, do_line_search=True, default_damp=0.1,
               save_dir=None, save_every=1, min_abs_T=None):
    '''
    PURPOSE:
    Converge an RPO guess with Newton-GMRES. The only thing that distinguishes
    plain Newton from Gauss-Newton here is `use_transpose`: when True, GMRES
    is preconditioned with the Jacobian transpose (the adjoint), which is
    equivalent to solving the Gauss-Newton normal equations (J^T J) s = -J^T f
    instead of the Newton step Js = -f. In practice Gauss-Newton needs far
    fewer Krylov directions and tolerates larger steps, at the cost of being
    a descent method (so it can stall in a local minimum of |f|^2 rather than
    a true root).

    This hides: JIT compilation of the whole Newton step, flatten/unflatten
    of pytrees to vectors for GMRES, the line search, dealiasing after each
    step, and per-step diagnostics/checkpointing.

    INPUT:
    input_dict    - RPO guess: {'fields', 'T', 'sx'}
    param_dict    - grid + physical + checkpointing parameters
    use_transpose - False: Newton.  True: Gauss-Newton (transpose-preconditioned GMRES)
    maxit         - number of Newton/Gauss-Newton steps
    mode          - timestepping scheme used inside the objective
    inner         - GMRES Krylov subspace dimension
    outer         - number of GMRES restarts (leave at 1 unless memory-limited)
    do_line_search - backtracking line search along the computed step
    default_damp  - fixed damping factor used only if do_line_search is False
    save_dir      - if given, periodically save a checkpoint + history here
    save_every    - checkpoint (and write history.json) every this many steps;
                    the final step is always saved regardless of this value

    OUTPUT:
    input_dict - converged (or partially converged) state
    history    - list of dicts with keys 'rel_err', 'residual_norm', 'gmres_residual'.
                 Also written incrementally to {save_dir}/history.json if save_dir
                 is given, so progress can be plotted from another process/notebook
                 while this is still running.
    '''
    import os
    import json

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    obj = lambda input_dict, param_dict: loss_functions.objective_RPO(input_dict, param_dict, mode)
    obj, jac = compile_objective_and_Jacobian(input_dict, param_dict, obj)

    flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]

    def relative_error_RPO(input_dict, f):
        norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x)))
        return norm(f["fields"]) / norm(input_dict["fields"])

    @jax.jit
    def step_fn(input_dict):
        f = obj(input_dict)
        b = flatten(f)
        _, unflatten_left = jax.flatten_util.ravel_pytree(f)

        A = lambda x: flatten(jac(input_dict, jax.flatten_util.ravel_pytree(input_dict)[1](x)))
        precond = []
        if use_transpose:
            _, jacT = jax.vjp(obj, input_dict, has_aux=False)
            A_T = lambda v: flatten(jacT(unflatten_left(v)))
            precond = [A_T]

        step, gmres_residual = gmres(A, b, inner, outer, preconditioner_list=precond)

        x, unravel_fn = jax.flatten_util.ravel_pytree(input_dict)
        if do_line_search:
            x, damp = line_search_unravel(x, step, obj, unravel_fn, b, max_iters=20)
        else:
            damp = default_damp
            x = x - damp * step
        new_input_dict = unravel_fn(x)

        # Same trivial-root issue as adjoint descent: T=0 (identity map) can be an
        # exact or near-exact root of the full RPO system too (e.g. when phase
        # conditions are small), so GN/Newton can also walk straight into it from
        # any starting point. Floor |T| away from zero here as well.
        if min_abs_T is not None:
            T = new_input_dict['T']
            sign = jnp.where(T >= 0, 1.0, -1.0)
            new_input_dict['T'] = sign * jnp.maximum(jnp.abs(T), min_abs_T)

        rel_err = relative_error_RPO(new_input_dict, f)

        # Dealias after the step
        fields = new_input_dict['fields']
        fields = param_dict['mask'] * jnp.fft.rfft2(fields)
        fields = jnp.fft.irfft2(fields)
        new_input_dict['fields'] = fields

        return new_input_dict, rel_err, jnp.linalg.norm(b), gmres_residual

    history = []
    for i in range(maxit):
        input_dict, rel_err, residual_norm, gmres_residual = step_fn(input_dict)
        print(f"{i}: rel_err={rel_err:.3e}, |f|={residual_norm:.3e}, "
              f"gmres_rel_res={gmres_residual:.3e}, T={input_dict['T']:.3f}, sx={input_dict['sx']:.3f}")
        history.append({'rel_err': float(rel_err), 'residual_norm': float(residual_norm),
                         'gmres_residual': float(gmres_residual)})

        if save_dir is not None and (i % save_every == 0 or i == maxit - 1):
            import lib.dictionaryIO as dictionaryIO
            remove_bloat = lambda param_dict: dictionaryIO.remove_grid_information(param_dict)
            dictionaryIO.save_dicts(f"{save_dir}/{i}.npz", input_dict, remove_bloat(param_dict))
            with open(f"{save_dir}/history.json", "w") as fh:
                json.dump(history, fh)

    return input_dict, history


def save_state(filename, input_dict, param_dict):
    '''
    PURPOSE:
    Save a converged (or in-progress) state to disk in the same format used
    by run_adjoint_descent/run_newton checkpoints, stripping out the bulky
    grid tensors first (they're cheaply recomputed by load_latest_checkpoint
    / dictionaryIO.recompute_grid_information).
    '''
    import os
    import lib.dictionaryIO as dictionaryIO

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    remove_bloat = lambda param_dict: dictionaryIO.remove_grid_information(param_dict)
    dictionaryIO.save_dicts(filename, input_dict, remove_bloat(param_dict))
    print(f"Saved {filename}")


def load_latest_checkpoint(save_dir):
    '''
    PURPOSE:
    Load the most recently saved (highest iteration number) checkpoint from
    a directory produced by run_adjoint_descent or run_newton, and recompute
    the grid information that is stripped out before saving.

    OUTPUT:
    input_dict, param_dict
    '''
    import os
    import glob
    import lib.dictionaryIO as dictionaryIO

    files = glob.glob(os.path.join(save_dir, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {save_dir}")

    latest = max(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    print(f"Loading {latest}")

    input_dict, param_dict = dictionaryIO.load_dicts(latest)
    param_dict = dictionaryIO.recompute_grid_information(input_dict, param_dict)
    return input_dict, param_dict


def plot_newton_comparison(histories, labels):
    '''
    PURPOSE:
    Compare convergence of multiple Newton-type runs (e.g. Newton vs Gauss-Newton),
    as produced by run_newton, on a single semilog plot of relative error per iteration.
    '''
    import matplotlib.pyplot as plt
    figure = plt.figure()
    for history, label in zip(histories, labels):
        plt.semilogy([h['rel_err'] for h in history], marker='o', label=label)
    plt.xlabel("iteration")
    plt.ylabel("relative error")
    plt.title("Newton vs Gauss-Newton convergence")
    plt.legend()
    plt.show()
    return figure


def run_adjoint_descent(input_dict, param_dict, maxit=1024, lr=1e-2,
                         save_every=32, save_dir="temp_data/adjoint_descent",
                         mode="RK4", print_every=1, max_walltime=None, min_abs_T=None):
    '''
    PURPOSE:
    Refine an RPO guess (input_dict, param_dict) with gradient descent (ADAM)
    on the mismatch loss. Hides ADAM state init/update, JIT compilation,
    dealiasing after each step, periodic checkpointing to disk, and logging.

    INPUT:
    input_dict  - optimizable state: {'fields', 'T', 'sx'}
    param_dict  - grid + physical parameters (and 'ministeps'/'num_checkpoints'
                  for fixed-timestep checkpointed integration)
    maxit       - number of ADAM steps to take
    lr          - ADAM learning rate
    save_every  - checkpoint to disk every this many steps
    save_dir    - directory to write checkpoints into
    mode        - timestepping scheme used inside the loss ("RK4", etc.)
    print_every - print a diagnostic line every this many steps

    OUTPUT:
    input_dict - refined state
    history    - list of loss values, one per step
    '''
    import time
    import os
    import lib.dictionaryIO as dictionaryIO

    os.makedirs(save_dir, exist_ok=True)

    m, v = adam.init_adam(input_dict)
    update_fn = jax.jit(adam.adam_update)

    loss_fn = lambda input_dict: loss_functions.loss_RPO(input_dict, param_dict, mode)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    history = []
    loop_start = time.time()
    for t in range(maxit):
        if max_walltime is not None and (time.time() - loop_start) > max_walltime:
            print(f"adjoint descent hit max_walltime={max_walltime}s at step {t}, stopping early")
            break
        start = time.time()
        (loss, info), grad = grad_fn(input_dict)
        walltime = time.time() - start

        if isinstance(info, dict) and info.get("completed") is False:
            print("ERROR: did not complete integration...")
            print("Integration info for debugging:")
            print(info)
            break

        input_dict, m, v = update_fn(input_dict, grad, m, v, t + 1, lr=lr,
                                      beta1=0.9, beta2=0.999, eps=1e-6)

        # loss_RPO has an exact global minimum at T=0 for ANY field (zero-duration
        # integration is trivially the identity), so unconstrained descent will
        # always collapse there unless we floor |T| away from zero.
        if min_abs_T is not None:
            T = input_dict['T']
            sign = jnp.where(T >= 0, 1.0, -1.0)
            input_dict['T'] = sign * jnp.maximum(jnp.abs(T), min_abs_T)

        # Dealias fields after the update
        f = input_dict['fields']
        f = jnp.fft.rfft2(f) * param_dict['mask']
        f = jnp.fft.irfft2(f)
        input_dict['fields'] = f

        history.append(float(loss))
        if t % print_every == 0:
            print(f"{t}: loss={loss:.6f}, walltime={walltime:.3f}, "
                  f"T={input_dict['T']:.3f}, sx={input_dict['sx']:.3f}")

        if t % save_every == 0 or t == maxit - 1:
            remove_bloat = lambda param_dict: dictionaryIO.remove_grid_information(param_dict)
            dictionaryIO.save_dicts(f"{save_dir}/{t}.npz", input_dict, remove_bloat(param_dict))

    return input_dict, history


def compare_initial_final(turb_dict, idx, input_dict, param_dict):
    '''
    PURPOSE:
    Visually sanity-check an RPO guess: show the initial snapshot f1 side by
    side with the final snapshot f2 after applying the symmetry g found by
    find_optimal_symmetry (stored in input_dict['sx'] and
    param_dict['shift_reflect_ny'/'rot']). If the guess is good, these two
    columns should look nearly identical.

    INPUT:
    turb_dict  - dict with key 'fs', the turbulent trajectory (spectral space)
    idx        - [i1, i2] snapshot indices used to build the RPO guess
    input_dict - output of create_state_from_turb (needs 'sx')
    param_dict - output of create_state_from_turb (needs 'shift_reflect_ny', 'rot')
    '''
    import matplotlib.pyplot as plt

    f1 = turb_dict['fs'][idx[0] - 1, :, :, :]
    f2 = turb_dict['fs'][idx[1] - 1, :, :, :]

    g_f2 = symm.shift_x(f2, input_dict['sx'], param_dict)
    g_f2 = symm.shift_reflect(g_f2, param_dict['shift_reflect_ny'], param_dict)
    if param_dict['rot']:
        g_f2 = symm.rot180(g_f2)

    vmin, vmax = -10, 10
    extent = [0, 2*jnp.pi, 0, 2*jnp.pi]
    ticks = [0, jnp.pi, 2*jnp.pi]
    ticklabels = ["0", r"$\pi$", r"$2\pi$"]

    figure, axis = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    columns = [("initial: f1", jnp.fft.irfft2(f1)), ("final (symmetry-shifted): g(f2)", jnp.fft.irfft2(g_f2))]
    rows = ["vorticity", "current"]

    for col, (col_title, data) in enumerate(columns):
        for row, field_name in enumerate(rows):
            ax = axis[row, col]
            im = ax.imshow(data[row].transpose(), vmin=vmin, vmax=vmax, cmap='bwr',
                            origin='lower', extent=extent, aspect='equal')
            ax.set_title(f"{col_title}\n{field_name}" if row == 0 else field_name)
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
            figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[vmin, 0, vmax])

    return figure, axis


def create_state_from_turb( turb_dict, idx, param_dict ):
    #Get conditions for RPO guess
    f1 = turb_dict['fs'][idx[0]-1,:,:,:]
    f2 = turb_dict['fs'][idx[1]-1,:,:,:]
    
    g = symm.find_optimal_symmetry(f1,f2,param_dict,nx=64)
    
    #Return the state in real space
    f = jnp.fft.irfft2(f1)

    #Period
    T = param_dict['dt'] * param_dict['ministeps'] * (idx[1] - idx[0])

    #number of timesteps 
    steps = param_dict['ministeps'] * (idx[1] - idx[0])
    steps = int(steps) #JAX complains if we do not cast steps

    param_dict.update({ 'steps': steps, 'shift_reflect_ny': g[1], 'rot': g[2] } )

    #Create a dictionary of optimizable field
    input_dict = {"fields": f, "T": T, "sx": g[0]}

    #Delete keys from the turbulent trajectory param_dict that we won't need anymore to avoid confusion
    del param_dict['dt']
    del param_dict['ministeps']

    return input_dict, param_dict






def compile_objective_and_Jacobian( input_dict, param_dict, obj ):
    
    #Capture param_dict and JIT the objective function
    objective = jax.jit( lambda input_dict: obj(input_dict, param_dict) )

    #Compile the objective function.
    f = objective(input_dict)

    import time
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    walltime0 = stop - start

    #Define the Jacobian action and compile it
    jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )
    _ = jac( input_dict, input_dict )

    start = time.time()
    Jf = jac( input_dict, input_dict )
    stop = time.time()
    walltime1 = stop - start

    print(f"Evaluating objective: {walltime0:.3} seconds")
    print(f"Evaluating Jacobian: {walltime1:.3} seconds")
    #print(f"Evaluating Jacobian transpose: {walltime2:.3} seconds")
    return objective, jac





def choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict ):
    #Check that all options are within their allowed values
    assert shooting_mode in {"single_shooting", "multi_shooting"}
    assert integrate_mode in {"adaptive", "fixed_timesteps"}


    if shooting_mode == "single_shooting" and integrate_mode == "fixed_timesteps":
        print(f"Choosing single shooting with fixed timesteps:")
        print(f"steps = {param_dict['steps']}")
        print(f"num_checkpoints = {num_checkpoints}")
        #define number of segements for memory checkpointing
        param_dict.update(  {"ministeps": int(param_dict["steps"]//num_checkpoints), "num_checkpoints": int(num_checkpoints)})
        #Define the RPO objective function
        obj = loss_functions.objective_RPO_with_checkpoints

    if shooting_mode == "single_shooting" and integrate_mode == "adaptive":
        print(f"Choosing single shooting with adaptive timestepping:")
        obj = lambda input_dict, param_dict: loss_functions.objective_RPO_adaptive( input_dict, param_dict, adaptive_dict )

    if shooting_mode == "multi_shooting":
        print(f"ERROR: multishooting is experimental and not quite implemented. Yell at Matt.")
        exit()
        #Define the RPO objective function
        #obj = loss_functions.objective_RPO_multishooting

    return obj, param_dict
