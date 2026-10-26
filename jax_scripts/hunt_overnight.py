'''
Autonomous continuous hunt for converged short-period RPOs.
Loop forever: generate turbulence -> scan recurrence diagram for short-period
near recurrences (filtered to genuinely turbulent snapshots) -> adjoint
descent (capped) -> Gauss-Newton (greedy, small Krylov) -> Newton (greedy,
small Krylov). Detect stalls/timeouts early and move to the next candidate.
Every genuine (non-degenerate) convergence is logged with full
reproducibility info to OUT_DIR/SOLUTIONS.md and OUT_DIR/solutions.jsonl, and
the search keeps going afterward -- multiple solutions are the goal.
'''

import copy
import json
import os
import time

import jax
import jax.numpy as jnp

import lib.utils as utils

print(jax.devices())

OUT_DIR = "temp_data/hunt"
os.makedirs(OUT_DIR, exist_ok=True)

N = 64
NU = 1/20
ETA = 1/20
B0 = [0.0, 0.1]
U0 = [0.0, 0.0]
FORCING_STR = "lambda x,y : -4*jnp.cos(4*y)"

DT = 1/128
TRANSIENT_TIME = 24.0
MACRO_STEPS = 256
MICRO_STEPS = 16          # finer sampling -> shorter candidate periods available
MAX_GAP = 24              # only consider candidates with idx[1]-idx[0] <= this (short period)
MIN_GAP = 12              # must be large enough that closeness isn't just temporal continuity

ADJOINT_MAXIT = 400
ADJOINT_LR = 2e-2
ADJOINT_MAX_WALLTIME = 8 * 60

GN_MAXIT = 20
GN_INNER = 8

NEWTON_MAXIT = 15
NEWTON_INNER = 64

# Expensive final polish: only run on near-misses (CONVERGED_TOL < rel_err < POLISH_THRESHOLD)
# after the cheap Newton pass. GMRES restarts (outer>1) buy real accuracy near the root
# that a bigger single Krylov subspace doesn't reliably buy (tested: inner=96 single-shot
# was unstable; inner=64 with outer=3 restarts cleanly improved 2.9e-2 -> 1.3e-3 on a test
# candidate). Too expensive to run on every candidate, so gate it on already-being-close.
POLISH_THRESHOLD = 1e-2
POLISH_MAXIT = 15
POLISH_INNER = 64
POLISH_OUTER = 3

CONVERGED_TOL = 1e-10
MIN_PERIOD = 0.3          # reject convergence to a trivial near-zero-period "orbit"
MAX_WALLTIME = 40 * 60    # give up on a candidate (adjoint+GN+Newton[+polish]) past this many seconds


class CandidateTimeout(Exception):
    pass


def check_walltime(start_time, tag):
    if time.time() - start_time > MAX_WALLTIME:
        raise CandidateTimeout(f"{tag} exceeded {MAX_WALLTIME/60:.0f} min wall time budget")


def is_stalled(history, window=6, tol=1e-3):
    if len(history) < window:
        return False
    recent = [h['rel_err'] for h in history[-window:]]
    if abs(recent[0] - recent[-1]) < tol * max(recent[0], 1e-30):
        return True
    recent_gmres = [h['gmres_residual'] for h in history[-window:]]
    if all(g > 0.9 for g in recent_gmres):
        return True
    return False


def find_short_period_candidates(fs, dist, min_gap, max_gap, n_candidates=6, energy_frac=0.5):
    '''
    Only consider pairs that are both genuinely turbulent (energy above
    energy_frac * mean trajectory energy) -- otherwise the "closest" pairs by
    raw L2 distance are just quiescent/near-laminar stretches that are
    trivially close to each other, not real recurrences.
    '''
    steps = dist.shape[0]
    energy = jnp.mean(jnp.square(jnp.abs(fs)), axis=(1, 2, 3))
    energy = jnp.asarray(energy)
    threshold = energy_frac * float(jnp.mean(energy))

    candidates = []
    for i in range(steps):
        if energy[i] < threshold:
            continue
        for j in range(i + min_gap, min(i + max_gap, steps)):
            if energy[j] < threshold:
                continue
            candidates.append((float(dist[i, j]), i + 1, j + 1))  # 1-indexed like idx convention
    candidates.sort(key=lambda c: c[0])
    print(f"  {len(candidates)} turbulent candidates found (energy threshold={threshold:.3e})", flush=True)
    return candidates[:n_candidates]


def log_solution(tag, seed, i1, i2, ministeps, nt_out, final_err, period, elapsed, worker_id):
    '''
    Append a full reproducibility record for a genuine converged solution:
    a JSON line (machine-readable) and a markdown section (human-readable,
    explains exactly how this solution was found and how to reproduce it).
    '''
    transient_time = TRANSIENT_TIME + (seed % 17) * 0.97
    record = {
        "tag": tag, "worker_id": worker_id, "seed": seed, "idx": [i1, i2],
        "n": N, "nu": NU, "eta": ETA, "b0": B0, "u0": U0, "forcing_str": FORCING_STR,
        "dt": DT, "transient_time": transient_time,
        "macro_steps": MACRO_STEPS, "micro_steps": MICRO_STEPS, "ministeps": ministeps,
        "adjoint_maxit": ADJOINT_MAXIT, "adjoint_lr": ADJOINT_LR,
        "gn_maxit": GN_MAXIT, "gn_inner": GN_INNER,
        "newton_maxit": NEWTON_MAXIT, "newton_inner": NEWTON_INNER,
        "final_rel_err": final_err, "T": period, "sx": float(nt_out['sx']),
        "walltime_s": elapsed, "npz_path": f"{OUT_DIR}/{tag}_converged.npz",
    }

    with open(f"{OUT_DIR}/solutions.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")

    md = f"""
## Solution: {tag}

Found by worker {worker_id}, converged in {elapsed:.1f}s. rel_err={final_err:.3e}, T={period:.4f}, sx={record['sx']:.4f}.

**Exact recipe to reproduce:**
1. `utils.initialize_system(n={N}, nu={NU}, eta={ETA}, b0={B0}, u0={U0}, seed={seed}, forcing_str="{FORCING_STR}")`
2. `utils.evolve_transient(f, param_dict, dt={DT}, transient_time={transient_time})`
3. `utils.generate_turbulent_trajectory(f, param_dict, dt={DT}, steps={MACRO_STEPS}, ministeps={MICRO_STEPS})`
4. `utils.create_state_from_turb(turb_dict, idx=[{i1},{i2}], param_dict)` (found via energy-filtered recurrence-diagram scan: only snapshot pairs with energy > 0.5x trajectory mean were considered, to avoid trivial near-laminar false recurrences; gap between idx must be in [{MIN_GAP},{MAX_GAP}] for a short period)
5. Set `param_dict['ministeps']={ministeps}`, `param_dict['num_checkpoints']=steps//{ministeps}`
6. `utils.run_adjoint_descent(..., maxit={ADJOINT_MAXIT}, lr={ADJOINT_LR}, max_walltime={ADJOINT_MAX_WALLTIME}, min_abs_T={MIN_PERIOD})` (min_abs_T floors |T| away from 0 -- the loss has a trivial exact-zero minimum at T=0 for any field, so this is essential, not optional)
7. `utils.run_newton(..., use_transpose=True, maxit={GN_MAXIT}, inner={GN_INNER}, mode="RK4", min_abs_T={MIN_PERIOD})` (Gauss-Newton; min_abs_T here too -- GN's own root-finding can also walk into the T=0 trivial root)
8. `utils.run_newton(..., use_transpose=False, maxit={NEWTON_MAXIT}, inner={NEWTON_INNER}, mode="RK4", min_abs_T={MIN_PERIOD})` (finishing Newton)
9. If rel_err was a near-miss (between {CONVERGED_TOL} and {POLISH_THRESHOLD}): `utils.run_newton(..., use_transpose=False, maxit={POLISH_MAXIT}, inner={POLISH_INNER}, outer={POLISH_OUTER}, mode="RK4", min_abs_T={MIN_PERIOD})` (expensive polish with GMRES restarts -- buys real accuracy near the root that a bigger single Krylov subspace doesn't)

Converged state saved at `{record['npz_path']}`.
"""
    with open(f"{OUT_DIR}/SOLUTIONS.md", "a") as fh:
        fh.write(md)

    print(f"\n*** LOGGED SOLUTION: {tag} ***", flush=True)


def try_candidate(seed, attempt_tag, worker_id):
    f, pd = utils.initialize_system(N, NU, ETA, B0, U0, seed, FORCING_STR)
    # Vary transient time per seed -- a fixed transient tends to funnel different
    # random initial conditions toward a similar point in phase space by the same
    # macro-step, which is why the search kept rediscovering the same orbit across
    # many different seeds. Desynchronizing this encourages genuine diversity.
    transient_time = TRANSIENT_TIME + (seed % 17) * 0.97
    f = utils.evolve_transient(f, pd, DT, transient_time)
    fs, dist = utils.generate_turbulent_trajectory(f, pd, DT, MACRO_STEPS, MICRO_STEPS)
    pd.update({'dt': DT, 'ministeps': MICRO_STEPS, 'steps': MACRO_STEPS})
    turb_dict = {'fs': fs}

    candidates = find_short_period_candidates(fs, dist, MIN_GAP, MAX_GAP)
    n_found = 0

    for score, i1, i2 in candidates:
        tag = f"{attempt_tag}_i{i1}_j{i2}"
        print(f"\n=== seed={seed} idx=[{i1},{i2}] score={score:.3f} tag={tag} ===", flush=True)
        candidate_start = time.time()
        try:
            pd_c = copy.deepcopy(pd)
            input_dict, pd_c = utils.create_state_from_turb(turb_dict, [i1, i2], pd_c)
            ministeps = 8
            if pd_c['steps'] % ministeps != 0:
                ministeps = 4
            if pd_c['steps'] % ministeps != 0:
                continue
            pd_c.update({'ministeps': ministeps, 'num_checkpoints': pd_c['steps'] // ministeps})

            # Adjoint descent -- hard cap, take whatever progress we get
            check_walltime(candidate_start, tag)
            input_dict, history = utils.run_adjoint_descent(
                input_dict, pd_c, maxit=ADJOINT_MAXIT, lr=ADJOINT_LR, save_every=ADJOINT_MAXIT,
                save_dir=f"{OUT_DIR}/{tag}_adjoint", print_every=50, max_walltime=ADJOINT_MAX_WALLTIME,
                min_abs_T=MIN_PERIOD)
            if history[-1] > 1.0:
                print(f"adjoint descent didn't get close enough (loss={history[-1]:.3f}), skipping", flush=True)
                continue

            # Gauss-Newton -- small/greedy Krylov subspace for speed
            check_walltime(candidate_start, tag)
            gn_out, gn_hist = utils.run_newton(
                copy.deepcopy(input_dict), copy.deepcopy(pd_c),
                use_transpose=True, maxit=GN_MAXIT, inner=GN_INNER, mode="RK4",
                min_abs_T=MIN_PERIOD)
            if is_stalled(gn_hist) and gn_hist[-1]['rel_err'] > 0.2:
                print(f"Gauss-Newton stalled at rel_err={gn_hist[-1]['rel_err']:.3e}, skipping", flush=True)
                continue
            if abs(float(gn_out['T'])) < MIN_PERIOD:
                print(f"GN converged to degenerate near-zero-period solution (T={float(gn_out['T']):.3e}), "
                      f"skipping before wasting time on Newton", flush=True)
                continue
            # With min_abs_T active, T can no longer collapse below the floor -- but
            # it can get pinned exactly AT the floor while the method keeps wanting
            # to push it lower. Only treat this as truly hopeless/degenerate if GN
            # hasn't even gotten close -- rel_err of a few percent is well within
            # Newton's quadratic basin and deserves a real shot with a larger
            # Krylov subspace, not an early bailout.
            if abs(abs(float(gn_out['T'])) - MIN_PERIOD) < 1e-6 and gn_hist[-1]['rel_err'] > 0.2:
                print(f"GN pinned at the T floor ({MIN_PERIOD}) and nowhere near converged "
                      f"(rel_err={gn_hist[-1]['rel_err']:.3e}), skipping before wasting time on Newton", flush=True)
                continue

            # Finishing Newton -- small/greedy Krylov subspace for speed
            check_walltime(candidate_start, tag)
            nt_out, nt_hist = utils.run_newton(
                copy.deepcopy(gn_out), copy.deepcopy(pd_c),
                use_transpose=False, maxit=NEWTON_MAXIT, inner=NEWTON_INNER, mode="RK4",
                min_abs_T=MIN_PERIOD)

            final_err = nt_hist[-1]['rel_err']
            period = float(abs(nt_out['T']))
            print(f"Fast Newton rel_err = {final_err:.3e}, T = {period:.4f}", flush=True)

            # Near-miss: worth an expensive polish pass (GMRES restarts) before giving up.
            if CONVERGED_TOL <= final_err < POLISH_THRESHOLD:
                check_walltime(candidate_start, tag)
                print(f"Near miss (rel_err={final_err:.3e}) -- running expensive polish pass", flush=True)
                nt_out, polish_hist = utils.run_newton(
                    copy.deepcopy(nt_out), copy.deepcopy(pd_c),
                    use_transpose=False, maxit=POLISH_MAXIT, inner=POLISH_INNER,
                    outer=POLISH_OUTER, mode="RK4", min_abs_T=MIN_PERIOD)
                final_err = polish_hist[-1]['rel_err']
                period = float(abs(nt_out['T']))

            # Note: no walltime check here -- Newton already ran and produced a result,
            # so always evaluate/log it even if this pushed us past the budget. The
            # walltime cap should only ever prevent STARTING more expensive work, never
            # discard work that already finished.
            elapsed = time.time() - candidate_start
            print(f"FINAL rel_err = {final_err:.3e}, T = {period:.4f}, walltime={elapsed:.1f}s", flush=True)

            if period < MIN_PERIOD:
                print(f"Rejecting degenerate near-zero-period solution (T={period:.3e}), skipping", flush=True)
                continue

            utils.save_state(f"{OUT_DIR}/{tag}_converged.npz", nt_out, pd_c)

            if final_err < CONVERGED_TOL:
                log_solution(tag, seed, i1, i2, ministeps, nt_out, final_err, period, elapsed, worker_id)
                n_found += 1

        except CandidateTimeout as e:
            print(f"TIMEOUT on {tag}: {e}", flush=True)
            continue
        except Exception as e:
            print(f"EXCEPTION on {tag}: {e}", flush=True)
            continue

    return n_found


def main():
    import sys
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    seed = 1000 + worker_id
    total_found = 0
    while True:
        total_found += try_candidate(seed, f"seed{seed}", worker_id)
        print(f"worker {worker_id}: {total_found} solutions found so far.", flush=True)
        seed += n_workers


if __name__ == "__main__":
    main()
