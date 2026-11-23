#!/usr/bin/env python3
"""
firefly_swarm_sensor_sweep.py

Minimal agent-based model linking one visual sensor property (temporal
resolution) to a swarm-level outcome (how tightly a group synchronizes).

MODEL (deliberately small)
  Agent    Integrate-and-fire phase oscillator, as in Martin et al. 2026
           (bioRxiv 2026.01.19.700439; code: github.com/peleg-lab/led_firesync):
               dphi/dt = 1/T_i        flash when phi reaches 1, then phi -> 0
               on a perceived flash:  phi -> phi + Z(phi_eval)
           phi_eval is the phase wrapped to (-0.5, 0.5]: positive = the flash
           was seen shortly after my own flash, negative = shortly before my
           next expected flash. If the kick crosses 1 the agent flashes at once.
  Sensor   The eye samples in windows of length 1/f (f = temporal resolution in
           Hz = frame rate in frames/s; 1/f = frame duration; compare with the
           ERG flicker-fusion frequency). Each agent has
           its own window clock. All neighbor flashes that start inside one
           window merge into ONE perceived event, reported when the window
           closes. So the eye cannot count or order flashes closer than 1/f.
           This replaces the "sum over every neighbor flash" assumption of the
           single-source model with "one kick per resolved event".
           Options: self_mask (a neighbor flash in the same window as my own
           flash is not seen), latency_comp (the agent corrects for the mean
           sensor latency 1/(2f)), k (each agent sees only k random neighbors;
           a stand-in for sensing range / field of view).
  Outcome  From the last `tail` seconds of each run:
             spread_ms  median time between a flash and the nearest flash of
                        each other agent (0 = perfect zero-lag synchrony).
                        This is a median lag, not a standard deviation. For a
                        synchronized group the two are close (N = 20: 8.9 vs
                        8.9 ms at 40 Hz, 25.6 vs 25.2 ms at 10 Hz)
             R          Kuramoto order parameter (1 = zero-lag synchrony)
             period_s   median individual flash period

RESPONSE RULES (--prc)
  measured   Parametric tanh fit of the measured P. frontalis PRC (Table S3 of
             the preprint; default = the 700 ms LED row, closest to the natural
             period of 0.64 s). Same function as R_tanh_step in led_firesync.
  restoring  Generic synchronizing reference rule, Z = -gain*sin(2*pi*phi_eval)
             (a pulsed Kuramoto rule: a flash seen just before mine advances
             me, just after mine delays me).

WHAT A FIRST RUN SHOWED (Nov 2026; defaults, 20 runs of 60 s per point)
  restoring: the group synchronizes, and the flash-time spread is about 0.3 to
             0.5 of the sensor window 1/f: about 25 ms at 10 Hz, 8 to 10 ms at
             35 to 50 Hz (the firefly eye), 5 ms at 100 Hz, 1 ms or less at
             1 kHz. Synchrony is lost below about 5 Hz (N = 5), 7 Hz (N = 20),
             10 Hz (N = 50): larger groups need a faster sensor.
  measured:  N mutually coupled agents do NOT settle at zero lag at any f
             (R about 0.2; for N = 50 about half the agents stop flashing).
             Near zero lag the fitted single-source PRC shortens the next
             cycle of the agent that flashed first and lengthens the cycle of
             the agent that flashed second, which pushes them apart (two
             agents settle about 250 ms apart). So the open modeling step is
             the rule for mutual, multi-source interaction, not the sensor.
  3 frames/s: never synchronizes (0 of 20 runs after 20 min), and a group that
             starts synchronized falls apart within 30 s. At 5 frames/s (N = 20)
             synchrony is intermittent: 0 to 3 of 20 runs at any one time.
  Not in the model yet: flash duration and brightness (detectability, range),
  spatial acuity (which neighbors can be told apart), motion.

USAGE
  python firefly_swarm_sensor_sweep.py --report           # default sweep + two-panel report figure
  python firefly_swarm_sensor_sweep.py --report --from-csv sweep_restoring.csv   # redraw that figure
  python firefly_swarm_sensor_sweep.py                    # default sweep + one-panel figure
  python firefly_swarm_sensor_sweep.py --prc measured     # same sweep, measured PRC
  python firefly_swarm_sensor_sweep.py --single 40 --n 20 # one condition, prints metrics
  python firefly_swarm_sensor_sweep.py --k 8              # each agent sees 8 neighbors
  python firefly_swarm_sensor_sweep.py --from-csv sweep_restoring.csv   # redraw figure
  python firefly_swarm_sensor_sweep.py --schematic        # figure of the model and its parameters
  python firefly_swarm_sensor_sweep.py --rates 1000 40 3  # the group at three sensor rates
Requires numpy and matplotlib only. Default sweep takes a few minutes on a laptop.
"""
import argparse
import csv
import time

import numpy as np

# ----------------------------------------------------------------------------
# Response rules (phase response curves). Input: phi_eval in (-0.5, 0.5].
# ----------------------------------------------------------------------------
T_NATURAL = 0.64  # s, mean within-bout inter-flash interval of P. frontalis

# Table S3 of the preprint: best-fit tanh parameters per LED period (ms).
TABLE_S3 = {  # T_LED: (Yamp, y0, phi_c, kappa)
    300: (1.40, 1.00, 0.00, 8.0), 400: (0.60, 1.20, -0.30, 12.0),
    500: (1.20, 1.80, -0.30, 10.0), 600: (1.20, 1.80, -0.30, 8.0),
    700: (0.80, 1.20, 0.00, 10.0), 770: (1.00, 1.40, 0.30, 4.0),
    850: (1.00, 1.00, 0.00, 12.0), 1000: (1.20, 1.20, 0.15, 10.0),
}


def wrap_half(phi):
    """Wrap a phase to (-0.5, 0.5], centered on the agent's own flash."""
    return (phi + 0.5) % 1.0 - 0.5


def prc_measured(phi_eval, Yamp=0.8, y0=1.2, phi_c=0.0, kappa=10.0):
    """Z = 1 - R with R the wrapped tanh step of the preprint (Eqs. 11-16)."""
    alpha = 2.0 / (np.tanh(kappa * (0.5 - phi_c)) + np.tanh(kappa * (0.5 + phi_c)))
    u = alpha * (y0 + np.tanh(kappa * (phi_eval - phi_c)))
    return -Yamp * (u - 2.0 * np.floor((u + 1.0) / 2.0))


def prc_restoring(phi_eval, gain=0.15):
    """Generic synchronizing rule (pulsed Kuramoto)."""
    return -gain * np.sin(2.0 * np.pi * phi_eval)


def make_prc(name, led_ms=700, gain=0.15):
    if name == "measured":
        Yamp, y0, phi_c, kappa = TABLE_S3[led_ms]
        return lambda p: prc_measured(p, Yamp, y0, phi_c, kappa)
    if name == "restoring":
        return lambda p: prc_restoring(p, gain)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Simulation. All arrays have shape (trials, N): trials run side by side.
# ----------------------------------------------------------------------------
def simulate(prc, N=20, f_sensor=40.0, duration=60.0, dt=2.5e-4, trials=20, seed=0,
             T0=T_NATURAL, period_cv=0.05, cycle_noise=0.02,
             self_mask=True, latency_comp=True, k=None, phase_spread=1.0):
    """Return (flashes, T): flashes is an array of rows (time_s, trial, agent)."""
    rng = np.random.default_rng(seed)
    window = 1.0 / f_sensor
    if window < 2 * dt:
        raise ValueError("f_sensor too high for this dt; lower dt")
    T = T0 * np.exp(period_cv * rng.standard_normal((trials, N)))  # natural periods
    Tcur = T * np.exp(cycle_noise * rng.standard_normal((trials, N)))  # this cycle
    phase = rng.random((trials, N)) * phase_spread  # 1 = random start, near 0 = start synchronized
    next_close = rng.random((trials, N)) * window  # each eye has its own clock
    seen = np.zeros((trials, N), bool)  # a neighbor flashed in my open window
    own = np.zeros((trials, N), bool)  # I flashed in my open window
    if k is not None and k < N - 1:  # who sees whom: A[trial, receiver, sender]
        A = np.zeros((trials, N, N), bool)
        for tr in range(trials):
            for i in range(N):
                others = np.delete(np.arange(N), i)
                A[tr, i, rng.choice(others, size=k, replace=False)] = True
    else:
        A = None
    out = []
    for step in range(int(duration / dt)):
        t = step * dt
        phase += dt / Tcur
        # 1) sensor windows that close now report at most one merged event
        closing = t >= next_close
        if closing.any():
            event = closing & seen
            if self_mask:
                event &= ~own
            if event.any():
                est = phase[event] - (0.5 * window / Tcur[event] if latency_comp else 0.0)
                phase[event] = np.maximum(phase[event] + prc(wrap_half(est)), 0.0)
            seen[closing] = False
            own[closing] = False
            next_close[closing] += window
        # 2) flashes: natural (phase reached 1) or evoked (kick crossed 1)
        fired = phase >= 1.0
        if fired.any():
            phase[fired] = 0.0
            Tcur[fired] = T[fired] * np.exp(cycle_noise * rng.standard_normal(int(fired.sum())))
            own |= fired
            if A is None:
                n = fired.sum(axis=1, keepdims=True)
                seen |= (n - fired) > 0  # someone other than me flashed
            else:
                seen |= (A & fired[:, None, :]).any(axis=2)
            tr, ag = np.nonzero(fired)
            out.append(np.column_stack([np.full(tr.size, t), tr, ag]))
    return (np.concatenate(out) if out else np.zeros((0, 3))), T


# ----------------------------------------------------------------------------
# Swarm-level outcomes
# ----------------------------------------------------------------------------
def sync_metrics(flashes, N, trials, duration, tail=20.0):
    """Per trial: (spread_ms, R, period_s, active) over the last `tail` seconds.
    active = fraction of agents that kept flashing (at least 3 flashes in the tail)."""
    res = np.full((trials, 4), np.nan)
    t0 = duration - tail
    grid = np.arange(t0, duration - 2.0, 0.005)
    for tr in range(trials):
        f = flashes[flashes[:, 1] == tr]
        times = [np.sort(f[f[:, 2] == a, 0]) for a in range(N)]
        times = [x for x in times if (x > t0).sum() >= 3]
        res[tr, 3] = len(times) / N
        if len(times) < 2:
            continue
        # Kuramoto order parameter from phases interpolated between flashes
        ph = np.full((len(times), grid.size), np.nan)
        for a, x in enumerate(times):
            i = np.searchsorted(x, grid, side="right") - 1
            ok = (i >= 0) & (i < x.size - 1)
            ph[a, ok] = (grid[ok] - x[i[ok]]) / (x[i[ok] + 1] - x[i[ok]])
        with np.errstate(invalid="ignore"):
            R = np.nanmean(np.abs(np.nanmean(np.exp(2j * np.pi * ph), axis=0)))
        # time from each flash to the nearest flash of each other agent
        lags, periods = [], []
        for a, xa_all in enumerate(times):
            xa = xa_all[xa_all > t0]
            periods.append(np.diff(xa))
            for b, xb in enumerate(times):
                if a == b:
                    continue
                j = np.clip(np.searchsorted(xb, xa), 1, xb.size - 1)
                lags.append(np.minimum(np.abs(xa - xb[j - 1]), np.abs(xa - xb[j])))
        res[tr, :3] = (1e3 * np.median(np.concatenate(lags)), R, np.median(np.concatenate(periods)))
    return res


def run_condition(prc, N, f_sensor, duration=60.0, trials=20, seed=0, **kw):
    fl, _ = simulate(prc, N=N, f_sensor=f_sensor, duration=duration, trials=trials, seed=seed, **kw)
    m = sync_metrics(fl, N, trials, duration)
    return dict(N=N, f_sensor=f_sensor,
                spread_ms=np.nanmedian(m[:, 0]),
                spread_lo=np.nanpercentile(m[:, 0], 25), spread_hi=np.nanpercentile(m[:, 0], 75),
                R=np.nanmean(m[:, 1]), frac_sync=float(np.mean(m[:, 1] > 0.8)),
                period_s=np.nanmedian(m[:, 2]), active=np.nanmean(m[:, 3]))


# ----------------------------------------------------------------------------
# Sweep, table, figure
# ----------------------------------------------------------------------------
DEFAULT_FREQS = [2, 3, 5, 7, 10, 15, 20, 30, 35, 40, 45, 50, 60, 100, 200, 500, 1000]
FIREFLY_BAND = (35.0, 50.0)  # Hz, ERG flicker fusion of P. frontalis (about 35 at night, 40 to 50 in the daytime)
FLASH_MS = 25.0  # ms, typical duration of one P. frontalis flash
DEFAULT_NS = [5, 20, 50]
COLORS = {5: "#C8A24A", 20: "#2E8B8B", 50: "#6B3F6B"}  # amber, teal, plum


def sweep(prc_name, freqs, Ns, out_prefix, **kw):
    prc = make_prc(prc_name, kw.pop("led_ms", 700), kw.pop("gain", 0.15))
    rows = []
    for N in Ns:
        for f in freqs:
            t0 = time.time()
            r = run_condition(prc, N, float(f), **kw)
            r["prc"] = prc_name
            rows.append(r)
            print(f"{prc_name:9s} N={N:3d} f={f:6.0f} Hz  spread={r['spread_ms']:7.1f} ms  "
                  f"R={r['R']:.2f}  synced={r['frac_sync']:.2f}  period={r['period_s']:.3f} s  active={r['active']:.2f}  "
                  f"({time.time() - t0:.1f} s)", flush=True)
    with open(out_prefix + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def plot(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": "serif", "font.size": 15,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=220)
    ax.axvspan(*FIREFLY_BAND, color="#E8C9CF", lw=0, zorder=0)  # firefly ERG flicker fusion
    ax.axvline(30, color="0.55", lw=1.2, ls=(0, (4, 3)), zorder=0)  # robot camera, 30 fps
    ax.axhline(1e3 * T_NATURAL / 4, color="0.75", lw=1, zorder=0)  # expected with no synchrony
    for N in sorted({r["N"] for r in rows}):
        rr = [r for r in rows if r["N"] == N]
        x = np.array([r["f_sensor"] for r in rr])
        ax.fill_between(x, [r["spread_lo"] for r in rr], [r["spread_hi"] for r in rr],
                        color=COLORS.get(N, "0.4"), alpha=0.18, lw=0)
        ax.plot(x, [r["spread_ms"] for r in rr], "-o", color=COLORS.get(N, "0.4"),
                ms=5, lw=2.2, label=f"N = {N}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    top = ax.get_ylim()[1] * 0.93
    ax.text(FIREFLY_BAND[1] * 1.05, top, "firefly eye", va="top", ha="left", fontsize=12, color="#9C5A68")
    ax.text(28.5, top, "30 fps camera", va="top", ha="right", fontsize=12, color="0.4")
    ax.text(ax.get_xlim()[1], 1e3 * T_NATURAL / 4 * 0.93, "no synchrony", va="top", ha="right",
            fontsize=12, color="0.5")
    ax.set_xlabel("Sensor frame rate (frames/s = Hz)")
    ax.set_ylabel("Flash-time spread (ms)")
    ax.legend(frameon=False, loc="lower left")
    sec = ax.secondary_xaxis("top", functions=(lambda f: 1e3 / np.maximum(f, 1e-9), lambda d: 1e3 / np.maximum(d, 1e-9)))
    sec.set_xticks([500, 100, 10, 1])
    sec.xaxis.set_major_formatter(plain)
    sec.set_xlabel("Frame duration (ms)", labelpad=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_model(out_png, f_sensor=40.0, N=20, gain=0.15, seed=1):
    """Schematic of the model and its parameters (drawn from the model itself)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    AMBER, TEAL, PLUM, ROSE = "#C8A24A", "#2E8B8B", "#6B3F6B", "#E8C9CF"
    T = T_NATURAL
    fig = plt.figure(figsize=(11.5, 6.4), dpi=220)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.62, wspace=1.3,
                          left=0.07, right=0.985, top=0.9, bottom=0.1)
    lab = dict(fontsize=13, fontweight="bold", va="bottom", ha="left")

    # A. one agent: phase rises at 1/T, flashes at 1; a perceived flash kicks the phase by Z
    ax = fig.add_subplot(gs[0, :4])
    dt, t, ph = 1e-3, 0.0, 0.0
    kicks = [(0, 0.8), (2, 0.2)]  # (own flashes so far, phase) at which a neighbor flash is perceived
    ts, phs, own, seen = [], [], [], []
    while t < 2.75:
        ts.append(t); phs.append(ph)
        ph += dt / T; t += dt
        if kicks and len(own) == kicks[0][0] and ph >= kicks[0][1]:
            z = float(prc_restoring(wrap_half(ph), gain))
            seen.append((t, ph, ph + z))
            ts += [t, t]; phs += [ph, ph + z]
            ph += z; kicks.pop(0)
        if ph >= 1.0:
            ts += [t, t, np.nan]; phs += [1.0, 0.0, np.nan]
            own.append(t); ph = 0.0
    ax.plot(ts, phs, color=PLUM, lw=2.2)
    for x in own:
        ax.plot([x, x], [1.05, 1.17], color=AMBER, lw=3, solid_capstyle="butt", clip_on=False)
    ax.text(own[1] + 0.03, 1.11, "own flash", color="#8F7024", ha="left", va="center", fontsize=12)
    for x, p0, p1 in seen:
        ax.axvline(x, color=TEAL, lw=1.4, ls=(0, (3, 2)), zorder=0)
        ax.annotate("", xy=(x, p1), xytext=(x, p0), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))
    ax.text(seen[1][0] + 0.03, 0.93, "flash seen", color=TEAL, ha="left", va="center", fontsize=12)
    ax.text(seen[0][0] - 0.03, 0.93, "advance", color=TEAL, ha="right", va="center")
    ax.text(seen[1][0] + 0.03, 0.28, "delay", color=TEAL, ha="left", va="center")
    ax.annotate("", xy=(own[1], 0.5), xytext=(own[0], 0.5), arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    ax.text(0.5 * (own[0] + own[1]), 0.56, f"T = {T:g} s", ha="center", va="bottom", color="0.25",
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlim(0, 2.75); ax.set_ylim(0, 1.02); ax.set_yticks([0, 1])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Phase")
    ax.text(-0.09, 1.1, "A", transform=ax.transAxes, **lab)

    # B. response rule Z(phase of the perceived flash relative to own flash)
    ax = fig.add_subplot(gs[0, 4:])
    p = np.linspace(-0.5, 0.5, 400)
    ax.axhline(0, color="0.8", lw=1); ax.axvline(0, color=AMBER, lw=3, alpha=0.6)
    ax.plot(p, prc_restoring(p, gain), color=TEAL, lw=2.2)
    ax.text(-0.25, 1.13 * gain, "advance", color=TEAL, ha="center", va="bottom", fontsize=12)
    ax.text(0.25, -1.13 * gain, "delay", color=TEAL, ha="center", va="top", fontsize=12)
    ax.annotate("", xy=(0.25, gain), xytext=(0.25, 0), arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    ax.text(0.28, 0.5 * gain, "gain", ha="left", va="center", color="0.25", fontsize=12)
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_xticklabels(["before", "own flash", "after"])
    ax.set_yticks([-gain, 0, gain]); ax.set_ylim(-1.55 * gain, 1.55 * gain)
    ax.set_xlabel("When the flash is seen"); ax.set_ylabel("Phase kick Z")
    ax.text(-0.33, 1.1, "B", transform=ax.transAxes, **lab)

    # C. sensor: flashes inside one window of 1/f merge into one perceived event
    ax = fig.add_subplot(gs[1, :2])
    w = 1e3 / f_sensor
    flashes = np.array([1.25, 1.52, 1.78, 3.85, 5.65, 5.9]) * w
    for i in range(8):
        if i % 2 == 0:
            ax.axvspan(i * w, (i + 1) * w, color=ROSE, lw=0, alpha=0.6, zorder=0)
    closes = np.unique(np.ceil(flashes / w) * w)
    for x in flashes:
        ax.plot([x, x], [0.72, 1.28], color=AMBER, lw=3, solid_capstyle="butt")
        ax.plot([x, np.ceil(x / w) * w], [0.72, 0.28], color="0.7", lw=0.8)
    for x in closes:
        ax.plot([x, x], [-0.28, 0.28], color=TEAL, lw=3, solid_capstyle="butt")
    ax.annotate("", xy=(7 * w, 1.62), xytext=(6 * w, 1.62), arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
                annotation_clip=False)
    ax.text(6.5 * w, 1.7, "1/f", ha="center", va="bottom", color="0.25")
    ax.set_xlim(0, 8 * w); ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["seen", "flashes"])
    ax.set_xlabel("Time (ms)")
    ax.text(-0.27, 1.12, "C", transform=ax.transAxes, **lab)

    # D, E. swarm outcome, from one run of the model
    fl, _ = simulate(make_prc("restoring", gain=gain), N=N, f_sensor=f_sensor, duration=8.0, trials=1, seed=seed)
    ax = fig.add_subplot(gs[1, 2:4])
    m = fl[:, 0] < 3.0
    ax.scatter(fl[m, 0], fl[m, 2] + 1, s=7, color=AMBER, lw=0)
    ax.set_xlim(0, 3); ax.set_ylim(0, N + 1); ax.set_yticks([1, N])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Agent")
    ax.text(-0.27, 1.12, "D", transform=ax.transAxes, **lab)

    ax = fig.add_subplot(gs[1, 4:])
    last = np.array([fl[fl[:, 2] == a, 0].max() for a in range(N)])
    rel = 1e3 * (last - np.median(last))
    order = np.argsort(rel)
    ax.scatter(rel[order], np.arange(1, N + 1), s=22, color=AMBER, lw=0)
    i, j = order[2], order[-3]
    yy = N + 2.2
    ax.annotate("", xy=(rel[j], yy), xytext=(rel[i], yy), arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
                annotation_clip=False)
    ax.text(0.5 * (rel[i] + rel[j]), yy + 0.6, "lag", ha="center", va="bottom", color="0.25")
    lim = max(1.25 * np.abs(rel).max(), 10)
    ax.set_xlim(-lim, lim); ax.set_ylim(0, N + 1); ax.set_yticks([1, N])
    ax.set_xlabel("Time within one group flash (ms)"); ax.set_ylabel("Agent (sorted)")
    ax.text(-0.27, 1.12, "E", transform=ax.transAxes, **lab)
    fig.savefig(out_png)
    plt.close(fig)


def plot_explainer(out_png, with_sensor=True, f_sensor=40.0, N=20, gain=0.15, seed=1):
    """Compact companion to the sweep figure: what the two axes mean.
    Sensor window (x axis of the sweep) and flash-time spread (y axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    plt.rcParams.update({"font.family": "serif", "font.size": 15,
                         "axes.spines.top": False, "axes.spines.right": False})
    AMBER, TEAL, PLUM, ROSE = "#C8A24A", "#2E8B8B", "#6B3F6B", "#E8C9CF"
    head = dict(fontsize=14, color="0.35", ha="left", va="bottom")
    if with_sensor:
        fig = plt.figure(figsize=(8.6, 5.6), dpi=220)
        gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], hspace=0.78, wspace=0.3,
                              left=0.11, right=0.975, top=0.9, bottom=0.11)
        # sensor: flashes inside one window of 1/f are seen as one event
        ax = fig.add_subplot(gs[0, :])
        w = 1e3 / f_sensor
        flashes = np.array([1.25, 1.52, 1.78, 3.85, 5.65, 5.9]) * w
        for i in range(0, 8, 2):
            ax.axvspan(i * w, (i + 1) * w, color=ROSE, lw=0, alpha=0.6, zorder=0)
        for x in flashes:
            ax.plot([x, x], [0.72, 1.28], color=AMBER, lw=3.5, solid_capstyle="butt")
            ax.plot([x, np.ceil(x / w) * w], [0.72, 0.28], color="0.7", lw=0.9)
        for x in np.unique(np.ceil(flashes / w) * w):
            ax.plot([x, x], [-0.28, 0.28], color=TEAL, lw=3.5, solid_capstyle="butt")
        ax.annotate("", xy=(7 * w, 1.55), xytext=(6 * w, 1.55), annotation_clip=False,
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
        ax.text(7.1 * w, 1.55, "1/f", ha="left", va="center", color="0.25")
        ax.set_xlim(0, 8 * w); ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["seen", "flashes"])
        ax.set_xlabel("Time (ms)")
        ax.text(0, 1.2, "Sensor window", transform=ax.transAxes, **head)
        axD, axE = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    else:
        fig, (axD, axE) = plt.subplots(1, 2, figsize=(8.6, 3.5), dpi=220)
        fig.subplots_adjust(left=0.1, right=0.975, top=0.8, bottom=0.2, wspace=0.3)

    # the group: every agent flashes once per cycle
    fl, _ = simulate(make_prc("restoring", gain=gain), N=N, f_sensor=f_sensor, duration=4.0, trials=1, seed=seed)
    late = fl[(fl[:, 0] > 1.7) & (fl[:, 0] < 2.3)]
    tc = np.median(late[:, 0])
    burst = fl[np.abs(fl[:, 0] - tc) < 0.08]
    m = fl[:, 0] < 3.0
    axD.scatter(fl[m, 0], fl[m, 2] + 1, s=9, color=AMBER, lw=0)
    axD.axvspan(tc - 0.06, tc + 0.06, color="0.85", lw=0, zorder=0)
    axD.set_xlim(0, 3); axD.set_ylim(0, N + 1); axD.set_yticks([1, N])
    axD.set_xlabel("Time (s)"); axD.set_ylabel("Agent")
    axD.text(0, 1.06, f"Group of {N}", transform=axD.transAxes, **head)

    # one group flash, zoomed in: the spread is the typical lag between two agents
    rel = 1e3 * (burst[:, 0] - tc)
    y = burst[:, 2] + 1  # same row as in the left panel
    axE.scatter(rel, y, s=30, color=AMBER, lw=0, zorder=3)
    d = np.abs(rel[:, None] - rel[None, :])
    med = np.median(d[np.triu_indices(rel.size, 1)])
    i, j = np.unravel_index(np.argmin(np.abs(np.triu(d, 1) - med) + 1e9 * np.tril(np.ones_like(d))), d.shape)
    top = N + 2.6
    for q in (i, j):
        axE.scatter(rel[q], y[q], s=46, color=PLUM, lw=0, zorder=4)
        axE.plot([rel[q], rel[q]], [y[q], top], color=PLUM, lw=1, ls=(0, (3, 2)), clip_on=False)
    axE.annotate("", xy=(rel[j], top), xytext=(rel[i], top), annotation_clip=False,
                 arrowprops=dict(arrowstyle="<->", color=PLUM, lw=1.6, shrinkA=0, shrinkB=0))
    axE.text(0.5 * (rel[i] + rel[j]), top + 0.9, "flash-time spread", color=PLUM, ha="center", va="bottom")
    lim = max(1.3 * np.abs(rel).max(), 10)
    axE.set_xlim(-lim, lim); axE.set_ylim(0, N + 1); axE.set_yticks([1, N])
    axE.set_ylabel("Agent", labelpad=-6)
    axE.set_facecolor("0.95")  # same gray as the shaded group flash on the left
    axE.set_xlabel("Shaded window, zoomed in (ms)", fontsize=14)
    for yy in (0, N + 1):  # zoom lines from the shaded group flash to the right panel
        fig.add_artist(ConnectionPatch(xyA=(tc + 0.06, yy), coordsA=axD.transData,
                                       xyB=(-lim, yy), coordsB=axE.transData,
                                       color="0.7", lw=0.9))
    fig.savefig(out_png)
    plt.close(fig)
    return med


def plot_rates(out_png, rates=(1000.0, 40.0, 3.0), N=20, gain=0.15, seed=0):
    """One row per sensor rate: the group (left) and the shaded window zoomed in (right),
    with the flash-time spread marked. The zoom spans about one frame either side, a bar
    shows one frame, and the zoom box is wider for longer frames (log scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    plt.rcParams.update({"font.family": "serif", "font.size": 14,
                         "axes.spines.top": False, "axes.spines.right": False})
    AMBER, PLUM = "#C8A24A", "#6B3F6B"
    T, n = T_NATURAL, len(rates)
    fig = plt.figure(figsize=(9.4, 2.55 * n + 0.3), dpi=220)
    xL, wL, xR, wR = 0.075, 0.42, 0.60, 0.385  # left panel, and the widest zoom box
    cell = 1.0 / n
    lo, hi = -4.5, N + 2.0  # the zoom box extends a little below and above the agent rows
    prc = make_prc("restoring", gain=gain)
    for r, f in enumerate(rates):
        frame = 1e3 / f  # ms
        h = 0.48 * cell
        y0 = 1 - r * cell - 0.2 * cell - h
        unit = h / (N + 1)  # figure height of one agent row
        wfrac = float(np.clip(0.36 + 0.64 * np.log10(frame) / np.log10(350.0), 0.36, 1.0))
        axD = fig.add_axes([xL, y0, wL, h])
        axE = fig.add_axes([xR, y0 + lo * unit, wR * wfrac, (hi - lo) * unit])
        # same runs as the sweep (20 runs of 60 s), so the numbers match the sweep figure
        fl, _ = simulate(prc, N=N, f_sensor=f, duration=60.0, trials=20, seed=seed)
        met = sync_metrics(fl, N, 20, 60.0)
        spread = np.nanmedian(met[:, 0])
        tr = int(np.nanargmin(np.abs(met[:, 0] - spread)))  # draw the most typical run
        synced = met[tr, 1] > 0.8
        fl = fl[fl[:, 1] == tr]
        fl = np.column_stack([fl[:, 0] - 55.0, fl[:, 1], fl[:, 2]])  # show 3 s late in the run
        # left: the group
        m = (fl[:, 0] >= 0) & (fl[:, 0] < 3.0)
        axD.scatter(fl[m, 0], fl[m, 2] + 1, s=8, color=AMBER, lw=0)
        if synced:  # zoom on one group flash, about one frame either side
            late = fl[(fl[:, 0] > 1.7) & (fl[:, 0] < 2.3)]
            tc = np.median(late[:, 0])
            grp = fl[np.abs(fl[:, 0] - tc) < 0.08]
            half = max(frame, 1.15 * np.percentile(1e3 * np.abs(grp[:, 0] - tc), 90))
        else:  # no group flash exists: one frame either side of an arbitrary time
            tc, grp, half = 2.0, None, frame
        band = max(half / 1e3, 0.03)
        axD.axvspan(tc - band, tc + band, color="0.85", lw=0, zorder=0)
        axD.set_xlim(0, 3); axD.set_xticks([0, 1, 2]); axD.set_ylim(0, N + 1); axD.set_yticks([1, N])
        axD.set_ylabel("Agent", labelpad=-6)
        fig.text(xL, y0 + (hi + 1.7) * unit, f"{f:g} frames/s  =  {f:g} Hz,   {frame:.3g} ms per frame",
                 fontsize=14, color="0.2", ha="left", va="bottom")
        # right: the shaded window stretched in time (same agent on the same row)
        win = fl[1e3 * np.abs(fl[:, 0] - tc) <= half]
        x, y = 1e3 * (win[:, 0] - tc), win[:, 2] + 1
        axE.scatter(x, y, s=26, color=AMBER, lw=0, zorder=3)
        if grp is not None:  # flashes of this group flash that fall outside the zoom
            off = grp[1e3 * np.abs(grp[:, 0] - tc) > half]
            for t_, ag in zip(off[:, 0], off[:, 2]):
                left = t_ < tc
                axE.plot(-half if left else half, ag + 1, marker="<" if left else ">", color=AMBER,
                         ms=7, clip_on=False, zorder=5)
        d = np.abs(x[:, None] - x[None, :]) + 1e9 * (y[:, None] == y[None, :])
        i, j = np.unravel_index(np.argmin(np.abs(d - spread)), d.shape)
        y_ar = hi + 1.0
        for k in (i, j):
            axE.scatter(x[k], y[k], s=40, color=PLUM, lw=0, zorder=4)
            axE.plot([x[k], x[k]], [y[k], y_ar], color=PLUM, lw=1, ls=(0, (3, 2)), clip_on=False)
        axE.annotate("", xy=(x[j], y_ar), xytext=(x[i], y_ar), annotation_clip=False,
                     arrowprops=dict(arrowstyle="<->", color=PLUM, lw=1.6, shrinkA=0, shrinkB=0))
        val = f"{spread:.1f}" if spread < 2 else f"{spread:.0f}"
        axE.text(-half, hi + 1.7, f"flash-time spread {val} ms", color=PLUM, ha="left", va="bottom",
                 fontsize=14, clip_on=False)
        # scale bar: one frame (the perceptual window of an agent)
        xb, yb = -0.94 * half, lo + 0.9
        axE.plot([xb, xb + frame], [yb, yb], color="0.25", lw=2.5, solid_capstyle="butt")
        axE.text(xb, yb + 0.5, "1 frame", color="0.25", ha="left", va="bottom", fontsize=12)
        if not synced:
            axE.text(0.96 * half, yb - 0.4, "no synchrony", color=PLUM, ha="right", va="bottom",
                     fontsize=14, style="italic")
        step = 10.0 ** np.floor(np.log10(0.8 * half))
        tick = max(c * step for c in (1, 2, 5) if c * step <= 0.8 * half)
        axE.set_xticks([-tick, 0, tick])
        axE.set_xticklabels([f"\u2212{tick:g}", "0", f"{tick:g}"], fontsize=12 if wfrac < 0.5 else 14)
        axE.set_xlim(-half, half); axE.set_ylim(lo, hi); axE.set_yticks([1, N])
        axE.set_ylabel("Agent", labelpad=-6)
        axE.set_facecolor("0.95")
        for yl, yr in ((0, lo), (N + 1, hi)):  # zoom lines, drawn off the axis of the left panel
            fig.add_artist(ConnectionPatch(xyA=(tc + band, yl), coordsA=axD.transData,
                                           xyB=(-half, yr), coordsB=axE.transData,
                                           color="0.6", lw=0.9, ls=(0, (4, 3))))
        if r == n - 1:
            axD.set_xlabel("Time (s)")
            axE.set_xlabel("Shaded window, zoomed in (ms)")
    fig.savefig(out_png)
    plt.close(fig)


def plot_report(rows, out_png, f_eye=40.0, N=20, gain=0.15, seed=0):
    """Figure for the report. (A) One simulated group whose sensor has the temporal
    resolution of the firefly eye (f_eye), with one group flash zoomed in. (B) Flash-time
    spread against sensor temporal resolution for the group sizes in `rows`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    AMBER, PLUM, ROSE = "#C8A24A", "#6B3F6B", "#E8C9CF"
    fig = plt.figure(figsize=(5.6, 4.3), dpi=300)
    axD = fig.add_axes([0.095, 0.72, 0.45, 0.215])   # A, the group
    axE = fig.add_axes([0.67, 0.70, 0.305, 0.255])   # A, one group flash zoomed in
    axS = fig.add_axes([0.095, 0.115, 0.88, 0.405])   # B, the sweep

    # ---- A: same runs as the sweep (20 runs of 60 s); draw the most typical run
    frame = 1e3 / f_eye
    prc = make_prc("restoring", gain=gain)
    fl, _ = simulate(prc, N=N, f_sensor=f_eye, duration=60.0, trials=20, seed=seed)
    met = sync_metrics(fl, N, 20, 60.0)
    spread = np.nanmedian(met[:, 0])
    tr = int(np.nanargmin(np.abs(met[:, 0] - spread)))
    fl = fl[fl[:, 1] == tr]
    fl = np.column_stack([fl[:, 0] - 55.0, fl[:, 2]])  # 3 s late in the run
    m = (fl[:, 0] >= 0) & (fl[:, 0] < 3.0)
    axD.scatter(fl[m, 0], fl[m, 1] + 1, s=4, color=AMBER, lw=0)
    late = fl[(fl[:, 0] > 1.7) & (fl[:, 0] < 2.3)]
    tc = np.median(late[:, 0])
    half = 1.1 * frame
    band = 0.035
    axD.axvspan(tc - band, tc + band, color="0.85", lw=0, zorder=0)
    axD.set_xlim(0, 3); axD.set_xticks([0, 1, 2, 3]); axD.set_ylim(0, N + 1); axD.set_yticks([1, N])
    axD.set_xlabel("Time (s)", labelpad=2); axD.set_ylabel("Agent", labelpad=-4)
    lo, hi = -3.5, N + 3.0
    win = fl[1e3 * np.abs(fl[:, 0] - tc) <= half]
    x, y = 1e3 * (win[:, 0] - tc), win[:, 1] + 1
    axE.scatter(x, y, s=11, color=AMBER, lw=0, zorder=3)
    d = np.abs(x[:, None] - x[None, :]) + 1e9 * (y[:, None] == y[None, :])
    i, j = np.unravel_index(np.argmin(np.abs(d - spread)), d.shape)  # a pair one spread apart
    y_ar = N + 1.6
    for k in (i, j):
        axE.scatter(x[k], y[k], s=16, color=PLUM, lw=0, zorder=4)
        axE.plot([x[k], x[k]], [y[k], y_ar], color=PLUM, lw=0.7, ls=(0, (3, 2)))
    axE.annotate("", xy=(x[j], y_ar), xytext=(x[i], y_ar),
                 arrowprops=dict(arrowstyle="<->", color=PLUM, lw=1.0, shrinkA=0, shrinkB=0))
    xb, yb = -0.5 * frame, lo + 1.2  # scale bar: one sampling period of the sensor
    axE.plot([xb, xb + frame], [yb, yb], color="0.25", lw=2, solid_capstyle="butt")
    axE.set_xlim(-half, half); axE.set_ylim(lo, hi); axE.set_yticks([1, N])
    axE.set_xticks([-20, 0, 20]); axE.set_xticklabels(["−20", "0", "20"])
    axE.set_xlabel("Time (ms)", labelpad=2); axE.set_ylabel("Agent", labelpad=-4)
    axE.set_facecolor("0.95")
    for yl, yr in ((0, lo), (N + 1, hi)):
        fig.add_artist(ConnectionPatch(xyA=(tc + band, yl), coordsA=axD.transData,
                                       xyB=(-half, yr), coordsB=axE.transData,
                                       color="0.6", lw=0.6, ls=(0, (4, 3))))

    # ---- B: the sweep
    axS.axvspan(*FIREFLY_BAND, color=ROSE, lw=0, zorder=0)          # firefly eye (ERG)
    axS.axhline(FLASH_MS, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=0)   # duration of one flash
    axS.axhline(1e3 * T_NATURAL / 4, color="0.75", lw=0.9, zorder=0)  # expected with no synchrony
    for n_ in sorted({r["N"] for r in rows}):
        rr = sorted((r for r in rows if r["N"] == n_), key=lambda r: r["f_sensor"])
        xs = np.array([r["f_sensor"] for r in rr])
        axS.fill_between(xs, [r["spread_lo"] for r in rr], [r["spread_hi"] for r in rr],
                         color=COLORS.get(n_, "0.4"), alpha=0.18, lw=0)
        axS.plot(xs, [r["spread_ms"] for r in rr], "-o", color=COLORS.get(n_, "0.4"),
                 ms=3, lw=1.4, label=f"N = {n_}")
    axS.set_xscale("log"); axS.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    axS.xaxis.set_major_formatter(plain); axS.yaxis.set_major_formatter(plain)
    axS.set_xlabel("Temporal resolution of the sensor (Hz)", labelpad=2)
    axS.set_ylabel("Flash-time spread (ms)")
    axS.legend(frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.2)
    sec = axS.secondary_xaxis("top", functions=(lambda f: 1e3 / np.maximum(f, 1e-9),
                                                 lambda d: 1e3 / np.maximum(d, 1e-9)))
    sec.set_xticks([500, 100, 10, 1]); sec.xaxis.set_major_formatter(plain)
    sec.set_xlabel("Sampling period (ms)", labelpad=4)
    for lab, ypos in (("A", 0.965), ("B", 0.59)):
        fig.text(0.008, ypos, lab, fontsize=12, fontweight="bold", ha="left", va="center")
    fig.savefig(out_png)
    plt.close(fig)
    return spread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prc", default="restoring", choices=["restoring", "measured"])
    ap.add_argument("--led-ms", type=int, default=700, choices=sorted(TABLE_S3), help="Table S3 row for --prc measured")
    ap.add_argument("--gain", type=float, default=0.15, help="gain of the restoring rule")
    ap.add_argument("--freqs", type=float, nargs="+", default=DEFAULT_FREQS, help="sensor temporal resolutions (Hz)")
    ap.add_argument("--n", type=int, nargs="+", default=DEFAULT_NS, help="group sizes")
    ap.add_argument("--k", type=int, default=None, help="visible neighbors per agent (default: all)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds simulated per run")
    ap.add_argument("--period-cv", type=float, default=0.05, help="spread of natural periods across agents")
    ap.add_argument("--phase-spread", type=float, default=1.0, help="1 = random start, 0.01 = start synchronized")
    ap.add_argument("--no-self-mask", action="store_true")
    ap.add_argument("--no-latency-comp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single", type=float, default=None, help="run one temporal resolution (Hz) and print metrics")
    ap.add_argument("--out", default=None, help="output prefix (default: sweep_<prc>)")
    ap.add_argument("--from-csv", default=None, help="redraw the figure from a saved sweep CSV")
    ap.add_argument("--rates", type=float, nargs="+", default=None,
                    help="draw the group and its flash-time spread at these sensor rates (Hz), e.g. --rates 1000 40 3")
    ap.add_argument("--schematic", action="store_true", help="draw the model schematic and the compact explainer figures")
    ap.add_argument("--report", action="store_true",
                    help="run the default sweep and draw the two-panel report figure (figure_sensor_resolution.png)")
    a = ap.parse_args()
    if a.report:
        if a.from_csv:  # redraw from a saved sweep
            with open(a.from_csv) as fh:
                rows = [{k: (v if k == "prc" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
            for r in rows:
                r["N"] = int(r["N"])
        else:
            rows = sweep("restoring", DEFAULT_FREQS, DEFAULT_NS, "sweep_restoring", gain=a.gain, seed=a.seed)
        plot_report(rows, "figure_sensor_resolution.png", gain=a.gain, seed=a.seed)
        print("wrote figure_sensor_resolution.png")
        return
    if a.rates:
        plot_rates("spread_by_sensor_rate.png", rates=a.rates, gain=a.gain)
        print("wrote spread_by_sensor_rate.png")
        return
    if a.schematic:
        plot_model("model_schematic.png", gain=a.gain)
        plot_explainer("explainer_3panel.png", with_sensor=True, gain=a.gain)
        plot_explainer("explainer_2panel.png", with_sensor=False, gain=a.gain)
        print("wrote model_schematic.png, explainer_3panel.png, explainer_2panel.png")
        return
    if a.from_csv:
        with open(a.from_csv) as fh:
            rows = [{k: (v if k == "prc" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
        for r in rows:
            r["N"] = int(r["N"])
        plot(rows, a.from_csv.replace(".csv", ".png"))
        return
    kw = dict(duration=a.duration, trials=a.trials, seed=a.seed, period_cv=a.period_cv,
              self_mask=not a.no_self_mask, latency_comp=not a.no_latency_comp, k=a.k,
              phase_spread=a.phase_spread)
    if a.single is not None:
        prc = make_prc(a.prc, a.led_ms, a.gain)
        for N in a.n:
            print(run_condition(prc, N, a.single, **kw))
        return
    out = a.out or f"sweep_{a.prc}"
    rows = sweep(a.prc, a.freqs, a.n, out, led_ms=a.led_ms, gain=a.gain, **kw)
    plot(rows, out + ".png")
    print("wrote", out + ".csv", "and", out + ".png")


if __name__ == "__main__":
    main()
