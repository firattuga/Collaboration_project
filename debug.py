# plot_filtered_paths.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from track_filter import RecoTrackFilter, HitT, KG_PER_GEVC2_PER_E

# ---- Debug helper: compare 3 omega estimators and m/|q| ----
def debug_fit_mq(rf: RecoTrackFilter, path: List[HitT], B: float = 2.0):
    fit = rf._fit_circle(path)
    if fit is None:
        print("    [debug] circle fit failed")
        return

    xc, yc, _ = fit
    xs = np.array([h[1] for h in path], float)
    ys = np.array([h[2] for h in path], float)
    tns = np.array([h[4] for h in path], float)  # ns (ascending)

    phi = np.unwrap(np.arctan2(ys - yc, xs - xc))

    # 1) total slope
    dt_all   = tns[-1] - tns[0]
    dphi_all = phi[-1] - phi[0]
    w_total  = float(dphi_all / dt_all) if dt_all != 0 else float("nan")

    # 2) linear fit
    A = np.vstack([tns, np.ones_like(tns)]).T
    w_lin, _ = np.linalg.lstsq(A, phi, rcond=None)[0]

    # 3) median of incremental slopes
    dphi = np.diff(phi)
    dt   = np.diff(tns)
    wi   = dphi / dt
    w_med = float(np.median(wi))

    def to_mq_abs(w_ns: float) -> float:
        w_si = w_ns * 1e9
        mq_SI = B / w_si
        return abs(mq_SI / KG_PER_GEVC2_PER_E)

    print(f"    [debug] dt_all = {dt_all:.6g} ns, dphi_all = {dphi_all:.6g} rad")
    print(f"    [debug] omega_total  = {w_total:.6f} rad/ns  ->  m/|q| = {to_mq_abs(w_total):.6f} GeV/c^2/e")
    print(f"    [debug] omega_linfit = {w_lin:.6f} rad/ns  ->  m/|q| = {to_mq_abs(w_lin):.6f} GeV/c^2/e")
    print(f"    [debug] omega_median = {w_med:.6f} rad/ns  ->  m/|q| = {to_mq_abs(w_med):.6f} GeV/c^2/e")


# ---- Plot one event: filtered paths + fitted circles ----
def plot_event_filtered_paths(csv_path: str,
                              event_id: int,
                              B: float = 2.0,
                              dz=0.1, dz_tol=1e-6, dt_tol=1e-3,
                              omega_method="linfit",
                              do_debug=True):
    rf = RecoTrackFilter(csv_path)
    hits = rf.events.get(event_id)
    if not hits:
        raise SystemExit(f"[!] Event {event_id} not found in {csv_path}")

    # Generate all paths → filter (enforces ascending z,t)
    paths_all = rf.gen_paths(hits)
    paths = rf.f_dzdt(paths_all, dz=dz, dz_tol=dz_tol, dt_tol=dt_tol)
    if not paths:
        raise SystemExit(f"[!] No valid paths after filtering (event={event_id}).")

    print(f"[i] Paths: {len(paths_all)} total → {len(paths)} kept after filter")

    # Base scatter: all hits
    plt.figure(figsize=(7, 7))
    all_x = [h[1] for h in hits]
    all_y = [h[2] for h in hits]
    plt.scatter(all_x, all_y, s=20, alpha=0.25, label="all hits", zorder=1)

    # For each kept path: plot hits + polyline + fitted circle + center
    for idx, p in enumerate(paths, 1):
        xs = np.array([h[1] for h in p], float)
        ys = np.array([h[2] for h in p], float)

        plt.scatter(xs, ys, s=30, zorder=3)
        plt.plot(xs, ys, linewidth=1.0, alpha=0.9, zorder=2, label=f"path {idx}")

        fit = rf._fit_circle(p)
        if fit is None:
            plt.scatter([xs.mean()], [ys.mean()], marker="x", s=70, color="red",
                        zorder=5, label=f"fit fail {idx}")
            print(f"  - path {idx}: circle fit FAILED")
            continue

        xc, yc, R = fit
        th = np.linspace(0, 2*np.pi, 400)
        cx, cy = xc + R*np.cos(th), yc + R*np.sin(th)
        plt.plot(cx, cy, linewidth=1.3, zorder=2, label=f"circle {idx} (R={R:.3g})")
        plt.scatter([xc], [yc], marker="x", s=60, zorder=4)

        # Compute m/q and print
        res = rf.fit_mq(p, B=B, method=omega_method)
        if res is None:
            print(f"  - path {idx}: fit_mq FAILED")
        else:
            print(f"  - path {idx}: R={res['R']:.3f}, "
                  f"w={res['w_ns']:.5f} rad/ns, "
                  f"m/q={res['mq_GeV']:.6e} GeV/c^2/e, "
                  f"|m/q|={abs(res['mq_GeV']):.6e}, "
                  f"sign={res['q_sign']}")
            if do_debug:
                debug_fit_mq(rf, p, B=B)

    plt.axis("equal")
    plt.title(f"Event {event_id}: Filtered paths with fitted circles (B = {B} T)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ---- Run directly ----
if __name__ == "__main__":
    csv_path = "hits.csv"   # adjust to your CSV
    event_id = 0            # choose your event
    plot_event_filtered_paths(csv_path, event_id, B=2.0,
                              omega_method="linfit",
                              do_debug=True)
