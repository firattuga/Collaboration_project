# time_gate_addon.py
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def _load_hits_by_event(csv_path: str):
    buckets = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = int(row["EventID"])
            buckets[ev].append(row)
    return buckets


def time_gate_filter_csv(
    input_csv: str,
    output_csv: str = "hits_timegated.csv",
    n_sensors: int = 5,
    dt_window_ns: float = 1.0,   # <-- main knob: tighter removes more noise
    min_hits_to_fit: int = 3,
):
    """
    Filters hits using an event-wise time-of-flight consistency gate.

    For each event:
      - take the earliest hit per sensor (to reduce noise influence)
      - fit t = a + b*z (b = 1/vz)
      - keep hits satisfying |t - (a + b*z)| <= dt_window_ns

    Writes a new CSV with same columns (so reconstruction code unchanged).
    """
    buckets = _load_hits_by_event(input_csv)

    # store stats for plots
    kept_counts = []
    total_counts = []

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    with open(output_csv, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for ev, rows in buckets.items():
            total_counts.append(len(rows))

            # group by sensor
            by_sensor = defaultdict(list)
            for r in rows:
                sid = int(r["SensorID"])
                by_sensor[sid].append(r)

            # pick earliest hit per sensor (robust seed)
            seed_pts = []
            for sid in range(n_sensors):
                if sid not in by_sensor:
                    continue
                # earliest time hit in that sensor for this event
                rmin = min(by_sensor[sid], key=lambda rr: float(rr["time_ns"]))
                z = float(rmin["z_nominal"])
                t = float(rmin["time_ns"])
                seed_pts.append((z, t))

            # if not enough points, keep everything (don’t destroy event)
            if len(seed_pts) < min_hits_to_fit:
                for r in rows:
                    writer.writerow(r)
                kept_counts.append(len(rows))
                continue

            zs = np.array([p[0] for p in seed_pts], float)
            ts = np.array([p[1] for p in seed_pts], float)

            # fit t = a + b*z
            b, a = np.polyfit(zs, ts, deg=1)

            kept = 0
            for r in rows:
                z = float(r["z_nominal"])
                t = float(r["time_ns"])
                t_pred = a + b * z
                if abs(t - t_pred) <= dt_window_ns:
                    writer.writerow(r)
                    kept += 1

            kept_counts.append(kept)

    print(f"[TIME-GATE] Wrote filtered CSV: {output_csv}")
    if len(total_counts) > 0:
        kept_frac = np.array(kept_counts) / np.array(total_counts)
        print(f"[TIME-GATE] Mean kept fraction: {np.mean(kept_frac):.3f}")

        # quick diagnostic plots (event-level)
        plt.figure()
        plt.hist(kept_frac, bins=30)
        plt.xlabel("kept fraction per event")
        plt.ylabel("count")
        plt.title(f"Time-gate filtering strength (dt_window={dt_window_ns} ns)")

        plt.figure()
        plt.scatter(range(len(total_counts)), total_counts, s=12, label="before")
        plt.scatter(range(len(kept_counts)), kept_counts, s=12, label="after")
        plt.xlabel("event")
        plt.ylabel("hits in event")
        plt.title("Event hit counts: before vs after time-gating")
        plt.legend()

    return output_csv


def compare_reco_quality(tracks_before: dict, tracks_after: dict):
    """
    Quick quality plots using fit['err'] distribution and track counts.
    """
    def errs(tracks):
        out = []
        for _, tr in tracks.items():
            fit = tr.get("fit")
            if fit and "err" in fit:
                out.append(float(fit["err"]))
        return np.array(out, float)

    e1 = errs(tracks_before)
    e2 = errs(tracks_after)

    plt.figure()
    plt.bar(["before", "after"], [len(tracks_before), len(tracks_after)])
    plt.ylabel("reconstructed track count")
    plt.title("Reconstruction yield: before vs after time-gating")

    if e1.size and e2.size:
        plt.figure()
        plt.hist(e1, bins=50, alpha=0.5, label="err before")
        plt.hist(e2, bins=50, alpha=0.5, label="err after")
        plt.xlabel("fit err")
        plt.ylabel("count")
        plt.title("Fit error distribution: before vs after time-gating")
        plt.legend()
