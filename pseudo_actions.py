"""Camera-to-world poses -> known action descriptions, Appendix B.3.

NPZ fields: rotations [T,3,3], positions [T,3], times [T]. Coordinate axes must
match the dataset; the paper does not specify their signs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_actions(rotations, positions, times, forward_axis="-z", yaw_sign=1, pitch_sign=1):
    rotations, positions, times = map(np.asarray, (rotations, positions, times))
    if rotations.shape != (len(times), 3, 3) or positions.shape != (len(times), 3) or len(times) < 2:
        raise ValueError("invalid pose shapes")
    if not all(np.isfinite(a).all() for a in (rotations, positions, times)) or not (np.diff(times) > 0).all():
        raise ValueError("finite poses and increasing times required")
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("rotations must lie in SO(3)")
    if forward_axis not in ("z", "-z") or yaw_sign not in (-1, 1) or pitch_sign not in (-1, 1):
        raise ValueError("forward-axis must be z/-z; signs must be +1/-1")
    dt = np.diff(times)
    velocity = np.einsum("tji,tj->ti", rotations[:-1], np.diff(positions, axis=0)) / dt[:, None]
    relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    omega = Rotation.from_matrix(relative).as_rotvec() / dt[:, None]
    speeds = np.linalg.norm(velocity, axis=1)
    median = np.median(speeds[speeds > 0]) if (speeds > 0).any() else 1.
    velocity /= median
    threshold = np.deg2rad(2)
    result = []
    for i, (v, w) in enumerate(zip(velocity, omega)):
        labels = []
        forward = v[2] * (-1 if forward_axis == "-z" else 1)
        if forward > .2:
            labels.append("move forward")
        elif forward < -.2:
            labels.append("move backward")
        if v[0] > .2:
            labels.append("move right")
        elif v[0] < -.2:
            labels.append("move left")
        if w[1] * yaw_sign > threshold:
            labels.append("turn left")
        elif w[1] * yaw_sign < -threshold:
            labels.append("turn right")
        if w[0] * pitch_sign > threshold:
            labels.append("look up")
        elif w[0] * pitch_sign < -threshold:
            labels.append("look down")
        if not labels and (abs(v[1]) > .2 or abs(w[2]) > threshold):
            raise ValueError("unmapped vertical translation or roll; cannot label this interval stationary")
        result.append({"start": float(times[i]), "end": float(times[i + 1]),
                       "labels": labels or ["stationary"], "prompt": " and ".join(labels) or "remain stationary"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses")
    parser.add_argument("output")
    parser.add_argument("--forward-axis", choices=["z", "-z"], default="-z")
    parser.add_argument("--yaw-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--pitch-sign", type=int, choices=[-1, 1], default=1)
    args = parser.parse_args()
    with np.load(args.poses, allow_pickle=False) as data:
        result = pose_actions(data["rotations"], data["positions"], data["times"],
                              args.forward_axis, args.yaw_sign, args.pitch_sign)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
