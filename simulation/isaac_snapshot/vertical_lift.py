import math

import numpy as np
import pinocchio as pin

from school_planner import CHECK_STEP, Planner, pose_matrix
from policy_rollout import PolicyStop


def checked_lift(config, urdf, evidence, fixtures, q, wrist, fingers, *, margin=0.002, height_m=0.030):
    """Check the fixed vertical verification motion against the robot and static scene."""
    geometry = Planner(config, urdf, evidence, None, obstacles=fixtures)
    requested = pose_matrix(wrist)
    predicted = geometry.world_frame(q)
    if np.linalg.norm(requested.translation - predicted.translation) > 0.002:
        raise PolicyStop("safety_stop", "measured wrist and lift model disagree")
    current = q
    points = []
    if not 0.0 < height_m <= 0.10:
        raise ValueError("Trial lift height must be within the supported guarded range")
    for height in np.linspace(height_m / 4, height_m, 4):
        goal = pin.SE3(requested.rotation, requested.translation + [0.0, 0.0, height])
        solved, error = geometry.ik(goal, current)
        if solved is None:
            raise PolicyStop("safety_stop", f"fixed vertical lift is unreachable: {error}")
        intervals = max(1, math.ceil(np.linalg.norm(solved - current, ord=1) / CHECK_STEP))
        for fraction in np.linspace(0, 1, intervals + 1):
            colliding = geometry.collision(current + fraction * (solved - current), fingers=fingers, margin=margin)
            if colliding:
                raise PolicyStop("safety_stop", f"fixed vertical lift violates clearance: {colliding}")
        points.append(solved)
        current = solved
    return points
