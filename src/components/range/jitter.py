import numpy as np

def jitter(error, type):
    rng = np.random.default_rng()
    match type:
        #sideband+
        case 1:
            bounds = [1.4, 1.85]
            jitter = rng.uniform(low = -0.03, high = 0.03)
            val = error + jitter
            if val >= bounds[0] and val <= bounds[1]:
                return val
            else: return error - jitter
        #sideband-
        case 2:
            bounds = [-2.25, -1.55]
            jitter = rng.uniform(low = -0.06, high = 0.06)
            val = error + jitter
            if val >= bounds[0] and val <= bounds[1]:
                return val
            else: return error - jitter
        #far+
        case 3:
            bounds = [1.85]
            jitter = rng.uniform(low = -0.1, high = 0.1)
            val = error + jitter
            if val >= bounds[0]:
                return val
            else: return error - jitter
        #far-
        case 4:
            bounds = [-2.25]
            jitter = rng.uniform(low = -0.15, high = 0.15)
            val = error + jitter
            if val <= bounds[0]:
                return val
            else: return error - jitter
        #near
        case 5:
            upper = [0.82, 1.4]
            lower = [-1.55, -0.52]
            jitter = rng.uniform(low = -0.03, high = 0.03)
            val = error + jitter

            if error >= upper[0]:
                if val <= 1.4:
                    return val
                else: return error - jitter
            else:
                if val >= -1.55:
                    return val
                else: return error - jitter

# JITTER FITTING TBD!!!