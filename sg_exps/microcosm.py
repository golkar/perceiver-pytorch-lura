import sys
import numpy as np

sys.path.append("../../chaospy/")
from src.dynamic_system import DynamicSystem

max_sys = 5
default_len = 500
max_dim = 40
min_dim = 3
max_hierarchies = 3
scales = [10, 100]  # other than 1
assert (
    max_hierarchies <= len(scales) + 1
), "maximum number of hierarchies has to be smaller than the number of scales provided."

step_multips = [1, 2, 4, 8, 16, 32]
sys_names = ["lorenz"]

num_chaotic_sys = len(sys_names)
num_step_multips = len(step_multips)
num_scales = len(scales)

num_tries = 5
lorenz_bound = 55


def check_series(series):
    if (series**2).sum(1).max() > lorenz_bound**2:
        return False

    return True


def generate_lorenz(num_steps, step_size, init_point=None, params=None, rng=None):
    if rng == None:
        rng = np.random.default_rng()

    if params == None:
        params = np.abs([8, 2.66, 25] * (1 + 0.4 * rng.normal(size=3)))

    if init_point == None:
        init_point = 3 * rng.normal(size=3)

    command_line = (
        "--init_point",
        " ".join(str(x) for x in init_point),
        "--points",
        str(num_steps),
        "--step",
        str(step_size),
        "lorenz",
        "--sigma",
        str(params[0]),
        "--beta",
        str(params[1]),
        "--rho",
        str(params[2]),
    )

    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()

    series = chaotic_system.model.get_coordinates()

    if check_series(series):
        return (
            series,
            {"params": params, "init_point": init_point, "step_size": step_size},
            True,
        )
    else:
        return ([], [params, init_point], False)


def sample_world(
    num_sys=None,
    num_steps=default_len,
    verbose=False,
    normalize=True,
    rotate=True,
    dilate=True,
    rotate_again=True,
    change_dim=True,
    new_dim=None,
    seed=1337,
):
    rng = np.random.default_rng(seed)

    if num_sys == None:
        num_sys = rng.integers(1, 6)
    sys_number = rng.integers(0, num_chaotic_sys, num_sys)
    sys_multip = rng.integers(0, num_step_multips, num_sys)

    series = []
    series_param_dict = {}

    for j, (num, multip_num) in enumerate(zip(sys_number, sys_multip)):
        success = False
        sys_name = sys_names[num]
        multip = step_multips[multip_num]

        if sys_name == "lorenz":
            for i in range(num_tries):
                sub_series, param_dict, success = generate_lorenz(
                    num_steps=multip * num_steps, step_size=700, rng=rng
                )
                if success:
                    series.append(sub_series[::multip])
                    param_dict["name"] = "lorenz"
                    param_dict["step_multip"] = multip
                    series_param_dict[j] = param_dict
                    break
                else:
                    if verbose:
                        print("Bound fail! (attempt {})".format(i))
            if not success:
                raise Exception("Generation failed after {} tries.".format(i))

    series = np.concatenate(series, 1)

    if normalize:
        m, s = series.mean(0), series.std(0)
        series_param_dict["normalization"] = [m, s]
        series = (series - m) / s

    old_dim = series.shape[1]

    if rotate:
        if change_dim:
            if new_dim == None:
                new_dim = rng.integers(min_dim, max_dim)
        else:
            if new_dim == None:
                new_dim = old_dim
            else:
                raise ValueError("new_dim is specified but change_dim=False!")

        W = rng.normal(size=[old_dim, new_dim])

        l, L, r = np.linalg.svd(W, full_matrices=False)

    else:
        assert not change_dim, "Cannot have change_dim with rotate=False."
        new_dim = old_dim
        l = 1
        r = 1

    if not rotate_again:
        r = 1

    if dilate:
        hierarchies = rng.integers(1, max_hierarchies)
        sys_scales = np.sort([1] + list(rng.choice(scales, hierarchies - 1)))

        L_dim = min(old_dim, new_dim)
        n = L_dim // hierarchies
        L = [x for el in sys_scales for x in [el] * n]
        L = [1] * (L_dim - len(L)) + L
        L = np.diag(L)

    else:
        L = 1
        r = 1

    series = multiply(matrices=[series, l, L, r])

    series_param_dict["embedding"] = [l, L, r]

    return series, series_param_dict


def multiply(matrices):
    out = matrices[0]

    for mat in matrices[1:]:
        if type(mat) == int:
            out = out * mat
        else:
            out = out @ mat

    return out
