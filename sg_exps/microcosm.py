sys.path.append("../../chaospy/")
from src.dynamic_system import DynamicSystem
import numpy as np

lorenz_bound = 40
max_systems = 5
chaotic_systems = {'lorenz':{'params':['sigma,beta,rho']}}
system_names = list(chaotic_systems.keys())

num_chaotic_sys = len(chaotic_systems)


def check_series(series):
    if (series ** 2).sum(1).max() > lorenz_bound**2:
        return False

    return True

def generate_lorenz(init_point, len, step_size, params):
    command_line = (
        "--init_point",
        ' '.join(str(x) for x in init_point),
        "--points",
        str(len),
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
        return (series, True)
    else:
        return ([], False)

def sample_world():

    num_systems = np.random.randint(1,6)
    system_number = np.random.randint(0,num_chaotic_sys, num_systems)

    for 

