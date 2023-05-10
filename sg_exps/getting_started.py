# %%

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from perceiver_pytorch import Perceiver
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("../../chaospy/")
from src.dynamic_system import DynamicSystem

loss_func = nn.MSELoss(reduction="sum")

# %%

# model = Perceiver(
#     input_channels=3,  # number of channels for each token of the input
#     input_axis=1,  # number of axis for input data (2 for images, 3 for video)
#     num_freq_bands=11,  # number of freq bands, with original value (2 * K + 1)
#     max_freq=10.0,  # maximum frequency, hyperparameter depending on how fine the data is
#     depth=4,  # depth of net. The shape of the final attention mechanism will be:
#     #   depth * (cross attention -> self_per_cross_attn * self attention)
#     num_latents=123,  # number of latents, or induced set points, or centroids. different papers giving it different names
#     latent_dim=321,  # latent dimension
#     cross_heads=2,  # number of heads for cross attention. paper said 1
#     latent_heads=7,  # number of heads for latent self attention, 8
#     cross_dim_head=62,  # number of dimensions per cross attention head
#     latent_dim_head=63,  # number of dimensions per latent self attention head
#     num_classes=51,  # output number of classes
#     attn_dropout=0.0,
#     ff_dropout=0.0,
#     weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
#     fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
#     self_per_cross_attn=2,  # number of self attention blocks per cross attention
# )

# img = torch.randn(1, 224, 3)  # 1 imagenet image, pixelized

# model(img)  # (1, 1000)

# %%

# Choosing step size of 400 for the dataset

for fact in [4, 8, 16]:
    command_line = (
        "--init_point",
        "7 3 2",
        "--points",
        str(3 * fact),
        "--step",
        str(50 * fact),
        # "--save_plots",
        # "--show_plots",
        # "--add_2d_gif",
        "lorenz",
    )
    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()
    plt.plot(
        chaotic_system.model.get_coordinates()[:, 0],
        chaotic_system.model.get_coordinates()[:, 1],
        "-o",
        markersize=2,
        label="step: {}, num_points={}".format(50 * fact, 10 * fact),
    )
for fact in [2, 4, 8]:
    command_line = (
        "--init_point",
        "7 3 2",
        "--points",
        str(30 * fact),
        "--step",
        str(5 * fact),
        # "--save_plots",
        # "--show_plots",
        # "--add_2d_gif",
        "rossler",
    )
    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()
    plt.plot(
        chaotic_system.model.get_coordinates()[:, 0],
        chaotic_system.model.get_coordinates()[:, 1],
        "-o",
        markersize=2,
        label="step: {}, num_points={}".format(50 * fact, 10 * fact),
    )
plt.legend()
# %%

# Collecting the set of parameters for training

num_params = 500

Lorenz_params = [10, 2.66, 28] * (1 + 0.4 * np.random.randn(num_params, 3))
plt.scatter(Lorenz_params[:, 0], Lorenz_params[:, 1])
plt.show()
plt.scatter(Lorenz_params[:, 1], Lorenz_params[:, 2])

start_points = 3 * np.random.randn(num_params, 3)

# %%

# Showing performance for a few of the elements

for start, param in zip(start_points[:3], Lorenz_params[:3]):
    start = " ".join([str(a) for a in start])
    command_line = (
        "--init_point",
        start,
        "--points",
        "2000",
        "--step",
        "400",
        "lorenz",
        "--sigma",
        str(param[0]),
        "--beta",
        str(param[1]),
        "--rho",
        str(param[2]),
    )
    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()
    plt.plot(
        chaotic_system.model.get_coordinates()[:, 0],
        chaotic_system.model.get_coordinates()[:, 1],
        "-",
        # markersize=2,
        label="sigma: {:.2f}, beta={:.2f}, rho={:.2f}".format(*param),
    )
    plt.legend()

    command_line = (
        "--init_point",
        start,
        "--points",
        "1000",
        "--step",
        "4000",
        "lorenz",
        "--sigma",
        str(param[0]),
        "--beta",
        str(param[1]),
        "--rho",
        str(param[2]),
    )
    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()
    plt.plot(
        chaotic_system.model.get_coordinates()[:, 0],
        chaotic_system.model.get_coordinates()[:, 1],
    )
    plt.show()
# %%

# collecting the data (and adding the time-step subsamples)
# cutting off large coordinate magnitudes
steps = [2, 5, 10, 15, 30, 60]
steps_max = np.max(steps)

data = []
for start, param in zip(start_points, Lorenz_params):
    start = " ".join([str(a) for a in start])
    command_line = (
        "--init_point",
        start,
        "--points",
        "3000",
        "--step",
        "400",
        "lorenz",
        "--sigma",
        str(param[0]),
        "--beta",
        str(param[1]),
        "--rho",
        str(param[2]),
    )
    chaotic_system = DynamicSystem(input_args=command_line, show_log=False)
    chaotic_system.run()

    if (chaotic_system.model.get_coordinates() ** 2).sum(1).max() < 1600:
        data.append(
            {
                "start": start,
                "params": param,
                "coords": chaotic_system.model.get_coordinates(),
                "step": 1,
            }
        )

        for step in steps:
            data.append(
                {
                    "start": start,
                    "params": param,
                    "coords": chaotic_system.model.get_coordinates()[::step],
                    "step": step,
                }
            )


# %%
# Adding in the block size train data

block_size = 4
full_input_blocks = []
full_target_blocks = []
for run in data:
    input_block = []
    target_block = []

    for i in range(len(run["coords"]) - block_size - 1):
        input_block.append(run["coords"][i : i + block_size + 1])

    input_block_len = len(input_block)
    samp_ind = np.random.permutation(input_block_len)[
        : int(input_block_len * run["step"] / steps_max)
    ]
    run["input_block"] = np.array(input_block)[samp_ind]

    full_input_blocks.append(run["input_block"])

full_input_blocks = torch.tensor(
    np.concatenate(full_input_blocks, 0), dtype=torch.float
)

dataset_len = len(full_input_blocks)
train_len = int(0.7 * dataset_len)
full_input_blocks_train = full_input_blocks[:train_len]
full_input_blocks_test = full_input_blocks[train_len:]

plt.hist((full_input_blocks[:, -1] ** 2).sum(-1).flatten().numpy(), bins=200)
plt.show()
# %%

# simple linear extrapolation baseline:

y_hat = 2 * full_input_blocks_test[:, -2] - full_input_blocks_test[:, -3]
y = full_input_blocks_test[:, -1]
loss = loss_func(y_hat, y) / len(y)

print("(baseline) test mse loss {:.3g}".format(loss.item()))
split1 = 0.1
split2 = 10

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
plt.hist(error, range=(0, split1), bins=100)
plt.title("test square error histogram - early (baseline)")
plt.show()

plt.hist(error[error > split1], range=(split1, split2), bins=100)
plt.title("test square error histogram - mid  (baseline)")
plt.show()

plt.hist(error[error > split2], bins=100)
plt.title("test square error histogram - late (baseline)")
plt.show()

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_test[:, -1] - full_input_blocks_test[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)
error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]
print("(baseline) relative test mse loss {:.3g}".format((error / step_length).mean()))

plt.hist(error / step_length, range=(0, 0.4), bins=100)
plt.title("Relative test square error histogram - early (baseline)")
plt.show()

plt.hist((error / step_length)[(error / step_length) > 0.4], bins=100)
plt.title("Relative test square error histogram - late (baseline)")
plt.show()

# %%
# Setting up an MLP to learn the full task

hidden_size = 1024
batch_size = 256

train_loader = DataLoader(
    full_input_blocks_train, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(full_input_blocks_test, batch_size=batch_size, shuffle=True)


net = nn.Sequential(
    nn.Linear(in_features=block_size * 3, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=3),
).cuda()

opt = optim.Adam(net.parameters(), lr=0.0002)
# %%

epochs = 40

loss_hist = []
for epoch in range(epochs):
    for batch in train_loader:
        batch_len = len(batch)

        x = batch[:, :-1].reshape(batch_size, block_size * 3).cuda()
        y = batch[:, -1].cuda()

        loss = loss_func(net(x), y) / batch_len

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_hist.append(loss.item())

plt.semilogy(loss_hist)
plt.show()

# %%
x = full_input_blocks_test[:, :-1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_test[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)
print("(trained mlp) test mse loss {:.3g}".format(loss.item()))

plt.hist(
    ((y_hat - y) ** 2).sum(1).data.cpu().numpy(),
    range=(0, 0.12),
    bins=100,
    label="test",
    alpha=1,
    density=True,
)

x = full_input_blocks_train[:, :-1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_train[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)

print("(trained mlp) train mse loss {:.3g}".format(loss.item()))
plt.hist(
    ((y_hat - y) ** 2).sum(1).data.cpu().numpy(),
    range=(0, 0.12),
    bins=100,
    label="train",
    alpha=0.7,
    density=True,
)
plt.legend()
plt.title("square error histogram (trained mlp)")
# %%

x = full_input_blocks_test[:, :-1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_test[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_test[:, -1] - full_input_blocks_test[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)
error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]

plt.hist(
    error / step_length,
    range=(0, 1),
    bins=100,
    label="test",
    alpha=1,
    density=True,
)

x = full_input_blocks_train[:, :-1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_train[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)


error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_train[:, -1] - full_input_blocks_train[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)

error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]

plt.hist(
    error / step_length,
    range=(0, 1),
    bins=100,
    label="train",
    alpha=0.7,
    density=True,
)
plt.legend()
plt.title("Relative square error histogram (trained mlp)")
print(
    "(trained mlp) relative test mse loss {:.3g}".format((error / step_length).mean())
)
# %%
##reducing block size to 3
#

# Instead of changing the block size, just use the lsat 3 elements of the input


block_size = 2
# full_input_blocks = []
# full_target_blocks = []
# for run in data:
#     input_block = []
#     target_block = []

#     for i in range(len(run["coords"]) - block_size - 1):
#         input_block.append(run["coords"][i : i + block_size + 1])

#     input_block_len = len(input_block)
#     samp_ind = np.random.permutation(input_block_len)[
#         : int(input_block_len * run["step"] / steps_max)
#     ]
#     run["input_block"] = np.array(input_block)[samp_ind]

#     full_input_blocks.append(run["input_block"])

# full_input_blocks = torch.tensor(
#     np.concatenate(full_input_blocks, 0), dtype=torch.float
# )

# dataset_len = len(full_input_blocks)
# train_len = int(0.7 * dataset_len)
# full_input_blocks_train = full_input_blocks[:train_len]
# full_input_blocks_test = full_input_blocks[train_len:]

# plt.hist((full_input_blocks[:, -1] ** 2).sum(-1).flatten().numpy(), bins=200)
# plt.show()

# %%

# simple linear extrapolation baseline:

y_hat = 2 * full_input_blocks_test[:, -2] - full_input_blocks_test[:, -3]
y = full_input_blocks_test[:, -1]
loss = loss_func(y_hat, y) / len(y)

print("(baseline) test mse loss {:.3g}".format(loss.item()))
split1 = 0.1
split2 = 10

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
plt.hist(error, range=(0, split1), bins=100)
plt.title("test square error histogram - early (baseline)")
plt.show()

plt.hist(error[error > split1], range=(split1, split2), bins=100)
plt.title("test square error histogram - mid  (baseline)")
plt.show()

plt.hist(error[error > split2], bins=100)
plt.title("test square error histogram - late (baseline)")
plt.show()

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_test[:, -1] - full_input_blocks_test[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)
error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]
print("(baseline) relative test mse loss {:.3g}".format((error / step_length).mean()))

plt.hist(error / step_length, range=(0, 0.4), bins=100)
plt.title("Relative test square error histogram - early (baseline)")
plt.show()

plt.hist((error / step_length)[(error / step_length) > 0.4], bins=100)
plt.title("Relative test square error histogram - late (baseline)")
plt.show()

# %%
# Setting up an MLP to learn the full task

hidden_size = 1024
batch_size = 256

loss_func = nn.MSELoss(reduction="sum")


train_loader = DataLoader(
    full_input_blocks_train, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(full_input_blocks_test, batch_size=batch_size, shuffle=True)


net = nn.Sequential(
    nn.Linear(in_features=block_size * 3, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=3),
).cuda()

opt = optim.Adam(net.parameters(), lr=0.0005)


# %%

epochs = 40

loss_hist = []
for epoch in range(epochs):
    for batch in train_loader:
        batch_len = len(batch)

        x = batch[:, -block_size - 1 : -1].reshape(batch_size, block_size * 3).cuda()
        y = batch[:, -1].cuda()

        loss = loss_func(net(x), y) / batch_len

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_hist.append(loss.item())

plt.semilogy(loss_hist)
plt.show()

# %%
x = full_input_blocks_test[:, -block_size - 1 : -1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_test[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)

print("(trained mlp) test mse loss {:.3g}".format(loss.item()))
plt.hist(
    ((y_hat - y) ** 2).sum(1).data.cpu().numpy(),
    range=(0, 0.22),
    bins=100,
    label="test",
    alpha=1,
    density=True,
)

x = full_input_blocks_train[:, -block_size - 1 : -1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_train[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)

print("(trained mlp) train mse loss {:.3g}".format(loss.item()))
plt.hist(
    ((y_hat - y) ** 2).sum(1).data.cpu().numpy(),
    range=(0, 0.22),
    bins=100,
    label="train",
    alpha=0.7,
    density=True,
)
plt.legend()
plt.title("square error histogram (trained mlp)")
# %%

x = full_input_blocks_test[:, -block_size - 1 : -1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_test[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)

error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_test[:, -1] - full_input_blocks_test[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)
error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]

plt.hist(
    error / step_length,
    range=(0, 1),
    bins=100,
    label="test",
    alpha=1,
    density=True,
)

x = full_input_blocks_train[:, -block_size - 1 : -1].reshape(-1, block_size * 3)
y_hat = net(x.cuda())
y = full_input_blocks_train[:, -1].cuda()
loss = loss_func(y_hat, y) / len(y)


error = ((y_hat - y) ** 2).sum(1).data.cpu().numpy()
step_length = (
    ((full_input_blocks_train[:, -1] - full_input_blocks_train[:, -2]) ** 2)
    .sum(1)
    .data.cpu()
    .numpy()
)

error = error[step_length > 0.2]
step_length = step_length[step_length > 0.2]

plt.hist(
    error / step_length,
    range=(0, 1),
    bins=100,
    label="train",
    alpha=0.7,
    density=True,
)
plt.legend()
plt.title("Relative square error histogram (trained mlp)")
print(
    "(trained mlp) relative test mse loss {:.3g}".format((error / step_length).mean())
)
# %%
