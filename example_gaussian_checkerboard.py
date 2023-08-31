from time import time
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import ot
import sklearn
import sklearn.datasets
import torch
import torch.nn as nn
from sklearn.utils import shuffle as util_shuffle
from torch import Tensor
from torch.distributions import Normal
from torchdiffeq import odeint
from tqdm import tqdm


# Dataset iterator
# Taken from https://github.com/rtqichen/residual-flows/blob/master/lib/toy_data.py
def inf_train_gen(data, batch_size=200):

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", batch_size)


# Model definition

class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])

        super().__init__(*layers[:-1])


class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        frequencies: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.net = MLP(2 * frequencies + features, features, **kwargs)

        self.register_buffer('frequencies', 2 ** torch.arange(frequencies) * torch.pi)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.frequencies * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)
        return self.net(torch.cat((t, x), dim=-1))

    # forward pass from data to Gaussian
    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    # backward pass from Gaussian to data
    def decode(self, z: Tensor, t=torch.tensor([1., 0.]), method='euler', options={}) -> Tensor:
        # return odeint(self, z, t, method="euler", options={"step_size": 1./nfe,})[-1]
        return odeint(self, z, t, method=method, options=options)[-1]

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        z = odeint(x, 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1)


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        t = torch.rand_like(x0[..., 0]).unsqueeze(-1)

        # rectified flow formulation
        xt = (1 - t) * x0 + t * x1
        u = x1 - x0

        return (self.v(t.squeeze(-1), xt) - u).square().mean()


# Training data
# we want to learn flow from x0 to x1
torch.manual_seed(1)
n_samples = 2000
data = torch.from_numpy(inf_train_gen('checkerboard', n_samples)).float()
# target = torch.from_numpy(inf_train_gen('2spirals', n_samples)).float()
target = torch.randn_like(data)

batch_size = 64
n_iters = n_samples // batch_size
n_epochs = 4000

### Training with all data
flow = CNF(2, hidden_features=[256] * 3)

# Training
loss = FlowMatchingLoss(flow)
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

temp = time()
losses = []
for epoch in tqdm(range(n_epochs), ncols=88):
    # we can make it batch training for replacing len(data) with something smaller
    optimizer.zero_grad()
    loss(data, target).backward()
    losses.append(loss(data, target).item())
    optimizer.step()

print(f'\nTotal training time: {time()- temp}');
plt.plot(losses);  # should have small variances

### Training with minibatch
flow_minibatch = CNF(2, hidden_features=[256] * 3)
loss = FlowMatchingLoss(flow_minibatch)
optimizer = torch.optim.AdamW(flow_minibatch.parameters(), lr=1e-3)

temp = time()
losses = []

for epoch in tqdm(range(n_epochs), ncols=88):
    subset = torch.randint(0, len(data), (batch_size * n_iters,))
    for i in range(n_iters):
        batch_data = data[subset[i * batch_size:(i+1)*batch_size]]
        batch_target = target[subset[i * batch_size:(i+1)*batch_size]]
        optimizer.zero_grad()
        loss(batch_data, batch_target).backward()
        losses.append(loss(batch_data, batch_target).item())
        optimizer.step()

print(f'\nTotal training time: {time()- temp}');
print(f'Standard Error of Training loss: {np.std(losses)}')
plt.plot(losses);


##%
# Multisample flow matching training (batchOT in Pooladian et al 2023)
flow_minibatch_ot = CNF(2, hidden_features=[256] * 3)

loss = FlowMatchingLoss(flow_minibatch_ot)
optimizer = torch.optim.AdamW(flow_minibatch_ot.parameters(), lr=1e-3)

temp = time()
losses = []
for epoch in tqdm(range(n_epochs), ncols=88):
    subset = torch.randint(0, len(data), (batch_size * n_iters,))
    # Minibatch OT
    for i in range(n_iters):
        batch_target = target[subset[i * batch_size:(i+1)*batch_size]]
        batch_data = data[subset[i * batch_size:(i+1)*batch_size]]
        M = ot.dist(batch_data, batch_target)

        # we assume data and noise on uniform weights
        a, b = torch.ones(batch_size) / batch_size, torch.ones(batch_size) / batch_size
        Pi = ot.emd(a, b, M)  # unregularized OT plan (earth mover distance)

        # with OT plan we can define which noise matched with which minibatch sample
        match_indices = torch.where(Pi != 0)[1]
        sorted_batch_noise = batch_target[torch.sort(match_indices)[1]]

        optimizer.zero_grad()
        loss(batch_data, batch_target).backward()
        losses.append(loss(batch_data, sorted_batch_noise).item())
        optimizer.step()

print(f'\nTotal training time: {time()- temp}');
print(f'Standard Error of Training loss: {np.std(losses)}');
plt.plot(losses);


content_flow = {'epoch': epoch + 1, 'model_dict': flow.state_dict()}
content_flow_minibatch = {'epoch': epoch + 1, 'model_dict': flow_minibatch.state_dict()}
content_flow_minibatch_ot = {'epoch': epoch + 1, 'model_dict': flow_minibatch_ot.state_dict()}

torch.save(content_flow, f'./saved_info/flow_content_{n_epochs}.pth')
torch.save(content_flow_minibatch, f'./saved_info/flow_minibatch_content_{n_epochs}.pth')
torch.save(content_flow_minibatch_ot, f'./saved_info/flow_minibatch_ot_content_{n_epochs}.pth')

# Sampling

for nfe in [2, 4, 8, 10, 20, 100]:
    method = 'euler'
    options = {'step_size': 1./nfe}
    t = torch.tensor([1.0, 0.1])
    with torch.no_grad():
        x = flow.decode(target, t=t, method=method, options=options).numpy()
        x_ot = flow_minibatch_ot.decode(target, t=t, method=method, options=options).numpy()
        x_batch = flow_minibatch.decode(target, t=t, method=method, options=options).numpy()

    nrows, ncols = 1, 4
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 3))
    ax[0].scatter(data[:, 0], data[:, 1], color='C0', alpha=0.5)
    ax[0].set_title('Data')
    ax[1].scatter(x[:, 0], x[:, 1], color='C1', alpha=0.5)
    ax[1].set_title('Full dataset')
    ax[2].scatter(x_batch[:, 0], x_batch[:, 1], color='C2', alpha=0.5)
    ax[2].set_title(f'Vanila Minibatch, nfe={nfe}')
    ax[3].scatter(x_ot[:, 0], x_ot[:, 1], color='C3', alpha=0.5)
    ax[3].set_title(f'OT Minibatch, nfe={nfe}')
    fig.savefig(f'./saved_info/checkerboard_samples_nfe={nfe}.pdf')
