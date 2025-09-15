import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

from walidacja_funkcji import *

sys.path.append('/home/mamusiaarusia/myenv/lib/python3.12/site-packages/torch')

prober = rs_prober_NKG(epsilon=0.1, looking_x_left=-1, looking_x_right=1, from_x=0)

def get_2d_data(PROBE_SIZE: int):
    rs = prober.rejection_sampling(length=PROBE_SIZE // 2)
    theta = np.random.uniform(0, 2*np.pi, size=PROBE_SIZE // 2)
    xs = rs * np.cos(theta)
    ys = rs * np.sin(theta)
    return np.vstack([xs, ys]).T


class NF_layer(nn.Module):
    def __init__(self, translate_layers: list[nn.Module], scale_layers: list[nn.Module], lr):
        super().__init__()
        self.translate = nn.Sequential(*translate_layers)
        self.scale = nn.Sequential(*scale_layers)
        self.optim = torch.optim.RMSprop(self.parameters(), lr=lr) # tu nie jestem pewny moze dwa oddzielne optimizery powinny isc idk

        self.to('cuda')

    # mozna dodac loss_backward z funkcja probkowania z NKG
    def loss(self, output: torch.Tensor, log_diag: torch.Tensor = None): 
        return 0.5 * (output ** 2).mean() - log_diag.mean() # No czekaj ale do czego to tak naprawde zmierza

    def loss_and_step(self, output: torch.Tensor, log_diag: torch.Tensor = None):
        self.zero_grad()
        loss = self.loss(output, log_diag)
        loss.backward()
        self.optim.step()
        return loss.item()

    def calculate_forward(self, input: torch.Tensor, function: callable) -> torch.Tensor:
        div_indx = input.shape[-1] // 2
        x1 = input[:, :div_indx]
        x2 = input[:, div_indx:]

        scaled = self.scale(x1)
        translated = self.translate(x1)
        diag_sum = scaled # we wzorze jest dzielenie przez N, czyli srednia troche nawet zawyza wynik, ale to chyba dobrze
        x1 = function(x2, scaled, translated)
        z = torch.cat([x2, x1], dim=-1) # swap

        return z, diag_sum, x1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.calculate_forward(input, lambda x2, scaled, translated: (x2 + translated) * torch.exp(scaled))

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        return self.calculate_forward(input, lambda x2, scaled, translated: x2 / torch.exp(scaled) - translated)


class NF(nn.Module):
    def __init__(self, num_layers: int, input_size: int, lr: float):
        super().__init__()

        def scaling_components():
            return [
                nn.Linear(input_size, input_size), nn.Tanh(),
                nn.Linear(input_size, input_size), 
                nn.Linear(input_size, input_size), nn.Tanh(),
                nn.Linear(input_size, input_size), 
                nn.Linear(input_size, input_size), 
            ]

        def translating_components():
            return [
                nn.Linear(input_size, input_size), 
                nn.Linear(input_size, input_size), 
            ]

        self.layers = nn.ModuleList([NF_layer(translating_components(), scaling_components(), lr) for _ in range(num_layers)])
        self.to('cuda')

    def forward(self, x):
        return self.iterate_layers(x, forward=True, learn=False)

    def inverse(self, x):
        return self.iterate_layers(x, forward=False, learn=False)

    def loss_and_step(self, x):
        return self.iterate_layers(x, forward=True, learn=True)

    def iterate_layers(self, input: torch.Tensor, forward: bool, learn: bool):
        output = input.clone()
        transformed = None
        losses = 0

        iter_layers = self.layers if forward else reversed(self.layers)
        for layer in iter_layers:
            if forward:
                output, diag_sum, transformed = layer.forward(output)
            else:
                output, diag_sum, transformed = layer.inverse(output)

            if learn:
                losses += layer.loss_and_step(transformed, diag_sum)
                output = output.detach()

        if learn:
            return losses

        return output


def learn_nf(nf_model: nn.Module, GENERATOR_SAMPLES_TO_RETURN: int, BATCH_SIZE: int, EPOCHS: int, dim: int = 0):
    PROBE_SIZE = GENERATOR_SAMPLES_TO_RETURN * BATCH_SIZE
    values = get_2d_data(PROBE_SIZE=PROBE_SIZE).reshape(BATCH_SIZE, -1, 2)
    x = torch.Tensor(values[:, :, dim]).cuda()
    loss_history = np.empty(EPOCHS)

    nf_model.train()
    for epoch in range(EPOCHS):
        print(f'\r{epoch / (EPOCHS - 1) * 100:.1f}%', end='', flush=True)
        loss_history[epoch] = nf_model.loss_and_step(x)
        if epoch % 5 == 0:
            values = get_2d_data(PROBE_SIZE=PROBE_SIZE).reshape(BATCH_SIZE, -1, 2)
            x = torch.Tensor(values[:, :, dim]).cuda()

    nf_model.eval()

    plt.plot(np.log10(loss_history), 'o', markersize=0.3)
    plt.title(r'$log_{10}$(Loss)')
    plt.show()

    temp = get_2d_data(PROBE_SIZE=PROBE_SIZE * 10).reshape(BATCH_SIZE * 10, -1, 2)
    x = torch.Tensor(temp[:, :, dim]).cuda()
    theory = np.random.randn(PROBE_SIZE * 5)
    values = nf_model.forward(x)
    inversed = nf_model.inverse(values).detach().flatten().cpu().numpy()
    values = values.detach().flatten().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['#FF6B6B', '#4ECDC4']

    ax1.hist(inversed, bins=100, range=[-5, 5], label='f⁻¹(f(x))', 
            color=colors[0], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.hist(x.detach().flatten().cpu().numpy(), bins=100, range=[-5, 5], label='Original x',
            color=colors[1], alpha=0.6, edgecolor='white', linewidth=0.5)
    ax1.set_title('Invertibility Check', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(frameon=False, fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.hist(values, bins=100, range=[-5, 5], label='After NF',
            color='#FF1744', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.hist(theory, bins=100, range=[-5, 5], label='Normal Distribution',
            color='#00E676', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend(frameon=False, fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()