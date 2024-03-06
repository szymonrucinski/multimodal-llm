from torch import nn
import torch


class MixtureOfExperts(nn.Module):
    def __init__(self, embed, num_experts):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(embed, embed) for _ in range(num_experts)]
        )
        self.gating = nn.Linear(embed, num_experts)

    def forward(self, x):
        gates = torch.softmax(self.gating(x), dim=-1).unsqueeze(-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = torch.sum(gates * expert_outputs, dim=2)
        return output


class Projections(nn.Module):
    def __init__(
        self,
        clip_embed,
        polka_embed,
        num_experts=4,
        num_projection_layers=6,
    ):
        super().__init__()

        self.MixtureOfExperts = MixtureOfExperts(clip_embed, num_experts)
        self.norm = nn.LayerNorm(polka_embed)
        self.output = nn.Linear(clip_embed, polka_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(polka_embed, polka_embed),
                    nn.GELU(),
                    nn.Linear(polka_embed, polka_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        x = self.MixtureOfExperts(x)
        x = self.output(x)
        self.norm(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual

        return x


class PatchReducerWithProjections(nn.Module):
    def __init__(
        self,
        num_patches,
        reduced_num_patches,
        clip_embed,
        phi_embed,
        num_projection_layers=4,
        num_experts=6,
    ):
        super().__init__()

        self.moe = MixtureOfExperts(clip_embed, num_experts)
        self.output = nn.Linear(clip_embed, phi_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(clip_embed, clip_embed),
                    nn.GELU(),
                    nn.Linear(clip_embed, clip_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        x = self.moe(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual
        x = self.output(x)
        return x
