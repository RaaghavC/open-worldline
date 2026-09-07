# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental sizes and fixed sampling settings."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Spec:
    name: str
    height: int
    width: int
    frames: int = 17

    @property
    def latent_shape(self):
        return (48, 5, self.height // 16, self.width // 16)

    @property
    def observation_shape(self):
        return (1, 48, 1, self.height // 16, self.width // 16)

    @property
    def tokens(self):
        return 5 * self.prefix_tokens

    @property
    def prefix_tokens(self):
        return (self.height // 32) * (self.width // 32)

    def record(self):
        return {**asdict(self), 'latent_shape': list(self.latent_shape),
                'observation_shape': list(self.observation_shape),
                'tokens': self.tokens, 'prefix_tokens': self.prefix_tokens}


SPECS = {'baseline': Spec('baseline', 288, 512),
         'spatial': Spec('spatial', 704, 1248)}
SETTINGS = {'steps': 50, 'shift': 5.0, 'guidance': 5.0}
SEED = 20260908
# Pair and codec modes are measurements used to admit a later clip.
STAGE_SECONDS = {'codec': 600.0, 'pair': 900.0, 'clip': 1800.0}


def spec(name):
    if not isinstance(name, str) or name not in SPECS:
        raise ValueError('Explicit baseline or spatial profile required')
    return SPECS[name]
