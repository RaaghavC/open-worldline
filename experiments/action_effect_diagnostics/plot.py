# SPDX-License-Identifier: Apache-2.0
"""Plot retained latent-patch RMS maps; these are not RGB model outputs."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    arrays = np.load(args.results/'patch-rms.npz')
    names = ['target_patch_rms','new128_noise_0_patch_rms']
    fig, axes = plt.subplots(2,4,figsize=(12,4.7),layout='constrained')
    for row, name in enumerate(names):
        values = arrays[name]
        for col in range(4):
            im = axes[row,col].imshow(values[col],vmin=0,vmax=float(values.max()),cmap='magma',interpolation='nearest')
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0: axes[row,col].set_title(f'Future latent {col+1}')
        axes[row,0].set_ylabel('Target change' if row == 0 else 'Predicted change\n(first saved noise)')
        fig.colorbar(im,ax=list(axes[row]),shrink=.7,label='Latent RMS')
    fig.suptitle('Where the command changes the latent\nSeparate row scales show location; color-bar values show strength',fontsize=13)
    fig.savefig(args.results/'spatial-response.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
