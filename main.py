import argparse
import os
import warnings

import numpy as np
import torch
from skimage import io

import models
import utils

ROOT = os.path.dirname(os.path.realpath(__file__))


def main(
    input: str,
    output: str,
    device: str = "cpu",
    registration=False,
    superresolution=True,
):
    assert registration ^ superresolution

    torch.set_grad_enabled(False)

    im = (
        torch.from_numpy(io.imread(input).astype(np.float32))
        .permute(2, 0, 1)
        .to(device)
    )

    if registration:
        CSR = models.CSR_Net(range_=10, in_dim=2, out_channels=2).float().to(device)
        CSR.load_state_dict(
            torch.load(
                f"{ROOT}/trained_models/CSR_Real_L1B.pth.tar",
                map_location=torch.device("cpu"),
            )["state_dictCSR"]
        )
        CSR.eval()

        im_r = utils.self_registered(im[:, None], csr=CSR)[:, 0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            io.imsave(output, im_r)

    if superresolution:
        REC = models.RCAN(n_colors=4).float().to(device)
        REC.load_state_dict(
            torch.load(
                f"{ROOT}/trained_models/REC_Real_L1B.pt",
                map_location=torch.device("cpu"),
            )
        )
        REC.eval()

        sr = utils.super_resolve(im[None], REC)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            io.imsave(output, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--task",
        type=str,
        default="superresolution",
        choices=["superresolution", "registration"],
    )

    args = parser.parse_args()
    main(
        args.input,
        args.output,
        args.device,
        args.task == "registration",
        args.task == "superresolution",
    )
