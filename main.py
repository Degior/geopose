import argparse
import os
from pathlib import Path

import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("LOCAL_RANK", "0")
import cv2
import torch.distributed as dist

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


if __name__ == "__main__":
    print('starting main')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument("--predict", action="store_true", help="predict for large images")
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    parser.add_argument(
        "--world_size", "--world-size", dest="world_size", type=int, default=1
    )
    parser.add_argument(
        "--local_rank", "--local-rank", dest="local_rank", type=int, default=0
    )
    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=1
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=4)
    parser.add_argument(
        "--downsample",
        type=int,
        help="factor for downsampling image at test time",
        default=1,
    )
    parser.add_argument(
        "--model-path",
        "--weights",
        dest="model_path",
        nargs="+",
        help="one or more model checkpoint paths",
    )
    parser.add_argument(
        "--dataset-dir",
        "--input-dir",
        dest="dataset_dir",
        type=str,
        help="dataset directory",
        default="./dataset",
    )

    parser.add_argument(
        "--test-sub-dir", type=str, help="test folder within datset-dir", default="test"
    )
    parser.add_argument(
        "--valid-sub-dir",
        type=str,
        help="validation folder within datset-dir",
        default="valid",
    )
    parser.add_argument(
        "--predictions-dir",
        "--output-dir",
        dest="predictions_dir",
        type=str,
        default="./predictions",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        help="square tile size used for large-image inference",
        default=2048,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="number of images to randomly sample for training",
        default=None,
    )
    parser.add_argument(
        "--rgb-suffix", type=str, help="suffix for rgb files", default="tif"
    )
    parser.add_argument(
        "--nan-placeholder",
        type=int,
        help="placeholder value for nans. use 0 for no placeholder",
        default=65535,
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="unit of AGLS (m, cm) -- converted inputs are in cm, downsampled data is in m",
        default="cm",
    )
    parser.add_argument(
        "--convert-predictions-to-cm-and-compress",
        type=str2bool,
        nargs="?",
        const=True,
        help="convert predictions to centimeters and compress them",
        default=True,
    )
    parser.add_argument(
        "--no-compress",
        dest="convert_predictions_to_cm_and_compress",
        action="store_false",
        help="skip prediction conversion and compression",
    )

    args = parser.parse_args()
    print(f"parsed args: {args}")

    if args.predict or args.test:
        if not args.model_path:
            parser.error("--model-path or --weights is required")
        missing_models = [
            model_path
            for model_path in args.model_path
            if not Path(model_path).is_file()
        ]
        if missing_models:
            parser.error(f"model checkpoint not found: {', '.join(missing_models)}")
        if not Path(args.dataset_dir).is_dir():
            parser.error(f"input directory not found: {args.dataset_dir}")
        if args.tile_size < 1:
            parser.error("--tile-size must be greater than zero")
        os.makedirs(args.predictions_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if (args.predict or args.test) and not torch.cuda.is_available():
        parser.error("CUDA is required for prediction")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    if args.predict:
        from utilities.inference import predict
        predict(args)

    elif args.test:
        from utilities.inference import test
        test(args)
    else:
        raise ValueError("Training is not supported, use train_segmentor instead")
