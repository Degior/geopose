import argparse
import json
import os
import re
import uuid
from typing import Dict


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from utilities.evaluate import create_arg_parser, evaluate_r2
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from training.logging_utils import print_main, terminal_progress

import torch.distributed as dist

from training.geopose_dataset import GeoposeDataset
from utilities.misc_utils import save_image
import numpy as np



import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from training.trainer import TrainConfiguration, PytorchTrainer, Evaluator

from torch.utils.data import DataLoader
import torch.distributed



class GeoposeEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"r2": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:

        print_main('Validation: generating predictions', rank=local_rank)
        for sample in terminal_progress(dataloader, rank=local_rank, desc='Validation'):
                image = sample["image"]
                city_ohe = sample["city_ohe"]
                agl = sample["agl"]
                cities = sample['city']
                names = sample['name']

                image = image.cuda().float()
                city_ohe = city_ohe.cuda().float()
                with torch.no_grad():
                    output = model.forward(image.float(), city_ohe.float())
                    xydir_pred, agl_pred, mag_pred, scale_pred = output['xydir'], output['height'], output['mag'], output[
                        'scale']
                    scale_pred = torch.unsqueeze(scale_pred, 1)
                agl = agl.cpu().detach().numpy()
                xydir_pred = xydir_pred.cpu().detach().numpy()
                agl_pred = agl_pred.cpu().detach().numpy()
                scale_pred = scale_pred.cpu().detach().numpy()

                for batch_ind in range(agl.shape[0]):
                    city = cities[batch_ind]
                    name = names[batch_ind]
                    # vflow pred
                    angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                    vflow_data = {
                        "scale": float(scale_pred[batch_ind].item()),
                        "angle": np.float64(angle),
                    }

                    # agl pred
                    curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                    curr_agl_pred[curr_agl_pred < 0] = 0
                    agl_resized = curr_agl_pred

                    # save
                    os.makedirs(self.args.preds_dir, exist_ok=True)
                    agl_path = os.path.join(self.args.preds_dir, f"{city}_{name}_AGL.tif")
                    vflow_path = os.path.join(self.args.preds_dir, f"{city}_{name}_VFLOW.json")

                    with open(vflow_path, "w") as f:
                        json.dump(vflow_data, f)
                    save_image(agl_path, agl_resized)
        torch.cuda.synchronize()
        dist.barrier()
        res = {}
        if local_rank == 0:
            e_args = create_arg_parser().parse_args('')
            e_args.truth_dir = self.args.train_dir
            e_args.predictions_dir = self.args.preds_dir
            e_args.output_dir = os.path.join(self.args.output_dir, "evaluation")
            e_args.num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
            e_args.compact = True
            res = {"r2": evaluate_r2(e_args)}
        dist.barrier()
        torch.cuda.empty_cache()
        return res

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        if current_metrics["r2"] > prev_metrics["r2"]:
            print("Validation R2 improved: {:.6f} -> {:.6f}".format(prev_metrics["r2"], current_metrics["r2"]), flush=True)
            improved["r2"] = current_metrics["r2"]
        else:
            print("Validation R2: current={:.6f}, best={:.6f}".format(current_metrics["r2"], prev_metrics["r2"]), flush=True)
        return improved


def parse_args():
    parser = argparse.ArgumentParser(" Segmentor Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/b3.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str)
    arg('--run-id', type=str)
    arg('--stage', choices=('train', 'base', 'tune', 'height_tune'), default='train')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='')
    arg('--city', type=str, default='')
    arg('--data-dir', type=str, default="/wdata/train/")
    arg('--train-dir', type=str, default="/mnt/sota/datasets/geopose/train/")
    arg('--preds-dir', type=str)
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str)
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", "--local-rank", default=int(os.environ.get("LOCAL_RANK", "0")), type=int)
    arg("--world-size", default=int(os.environ.get("WORLD_SIZE", "1")), type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--crop', type=int, default=1024)
    arg('--multiplier', type=int, default=1)
    arg('--min-valid-fraction', type=float, default=0.0,
        help='Preferred valid crop fraction; best usable attempt is used if it cannot be reached')
    arg('--auto-schedule', action='store_true')
    arg('--log-every', type=int, default=100, help='Rank 0 training output interval in steps')

    args = parser.parse_args()
    if args.log_every < 1:
        parser.error('--log-every must be positive')
    if not 0 <= args.min_valid_fraction <= 1:
        parser.error('--min-valid-fraction must be between 0 and 1')
    if args.resume and not os.path.isfile(args.resume):
        parser.error(f'Checkpoint not found: {args.resume}')
    if not args.run_id:
        shared_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('TORCHELASTIC_RUN_ID')
        if shared_id and shared_id != 'none':
            args.run_id = 'run-' + re.sub(r'[^A-Za-z0-9_-]', '-', shared_id)
        elif args.world_size > 1:
            parser.error('--run-id is required when the launcher provides no shared run ID')
        else:
            args.run_id = 'run-' + uuid.uuid4().hex[:12]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', args.run_id):
        parser.error('Invalid --run-id')
    stage_dir = os.path.join('weights', 'experiments', args.run_id, f'fold_{args.fold}', args.stage)
    args.output_dir = args.output_dir or os.path.join(stage_dir, 'weights')
    args.logdir = args.logdir or os.path.join(stage_dir, 'tensorboard')
    args.preds_dir = args.preds_dir or os.path.join(stage_dir, 'predictions')

    return args


def create_data_datasets(args):
    train_dataset = GeoposeDataset(mode="train", dataset_dir=args.data_dir, folds_csv=args.folds_csv, fold=args.fold,
                                   crop_size=args.crop, multiplier=args.multiplier,
                                   min_valid_fraction=args.min_valid_fraction)
    val_dataset = GeoposeDataset(mode="val", dataset_dir=args.data_dir, folds_csv=args.folds_csv, fold=args.fold,
                                 crop_size=args.crop)
    if args.city:
        train_dataset.set_city(args.city)
        val_dataset.set_city(args.city)
    return train_dataset, val_dataset


def main():
    args = parse_args()
    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        auto_schedule=args.auto_schedule,
        log_every=args.log_every,
    )

    data_train, data_val = create_data_datasets(args)
    seg_evaluator = GeoposeEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)
    try:
        trainer.fit()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
