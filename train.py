import sys
import traceback

import comet_ml
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from contextlib import redirect_stdout
from os.path import join
from shutil import rmtree

from config import update_args, parse_console
from data import get_dataloader
from data.dataset import get_dataset
from models import get_model

def main(
    args, 
    args_file_path: str, 
    tmp_results_dir: str, 
    train_log_file_path: str
    ) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    train_data_dict = get_dataset(args).train_data_dict

    train_dataloader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['train_sampler'],
    )

    validation_dataloader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['validation_sampler'],
    )

    # model
    model = get_model(
        args=args,
        device=device,
        hparams={
            'learning rate': args.TRAIN.LR,
            'batch size': args.TRAIN.BATCH_SIZE,
        },
    ).to(device)

    comet_logger = CometLogger(
        api_key=args.COMET.API_KEY, 
        project_name=args.COMET.PROJECT_NAME, 
        workspace=args.COMET.WORKSPACE, 
    )

    checkpoint_callback = ModelCheckpoint(monitor='validation_accuracy')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        distributed_backend=args.TRAIN.DISTRIBUTED_BACKEND,
        gpus=args.TRAIN.GPUS,
        logger=comet_logger,
        max_epochs=args.TRAIN.MAX_EPOCHS,
        replace_sampler_ddp=False,
    )
    
    trainer.fit(model, train_dataloader, validation_dataloader)
    with open(args_file_path, 'w') as f:
        with redirect_stdout(f):
            print(args.dump())

    rmtree(tmp_results_dir, ignore_errors=True)

if __name__ == '__main__':
    option = parse_console()
    args = update_args(cfg_file=option.cfg_file_path)
    main(
        args=args,
        args_file_path=option.args_file_path,
        tmp_results_dir=option.tmp_results_dir,
        train_log_file_path=option.train_log_file_path
    )
    print('Finished training')