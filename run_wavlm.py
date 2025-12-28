import os
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import ModelParallelStrategy

from wavlm_src.params import get_params
from wavlm_src.module import AsrModule
from utils.dataset import AsrDataModule

def main():
    # Get parameters
    params = get_params()
    
    # Random seed for reproducibility
    seed_everything(params.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Define callbacks
    callbacks = []
    ckpt_path = os.path.join('checkpoints', params.run_name)
    if os.path.exists(ckpt_path):
        os.system(f'rm -rf {ckpt_path}')
    os.mkdir(ckpt_path)
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_path,
        filename='best',
        monitor='val/loss',
        mode='min',
    ))
    callbacks.append(EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=params.patience,
        min_delta=params.min_delta,
        stopping_threshold=0.0
    ))

    # Define distribution strategy
    if params.gpus>1:
        # strategy = ModelParallelStrategy(data_parallel_size=2, 
        #                                 tensor_parallel_size=1,)
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    # Define trainer
    trainer = Trainer(
            callbacks=callbacks,
            fast_dev_run=params.debug,
            accelerator='gpu',
            devices=params.gpus,
            overfit_batches=params.overfit_batches if params.overfit_batches<1 else int(params.overfit_batches),
            limit_train_batches=1,
            limit_val_batches=10,
            max_epochs=params.max_epochs,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=1,
            num_sanity_val_steps=1,
            strategy=strategy
        )

    # Define model and data
    data_module = AsrDataModule(params)
    vocab_size = max([data_module.vocab_size, params.vocab_size])
    module = AsrModule(params, vocab_size, data_module.tokenizer)
    
    trainer.fit(module, data_module)
    trainer.test(module, data_module, ckpt_path='best')

if __name__=='__main__':
    main()