import torch
from lightning.pytorch import Trainer

from wavlm_src.params import get_params
from wavlm_src.module import WavlmModule
from utils.dataset import AsrDataModule

def main():
    # Get parameters
    params = get_params()

    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Define trainer
    trainer = Trainer(
            accelerator='gpu',
            devices=1,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=1,
            num_sanity_val_steps=1,
            precision='bf16-mixed',
        )

    # Define model and data
    data_module = AsrDataModule(params)
    vocab_size = max([data_module.vocab_size, params.vocab_size])
    # module = WavlmModule(params, vocab_size, data_module.tokenizer)
    module = WavlmModule.load_from_checkpoint('checkpoints/finetune_epoch60_layerweights_wer.0948/best.ckpt',
                                              params=params, vocab_size=vocab_size, tokenizer=data_module.tokenizer)
    
    trainer.test(module, data_module)

if __name__=='__main__':
    main()