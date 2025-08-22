import fsspec
import hydra
# import lightning as L
import omegaconf

import os
import torch
import tqdm

import dataloader
import diffusion
import utils

# DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def _load_from_checkpoint(config, tokenizer):
    if 'hf' in config.backbone:
        return diffusion.Diffusion(
            config, tokenizer=tokenizer).to('cuda')

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config)


# @L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    import rich.syntax
    import rich.tree

    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
                '{}/config_tree.txt'.format(
                    config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


# @L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def generate_samples(config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    print('Text samples:', text_samples)
    if not config.sampling.semi_ar:
        print('Generative perplexity:',
              model.gen_ppl_metric.compute())
    return text_samples


def _ppl_eval(config, logger, tokenizer):
    logger.info('Starting Zero Shot Eval.')

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        # wandb_logger = L.pytorch.loggers.WandbLogger(
        #     config=omegaconf.OmegaConf.to_object(config),
        #     **config.wandb)
        import wandb
        wandb_config = omegaconf.OmegaConf.to_container(config, resolve=True)
        wandb_logger = wandb.init(
            project=config.wandb.get("project", "default_project"),
            config=wandb_config,
            **{k: v for k, v in config.wandb.items() if k != "project"}
        )
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
    wandb_logger = None
    if config.get('wandb', None) is not None:
        import wandb
        wandb_config = omegaconf.OmegaConf.to_container(config, resolve=True)

        wandb_logger = wandb.init(
            project=config.wandb.get("project", "default_project"),
            config=wandb_config,
            **{k: v for k, v in config.wandb.items() if k != "project"}
        )

    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(
                config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    ddp_setup()
    logger.info('Starting Training.')
    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer, save_every=100, snapshot_path="snapshot.ckpt")
    # model = model.to(device)
    model.logger = wandb_logger

    # enable grads
    torch.set_grad_enabled(True)
    train_ds = model.on_train_start(train_ds)
    if model.gpu_id == 0:
        _print_batch(train_ds, valid_ds, tokenizer)

    optimizer, sched_dict = model.configure_optimizers()
    scheduler = sched_dict.get("scheduler", None)  # interval='step' in your config

    for ei in range(model.epochs_run, config.trainer.max_steps):
        losses = []

        for idx, batch in enumerate(train_ds):
            if (idx + 1) % config.trainer.log_every_n_steps == 0:
                arr = torch.tensor(losses)
                if model.gpu_id == 0:
                    logger.info(
                        f"GPU: {model.gpu_id}, epoch {ei}, batch {idx + 1}/{len(train_ds)}, loss mean {torch.mean(arr):6f}, loss std {torch.std(arr):6f}")
                # sys.stdout.flush()
            # Move batch tensors to the correct device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.gpu_id, non_blocking=True)

            # train step
            loss = model.training_step(batch)

            # clear gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            model.optimizer_step(optimizer, scheduler)

            losses.append(loss.detach())

            if (idx + 1) % config.trainer.val_check_interval == 0:
                if model.gpu_id == 0:
                    logger.info("Validation step...")
                model.on_validation_epoch_start()
                val_losses = []
                for val_batch in tqdm.tqdm(valid_ds):
                    for k, v in val_batch.items():
                        if isinstance(v, torch.Tensor):
                            val_batch[k] = v.to(model.gpu_id, non_blocking=True)
                    val_loss = model.validation_step(val_batch)
                    val_losses.append(val_loss.detach())
                    model.logger.log({"val_loss": val_loss})
                arr = torch.tensor(val_losses)
                if model.gpu_id == 0:
                    print(f"validation loss mean {torch.mean(arr):6f}, loss std {torch.std(arr):6f}")
                model.on_validation_epoch_end()

            if config.callbacks.checkpoint_every_n_steps.every_n_train_steps == 0:
                model.on_save_checkpoint(ckpt_path=ckpt_path)

    destroy_process_group()


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    utils.seed_everything(config.seed)

    # Only print config on GPU 0 (main process) in DDP setting
    if not os.environ.get("LOCAL_RANK") or int(os.environ["LOCAL_RANK"]) == 0:
        _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'sample_eval':
        generate_samples(config, logger, tokenizer)
    elif config.mode == 'ppl_eval':
        _ppl_eval(config, logger, tokenizer)
    else:
        _train(config, logger, tokenizer)


if __name__ == '__main__':
    main()
