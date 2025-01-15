import os
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import torch.utils.checkpoint
from dataset.drone_dataset import VisDroneDataset
from data_script.process_data import drone_df
from AeroDiffusion import DiffusionModel


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

drone_data = drone_df('path/to/VisDrone/Data')

class VisDroneDataModule(pl.LightningDataModule):
    def __init__(self, df, args: DictConfig):
        super(VisDroneDataModule, self).__init__()
        self.args = args
        self.df = df
        self.kwargs = {
            "num_workers": args.num_workers,
            "persistent_workers": True if args.num_workers > 0 else False,
            "pin_memory": True
        }

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_data = VisDroneDataset(self.df, self.args)
        if stage == 'test':
            self.test_data = VisDroneDataset(self.df, self.args)

    def train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return self.trainloader

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def len_of_dataloader(self):
        return len(self.train_dataloader())


# training the model
def train(args: DictConfig):
    dataloader = VisDroneDataModule(drone_data, args)
    dataloader.setup('fit')

    model = DiffusionModel(args, steps_per_epoch=dataloader.len_of_dataloader())

    logger = TensorBoardLogger(save_dir=str(os.path.join(args.ckpt_dir, args.run_name)), name='log',
                               default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(os.path.join(args.ckpt_dir, args.run_name)),
        save_top_k=0,
        every_n_epochs=5,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callback_list = [lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list,
        strategy=DDPStrategy(find_unused_parameters=False)
    )

    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def sample(args: DictConfig):
    assert args.gpu_ids == 1 or len(args.gpu_ids) == 1, "Only one GPU is supported in test mode"
    dataloader = VisDroneDataModule(drone_data, args)
    dataloader.setup('test')
    model = DiffusionModel.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True
    )

    predictions = predictor.predict(model, dataloader)
    images = [sublist[0] for sublist in predictions]

    if not os.path.exists(args.sample_output_dir):
        try:
            os.mkdir(args.sample_output_dir)
        except:
            pass
    for i, image in enumerate(images):
        image.save(os.path.join(args.sample_output_dir, '{:04d}.png'.format(i)))


@hydra.main(config_path='.', config_name='config.yaml')
def main(args: DictConfig):
    pl.seed_everything(args.seed)

    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        sample(args)


if __name__ == '__main__':
    main()
