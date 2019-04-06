import wandb
from fastai.basic_train import LearnerCallback
from fastai.callbacks import SaveModelCallback
import matplotlib.pyplot as plt


class WandBCallback(LearnerCallback):
    def __init__(self,
                 learn,
                 log="all",
                 show_results=True,
                 save_model=False,
                 monitor='val_loss',
                 mode='auto'):
        """WandB fast.ai Callback

        Automatically saves model topology, losses & metrics.
        Optionally logs weights, gradients, sample image predictions and best trained model.

        Args:
            learn (fastai.basic_train.Learner): the fast.ai learner to hook.
            log (str): One of "gradients", "parameters", "all", or None
            show_results (bool): whether we want to display sample predictions
            save_model (bool): save model at the end of each epoch
            monitor (str): metric to monitor for saving best model
            mode (str): "auto", "min" or "max" to compare "monitor" values and define best model
        """

        if wandb.run is None:
            raise ValueError(
                'You must call wandb.init() before WandbCallback()')
        super().__init__(learn)
        self.log = log
        self.show_results = show_results

        # Add fast.ai callback for auto-saving best model
        if save_model:
            if self.learn.path != wandb.run.dir:
                raise ValueError(
                    'You must initialize learner with "path=wandb.run.dir" to sync model on W&B'
                )
            self.save_model = save_model
            self.learn.callback_fns.append(
                SaveModelCallback(
                    self.learn, monitor=self.monitor, mode=self.mode))

    def on_train_begin(self, **kwargs):
        "Logs model topology and optionally gradients and weights"

        super().on_train_begin(**kwargs)
        wandb.watch(self.learn.model, log=self.log)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Logs training loss, validation loss and custom metrics"

        # Log sample predictions
        if self.show_results:
            self.learn.show_results()  # pyplot display of sample predictions
            plt.tight_layout()  # adjust layout
            wandb.log({"chart": plt}, commit=False)

        # Log losses & metrics
        logs = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] +
                    last_metrics))[1:]
        }

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs)
