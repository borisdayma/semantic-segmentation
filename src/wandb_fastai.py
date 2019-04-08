'''WandB Callback for fast.ai

This module hooks fast.ai Learners to Weights & Biases through a callback.
Requested logged data can be configured through the callback constructor.

Examples:
    WandBCalback can be used when initializing the Learner::

        from wandb_fastai import WandbCallback
        [...]
        learn = Learner(data, ..., callback_fns=WandbCallback)
        learn.fit(epochs)
    
    Custom parameters can be given using functools.partial::

        from wandb_fastai import WandbCallback
        from functools import partialmethod
        [...]
        learn = Learner(data,
                    callback_fns=partial(WandbCallback, ...),
                    ...)  # add "path=wandb.run.dir" if saving model
        learn.fit(epochs)

    Finally, it is possible to use WandbCallback only when starting
    training. In this case it must be instantiated::

        learn.fit(..., callbacks=WandbCallback())

    or, with custom parameters::

        learn.fit(..., callbacks=WandBCallback(learn, ...))
'''
import wandb
from fastai.basic_train import LearnerCallback
from fastai.callbacks import SaveModelCallback
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial


class WandbCallback(LearnerCallback):

    watch_called = False  # record if wandb.watch has been called

    def __init__(self,
                 learn,
                 log="all",
                 show_results=True,
                 save_model=False,
                 monitor='val_loss',
                 mode='auto'):
        """WandB fast.ai Callback

        Automatically saves model topology, losses & metrics.
        Optionally logs weights, gradients, sample predictions and best trained model.

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
        self.show_results = show_results

        # Logs model topology and optionally gradients and weights
        if not WandbCallback.watch_called:
            wandb.watch(self.learn.model, log=log)
            WandbCallback.watch_called = True

        # Add fast.ai callback for auto-saving best model
        if save_model:
            if Path(self.learn.path).resolve() != Path(
                    wandb.run.dir).resolve():
                raise ValueError(
                    'You must initialize learner with "path=wandb.run.dir" to sync model on W&B'
                )

            self.learn.callback_fns.append(
                partial(SaveModelCallback, monitor=monitor, mode=mode))

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Logs training loss, validation loss and custom metrics"

        # Log sample predictions
        if self.show_results:
            self.learn.show_results()  # pyplot display of sample predictions
            plt.tight_layout()  # adjust layout
            wandb.log({"Prediction Samples": plt}, commit=False)

        # Log losses & metrics
        logs = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] +
                    last_metrics))[1:]
        }

        wandb.log(logs)

        if self.show_results:
            plt.close()  # we can now close our figure
