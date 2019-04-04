# Train Neural network on dataset with fast.ai

from pathlib import Path
from fastai.vision import *
import wandb

# Initialize W&B project
wandb.init(project="semantic-segmentation")

# Define hyper-parameters
config = wandb.config               # for shortening
config.framework = "fast.ai"        # AI framework used
config.img_size = 256               # size of image - can be 1 dim or tuple
config.batch_size = 8               # Batch size during training
config.epochs = 10                  # Number of epochs for training
encoder = models.resnet18           # encoder of unet (contracting path)
config.encoder = encoder.__name__
config.pretrained = True            # whether we use a frozen pre-trained encoder
config.weight_decay = 1e-4          # weight decay applied on layers
config.bn_weight_decay = False      # whether weight decay is applied on batch norm layers
config.one_cycle = False            # use the "1cycle" policy -> https://arxiv.org/abs/1803.09820
config.learning_rate = 3e-3         # learning rate

# Data paths
path_data = Path('../data/bdd100k/seg')
path_lbl = path_data/'labels'
path_img = path_data/'images'

# Associate a label to an input
get_y_fn = lambda x: path_lbl/x.parts[-2]/f'{x.stem}_train_id.png'

# Segmentation Classes extracted from dataset source code
# See https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py
segmentation_classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                        'pole','traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky', 'person', 'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle', 'bicycle', 'void']
void_code = 19      # used to define accuracy and disconsider unlabeled pixels

# Create a callback for logging on W&B
class WandBLogger(LearnerCallback):
    "Logs metrics on W&B while training"
    def __init__(self, learn:Learner): 
        super().__init__(learn)
        
    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        vals = {name:stat for name, stat in list(zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics))[1:]}
        wandb.log(vals)

# Load data into train & validation sets
src = (SegmentationItemList.from_folder(path_img)
       .split_by_folder(train='train', valid='val')
       .label_from_func(get_y_fn, classes = segmentation_classes))

# Resize, augment, load in batch & normalize
data = (src.transform(get_transforms(), size=config.img_size, tfm_y=True)
        .databunch(bs=config.batch_size)
        .normalize(imagenet_stats))     # let us use pre-trained networks

# Define accuracy & ignore unlabeled pixels
def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics = acc

# Create NN
learn = unet_learner(data, arch = encoder, pretrained = config.pretrained, metrics=metrics,
                     wd=config.weight_decay, bn_wd = config.bn_weight_decay, path=wandb.run.dir,
                     callback_fns=[WandBLogger])

# Store network topology
wandb.watch(learn.model)

# Train
if config.one_cycle:
    learn.fit_one_cycle(config.epochs, max_lr = slice(config.learning_rate))
else:
    learn.fit(config.epochs, lr = slice(config.learning_rate))
