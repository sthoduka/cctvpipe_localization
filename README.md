This repository contains the code for the submission to the [ICPR VideoPipe Challenge - Track on Temporal Defect Localization](https://codalab.lisn.upsaclay.fr/competitions/2284).


The main requirements are:

```
pytorch==1.9.0
pytorch-lightning==1.6.1
torchnet==0.0.4
tensorboard==2.8.0
torchvision==0.10.0
opencv-python==4.5.5.62
```

To train the network, set parameters in `train.sh` and run it. To produce the results on the test set, run `test.sh` after specifying which checkpoint to use.
