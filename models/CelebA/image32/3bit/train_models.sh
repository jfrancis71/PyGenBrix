#!/bin/bash
#Pass in tensorboard_log folder in 1st argument


python PixelCNN_Spatial.py --tensorboard_log=$1/PixelCNN --max_epochs=5
python PixelCNN_Spatial.py --tensorboard_log=$1/PixelCNN --max_epochs=5
python LBAE_PixelCNN.py --tensorboard_log=$1/LBAE --max_epochs=5
python LBAE_PixelCNN.py --tensorboard_log=$1/LBAE --max_epochs=5
python GenderPixelCNN.py --tensorboard_log=$1/GenderPixelCNN --max_epochs=5
python GenderPixelCNN.py --tensorboard_log=$1/GenderPixelCNN --max_epochs=5
