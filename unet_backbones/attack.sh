#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`
python attack.py --attack cospgd --iterations 20 --source_model deeplabv3_resnet50_voc --dataset pascal_voc --mode adv_attack
#python attack.py --attack cospgd --iterations 20 --source_model deeplabv3_resnet50_voc --target_model deeplabv3_resnet50_voc --dataset pascal_voc --mode trans_test
#python attack.py --attack cospgd --iterations 20 --source_model deeplabv3_resnet50_voc --target_model deeplabv3_resnet101_voc --dataset pascal_voc --mode trans_test
#python attack.py --attack cospgd --iterations 20 --source_model deeplabv3_resnet50_voc --target_model psp_resnet50_voc --dataset pascal_voc --mode trans_test
#python attack.py --attack cospgd --iterations 20 --source_model deeplabv3_resnet50_voc --target_model psp_resnet101_voc --dataset pascal_voc --mode trans_test

#python attack.py --attack proposed --iterations 20 --targeted False --norm inf --alpha 0.01 --epsilon 0.03 --model pspnet_resnet101 -source_layer layer3_4 --lamda 0.5
#python attack.py --attack cospgd --iterations 20 --targeted False --norm inf --alpha 0.01 --epsilon 0.03 --model deeplabv3_resnet50 -source_layer layer3_4 --lamda 0.5 --mode test

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime