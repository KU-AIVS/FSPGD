#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

#python test.py  --model deeplabv3_resnet101 --path /home/aivs/바탕화면/hdd/ES/pretrained/best_deeplabv3plus_resnet101_voc_os16.pth --mode test
#python test.py  --model deeplabv3_resnet101 --path /home/aivs/바탕화면/hdd/ES/pretrained/best_deeplabv3_resnet101_voc_os16.pth --attack_root /home/aivs/바탕화면/cospgd/results/adversarial_examples/proposed_deeplabv3_resnet50 --mode test
#python test.py  --model deeplabv3_resnet101 --path /home/aivs/바탕화면/hdd/ES/pretrained/best_deeplabv3_resnet101_voc_os16.pth --attack_root /home/aivs/바탕화면/cospgd/results/adversarial_examples/proposed_deeplabv3_resnet50 --mode test
python test.py  --model pspnet_resnet50 --path /home/aivs/바탕화면/hdd/ES/pretrained/pspnet_resnet50_voc.pth --mode test

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime