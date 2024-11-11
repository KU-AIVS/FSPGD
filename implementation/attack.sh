#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`
python attack.py --attack $1 --mode $1 --dataset $2 --pretrained_data $3  --cosine $4 --source_model $5 --target_model $6

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime