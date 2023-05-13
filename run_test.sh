#!/bin/sh
#SBATCH --job-name=test_01 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32 # 80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e

export MASTER_PORT=12004
export WORLD_SIZE=1

export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

MODEL_TYPE=DiT-L/2
EPOCH_ID=475
DATASET=lsun_church
EXP=laflo_church_f8_dit
METHOD=dopri5
STEPS=0

CUDA_VISIBLE_DEVICES=0 python test_flow_latent.py --exp ${EXP} \
    --dataset ${DATASET} --batch_size 100 --epoch_id ${EPOCH_ID} \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    --model_type ${MODEL_TYPE} --num_classes 1 --label_dropout 0. \
    --master_port $MASTER_PORT --num_process_per_node 1 \
    --compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \
    --method ${METHOD} --num_steps ${STEPS} \
    # --use_karras_samplers \
    # --measure_time \
    # --method euler --step_size 0.02 \
    # --compute_nfe \

# CUDA_VISIBLE_DEVICES=0 python test_flow_latent.py --exp laflo_celeb_f8_dit \
#     --dataset celeba_256 --batch_size 100 --epoch_id 350 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --compute_fid \

# CUDA_VISIBLE_DEVICES=0 python test_flow_latent.py --exp laflo_f8_ \
#     --dataset ffhq_256 --batch_size 100 --epoch_id 275 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --compute_fid \
#     --master_port $MASTER_PORT \

# CUDA_VISIBLE_DEVICES=0 python test_flow_latent.py --exp laflo_f8_ \
#     --dataset ffhq_256 --batch_size 100 --epoch_id 275 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --compute_fid \
#     --master_port $MASTER_PORT \

# CUDA_VISIBLE_DEVICES=0 python test_flow_latent.py --exp laflo_imnet_f8_ditb2 \
#     --dataset latent_imagenet_256 --batch_size 50 --epoch_id 175 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --model_type DiT-B/2 --num_classes 1000 --label_dropout 0.1 \
#     --master_port $MASTER_PORT \
#     --method euler --num_steps 50 --cfg_scale 7. \
#     # --compute_fid --output_log ${EXP}_${EPOCH_ID}_adaptive.log \
#     # --method euler --step_size 0.02 \
#     # --measure_time \
#     # --compute_nfe \
