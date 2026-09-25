export WANDB_DIR=/home/s0101/_scratch/wandb
export WANDB_CACHE_DIR=/home/s0101/_scratch/wandb
export WANDB_DATA_DIR=/home/s0101/_scratch/wandb
export TORCH_HOME=/home/s0101/_scratch/torch
export TRANSFORMERS_CACHE=/home/s0101/_scratch/transformers

source /opt/intelpython3/bin/activate
conda activate pytorch_k40_1.9
module load gpu/cuda-11.5

# sbatch  -p v100 --mem=32000 --gres=gpu:v100:1 --cpus-per-task=1 -t 20:00:00 -x tesla-a100 --job-name=depth --output='./logs/test_cs %j.txt' --wrap='./run.sh'
#sbatch  -p v100 --mem=32000 --gres=gpu:v100:1 -x "tesla-a101" --cpus-per-task=1 -t 20:00:00 --job-name=depth1 --output='./logs/test_cs %j.txt' --wrap='./run.sh'

sbatch \
-n1 -p hiperf \
--gres=gpu:8 \
--nodelist=tesla-a100 \
--cpus-per-task=8 \
--mem=60000 \
-t 20:00:00 \
--job-name=depth \
--output=./logs/train"_%j.txt" \
--wrap="./run.sh"