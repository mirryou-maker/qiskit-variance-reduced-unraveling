cd /scratch/cyyou68/qiskit-co
. /etc/profile.d/modules.sh
module load ANACONDA/1-2024.10 >/dev/null 2>&1
source activate qiskit 2>/dev/null || conda activate qiskit
export PYTHONUNBUFFERED=1
hostname; date
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
