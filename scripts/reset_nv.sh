# 1. Kill ALL related processes
pkill -9 -f nanodeploy
pkill -9 -f spoke_hub
pkill -9 -f test_deep

# 2. Check for stuck GPU processes
nvidia-smi

# 3. Check NVSHMEM shared memory remnants
ls -la /dev/shm
ipcs -m

# 4. Reset CUDA (if nvidia-smi shows stuck processes)
sudo nvidia-smi --gpu-reset

# 5. Clear any NVSHMEM files
rm -f /dev/shm/nvshmem* 2>/dev/null
rm -f /tmp/nvshmem* 2>/dev/null
