# LeRobot pod setup — run once on the network volume, then survive forever

uv pip install --python /lerobot/.venv/bin/python jupyterlab gym-pusht

# .pth shim — fixes gym_pusht autoload + task_description backfill
# Root cause: LeRobot 0.5.2 builds env via gym.make() relying on gymnasium autoload,
# but this gym-pusht build ships no gymnasium.envs entry point.
# Also backfills task_description/task on PushTEnv (0.5.2 reads them unguarded).
cat > /lerobot/.venv/lib/python3.12/site-packages/zzz_autoload_gym_pusht.pth << 'PTH'
import gym_pusht, gym_pusht.envs.pusht as _p; _p.PushTEnv.task_description = getattr(_p.PushTEnv, "task_description", ""); _p.PushTEnv.task = getattr(_p.PushTEnv, "task", "")
PTH

# Verify
/lerobot/.venv/bin/python -c "import gymnasium as g; e=g.make('gym_pusht/PushT-v0'); e.get_wrapper_attr('task_description'); print('gym_pusht setup OK')"

# Start Jupyter
/lerobot/.venv/bin/python -m jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
  --ServerApp.token="" --ServerApp.password="" \
  --ServerApp.allow_origin="*" --ServerApp.allow_remote_access=True
