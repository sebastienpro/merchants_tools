if [ ! -d "./VirtualEnv" ]; then
  python3 -m venv VirtualEnv
fi

source ./VirtualEnv/bin/activate
pip install -r requirements.txt