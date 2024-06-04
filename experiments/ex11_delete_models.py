import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':

    dataset = sys.argv[1]
    model = sys.argv[2]
    seed = int(sys.argv[3])

    for i in range(4):
        try:
            os.remove(f'models/{model}_nobias_{dataset}_{seed}_{i}.pt')
        except OSError as e:
            print(f"model models/{model}_nobias_{dataset}_{seed}_{i}.pt could not be removed")
