import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.model.model import Model

model = Model()
w, b = model.learn(1000)

print()
print('Result learning')
print(f'Weight: {w}')
print(f'Bias: {b}')