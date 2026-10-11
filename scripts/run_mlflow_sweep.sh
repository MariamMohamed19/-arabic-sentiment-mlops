#!/usr/bin/env bash
# >= 5 distinct MLflow runs (T010). Edit grid to fit your GPU budget.
set -euo pipefail
python -m src.models.train --model arabert   --lr 2e-5 --epochs 3 --out artifacts/run1
python -m src.models.train --model arabert   --lr 3e-5 --epochs 3 --out artifacts/run2
python -m src.models.train --model arabert   --lr 5e-5 --epochs 2 --out artifacts/run3
python -m src.models.train --model camelbert --lr 2e-5 --epochs 3 --out artifacts/run4
python -m src.models.train --model camelbert --lr 3e-5 --epochs 3 --out artifacts/run5
echo "Open the UI:  mlflow ui   -> pick the best run by test_macro_f1 and register it."
