# -*- coding: utf-8 -*-
"""Simple test to verify Python environment works."""

import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import pandas as pd
    print("pandas version:", pd.__version__)
except ImportError as e:
    print("ERROR: pandas not found:", e)

try:
    import numpy as np
    print("numpy version:", np.__version__)
except ImportError as e:
    print("ERROR: numpy not found:", e)

try:
    import yfinance as yf
    print("yfinance version:", yf.__version__)
except ImportError as e:
    print("ERROR: yfinance not found:", e)

try:
    import loguru
    print("loguru: installed")
except ImportError as e:
    print("ERROR: loguru not found:", e)

print("\nBasic environment check complete!")
