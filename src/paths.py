import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "config"
RESULTS_DIR = ROOT_DIR / "results"
PDFS_DIR = DATA_DIR / "pdfs"