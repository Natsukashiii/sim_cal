from pathlib import Path


# current dir
CURRENT_DIR = Path(__file__).resolve().parent
# project dir
PROJECT_DIR = CURRENT_DIR.parent

INPUT_DIR = PROJECT_DIR/'input'
RESULT_DIR = PROJECT_DIR/'0_result'
PLOTS_DIR= PROJECT_DIR/'1_plots'

REPOS_FILE_PATH = "input/repos.txt"

##baseline data
BASELINE_DIR = Path("/repos/")
DATASET_DIR = BASELINE_DIR / 'dataset' / 'crosssim_repos'

TOOL_RESULT_DIR = BASELINE_DIR /'tools'
REPOPAL_RESULT = TOOL_RESULT_DIR/ 'repopal' / 'result'
CROSSSIM_RESULT = TOOL_RESULT_DIR/'crosssim' / 'result'
MUDABLUE_RESULT = TOOL_RESULT_DIR/'mudablue' / 'result'



print(f"CURRENT_DIR: {CURRENT_DIR}")