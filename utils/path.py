from pathlib import Path

# current dir
CURRENT_DIR = Path(__file__).resolve().parent
# project dir
PROJECT_DIR = CURRENT_DIR.parent

INPUT_DIR = PROJECT_DIR/'input'
RESULT_DIR = PROJECT_DIR/'0_result'
PLOTS_DIR= PROJECT_DIR/'1_plots'

REPOS_FILE_PATH = "input/repos.txt"

##baseline data same level with the project dir
BASELINE_DIR = PROJECT_DIR.parent / 'scripts'
DATASET_DIR = BASELINE_DIR / 'dataset' / 'crosssim_repos'

TOOL_RESULT_DIR = BASELINE_DIR /'tools'
REPOPAL_RESULT = TOOL_RESULT_DIR/ 'repopal' / 'result'
CROSSSIM_RESULT = TOOL_RESULT_DIR/'crosssim' / 'result'
MUDABLUE_RESULT = TOOL_RESULT_DIR/'mudablue' / 'result'

EVA_RESULT_DIR = BASELINE_DIR /'eva_res'
EVA_REPOPAL_RESULT = EVA_RESULT_DIR/ 'repopal' / 'result'
EVA_CROSSSIM_RESULT = EVA_RESULT_DIR/'crosssim' / 'result'
EVA_MUDABLUE_RESULT = EVA_RESULT_DIR/'mudablue' / 'result'
EVA_BUILD_RESULT = EVA_RESULT_DIR/'build' 
ESM_RESULT = EVA_RESULT_DIR/'esm' 


print(f"CURRENT_DIR: {CURRENT_DIR}")