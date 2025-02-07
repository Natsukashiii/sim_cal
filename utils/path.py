from pathlib import Path


# current dir
CURRENT_DIR = Path(__file__).resolve().parent
# project dir
PROJECT_DIR = CURRENT_DIR.parent

INPUT_DIR = PROJECT_DIR/'input'
RESULT_DIR = PROJECT_DIR/'0_result'


##baseline data
BASELINE_DIR = PROJECT_DIR/'project_sibling_update'
BASELINE_REPOPAL = BASELINE_DIR/'baseline_output'/'repopal'
BASELINE_MUDABLUE = BASELINE_DIR/'baseline_output'/'mudablue'
BASELINE_CROSSSIM = BASELINE_DIR/'baseline_output'/'crosssim'
BASELINE_CROSSSIM_GRAPH =  BASELINE_DIR / 'dataset'/'crosssim'/'graph'


print(f"CURRENT_DIR: {CURRENT_DIR}")