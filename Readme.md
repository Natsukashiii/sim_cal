## ReadMe

### 1. Add Data Sources
Place your data source(s) into the `input/` folder as `.csv` files.
Each CSV should represent the output of a model or feature extractor.  



### 2. Configure Parameters (`config.py`)

Update the following fields in `config.py`:

- **custom_attributes**: List of data source names (must match CSV file names in `input/`)
- **normalize**: Whether to apply normalization across different data sources (apply to current data source)
- **integrate_level_low / integrate_level_high**: Range of how many models to combine and compare.  
  Example: 1 to 3 means comparisons of individual models up to 3-model combinations.
- **pick_repo_num**: Number of repositories to sample (use `0` to include all)
---

### 3. Run the System

```bash
python run.py

(The execution may take 15mins ish)


### Environments
Python 3.11.7
pip install -r requirements.txt

