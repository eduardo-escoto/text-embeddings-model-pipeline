# Text Embeddings Model
- Author: Eduardo Escoto

## Usage

Setup YAML configuration files and setup the necessary db connection parameters, and tables that will be used to store data, as well as any lines with a #change this comment. 

Then Set pipeline options in run_model_pipeline_config which will dictate how the pipeline will automatically run. It is currently designed to run once a night and generate predictinos, and to be trained once every 7 days. This can all be set in that config file.

Schedule / run with: `python run_model_pipeline.py`
