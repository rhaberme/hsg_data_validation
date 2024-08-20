# hsg_data_validation

## Usage of the webapp

Short description on how to install and run the webapp locally.

### Installation
1. Use your Python installation of choice (if you don't have any yet, you can get it e.g. from https://www.python.org/)
2. Create a new virtual environment and activate it (not strictly necessary but recommended):
   - bash (Unix, including MacOS):
     - Creation: `python -m venv <venv>` (`python` might need to be replaced by your actual python installation)
     - Activation: `source <venv>/bin/activate`
       
     where `<venv>` is the path where the virtual environment is to be located
   - Windows (in the CMD)
     - Creation: `python -m venv <venv>`
     - Activation: `.\<venv>\Scripts\activate`
     
     where `<venv>` is the path where the virtual environment is to be located
3. Install all requirements: `pip install -r Webtool/requirements.txt`

### Running the web app
Run the webapp using `streamlit run Webapp/Start.py`
- If this doesn't work use `python -m streamlit run Webapp/Start.py`

Then, a browser tab with the webapp should open automatically 
