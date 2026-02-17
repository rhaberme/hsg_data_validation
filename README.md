# hsg_data_validation

## Usage of the webapp

Short description on how to install and run the webapp locally.

### Installation
1. Use your Python installation of choice (if you don't have any yet, you can get it e.g. from https://www.python.org/)
    Make sure to check the box that says "Add Python to PATH" during installation.
2. Create a new virtual environment and activate it (not strictly necessary but recommended):
   - bash (Unix, including MacOS):
     - Creation: `python -m venv <venv>` (`python` might need to be replaced by your actual python installation)
     - Activation: `source <venv>/bin/activate`
       
     where `<venv>` is the path where the virtual environment is to be located
   - Windows (in the CMD)
     - Creation: `python -m venv <venv>` 
     - Activation: `.\<venv>\Scripts\activate`
     
     where `<venv>` is the path where the virtual environment is to be located
3. Clone the GitLab Repository to your PC: `git clone https://gitlab.com/rhaberme/hsg_data_validation.git`
4. Install all requirements: `pip install -r requirements.txt`

### Access
Link to the deployed Webapp:
https://hsgdatatool.streamlit.app/

### Running the web app locally
1. Open windows CMD console and navigate to the GitLab Repository to your PC
2. Run the webapp using `streamlit run Webtool/Start.py`
- If this doesn't work use `python -m streamlit run Webtool/Start.py`
- Please note that the Streamlit version must be 1.52 or higher (To update, run the command: `pip install --upgrade streamlit`)
3. A browser tab with the webapp should open automatically 
