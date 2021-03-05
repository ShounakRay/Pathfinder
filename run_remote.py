# @Author: Shounak Ray <Ray>
# @Date:   05-Mar-2021 10:03:92:926  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: run_remote.py
# @Last modified by:   Ray
# @Last modified time: 05-Mar-2021 15:03:62:626  GMT-0700
# @License: [Private IP]

import os
import subprocess
import sys

import requests

# >> FILE STRUCTURE:
# REQUIREMENTS: root
#               |
#               –– Access Tokens                                    --> Contains access tokens for a specific owner(s)
#                  |
#                  -- {OwnerName}_TOKEN.txt                         --> The text file only with the access token
#               -- __init__.py                                      --> Indicates that the same_level Python File
#               -- {run_remote.py}                                  --> This interfacing
# GENERATED:    -- requirements.txt                                 --> The generated requirement for specific .pys
#               -- remote_shell_output.txt                          --> Used for debugging, cmd line output
#               |
#               -- ReferencePY                                      --> Directory for local Github repo storage
#                  |
#                  -- remote_{FileName}.py **                       --> The retrieved Python file from GitHub
#                  -- __pycache__                                   --> Cache files generated due to cmd runs
#                     |
#                     -- **                                         --> Content inside the cache folder

# Define Github authentication credentials and file retrieval info to complete a request
OWNER = 'ShounakRay'
REPO = 'AnomalyDetection'
BRANCH = 'main'
REPO_TYPE = 'private'
FILE_PATH = 'Anomaly_Detection_PKG.py'
REM_TO_LOC_PATH = 'ReferencePY'
GITHUB_TOKEN = token = open('Access Tokens/{owner}_TOKEN.txt'.format(owner=OWNER)).read().strip()


def _github(url: str, mode: str = "private", token: str = GITHUB_TOKEN):
    """Pulls the data from GitHub.

    Parameters
    ----------
    url : str
        The url to access the specific file.
    mode : str
        Whether the containing repository is private or public.
    token : str
        The owner's access token if the repository is private.

    Returns
    -------
    requests.models.Response
        The request object from GitHub repository, field contains text for respective file.

    """
    if mode == "public":
        return requests.get(url)
    else:
        headers = {'accept': 'application/vnd.github.v3.raw',
                   'authorization': 'token {}'.format(token)}
        return requests.get(url, headers=headers)


def cmd_runprint(command: str, prnt_file: bool = True, prnt_scrn: bool = False, ret: bool = False):
    """Runs a command in Python script and uses remote_shell_output.txt or screen as stdout/console.

    Parameters
    ----------
    command : str
        The command to be executed.
    prnt_file : bool
        Whether output of command should be printed to remote_shell_output.txt
    prnt_scrn : bool
        Whether output of command should be printed to screen.
    ret : bool
        Whether output of command should be returned.

    Returns [Optional]
    -------
    None
        Nothing is returned if ret is False
    str
        The output of the command is returned if ret is True

    """
    exec_output = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    if(prnt_file):
        print(exec_output, file=open('remote_shell_output.txt', 'w'))
    if(prnt_scrn):
        print(exec_output)
    if(ret):
        return exec_output


# Retrieve user's file from specific repository, given authentication token
out = _github('https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{FILE_PATH}'.format(OWNER=OWNER,
                                                                                             REPO=REPO,
                                                                                             BRANCH=BRANCH,
                                                                                             FILE_PATH=FILE_PATH),
              mode=REPO_TYPE)

print('STATUS: Request Completed.')

# Alert User w/ Status Code, request success or failure (waterfall)
if(out.status_code != 200):
    raise Exception('ERROR: Status Code: ' + str(out.status_code) +
                    '\nUnable to retrieve {FPATH}, check parameters!'.format(FPATH=FILE_PATH))
else:
    print('STATUS: {FPATH} Successfully Retrieved!'.format(FPATH=FILE_PATH))

# Check if "remote to local" storage directory exists and save python file
if not os.path.exists(REM_TO_LOC_PATH):
    os.makedirs(REM_TO_LOC_PATH)
print(out.text, file=open(REM_TO_LOC_PATH + '/' + 'remote_' + FILE_PATH, 'w'))

# Add specific folder to system path (to ensure that requirements.txt searches everywhere for dependencies)
sys.path.append(os.getcwd() + '/' + REM_TO_LOC_PATH)

# Any base packages need for run_remote.py to work
# NOTE: pipreqs is required to generate requirements.txt
_modules = ['pipreqs']
for mod in _modules:
    if(mod not in sys.modules):
        cmd_runprint('pip3 install {module}'.format(module=mod), prnt_file=False)
        pip_installed = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze',
                                                                                        ret=True).split()}
        if(mod not in pip_installed):
            raise ValueError('ERROR: {module} not installed properly by pip'.format(module=mod))

# Generate requirements.txt file
cmd_runprint('pipreqs')


def pip_install_confirm(mod: str):
    if(mod not in sys.modules):
        cmd_runprint('pip3 install {module}'.format(module=mod), prnt_file=False)
        pip_installed = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze',
                                                                                        ret=True).split()}
        if(mod not in pip_installed):
            raise ValueError('ERROR: {module} not installed properly by pip'.format(module=mod))


# Determine modules already installed via pip
installed_mods = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze', ret=True).split()}
# Determine modules which need to be installed by pip
required_mods = {mod.split("==")[0]: mod.split("==")[1] for mod in open('requirements.txt').read().split()}

# Install all the required modules
for r_mod in required_mods:
    if(r_mod in installed_mods):
        print('STATUS: {module} already installed. Version {version}'.format(module=r_mod,
                                                                             version=required_mods.get(r_mod)))
    else:  # Required module not installed, install it
        print('> STATUS: {module} not installed. Version {version} required.'.format(module=r_mod,
                                                                                     version=required_mods.get(r_mod)))
        cmd_runprint('pip3 install {module}'.format(module=r_mod))
        print('STATUS: {module} installed successfully.'.format(module=r_mod))

# Import the necessary package
# NOTE: Exec statement is used since import statement should be at the top of the file (PEP)
exec('from remote_{FPATH} import *'.format(FPATH=FILE_PATH.replace('.py', '')))

# EOF

# EOF
