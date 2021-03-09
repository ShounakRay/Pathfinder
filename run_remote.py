# @Author: Shounak Ray <Ray>
# @Date:   05-Mar-2021 10:03:92:926  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: run_remote.py
# @Last modified by:   Ray
# @Last modified time: 08-Mar-2021 13:03:95:958  GMT-0700
# @License: [Private IP]

import os
import shutil
import subprocess
import sys
import time

import requests

# >> FILE STRUCTURE:
# REQUIREMENTS: root
#               |
#               –– Access Tokens                                    --> Contains access tokens for a specific owner(s)
#                  |
#                  -- {OwnerName}_TOKEN.txt                         --> The text file only with the access token
#               -- __init__.py                                      --> Indicates that the .py File is a package
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

_ = """
#######################################################################################################################
##########################################   AUTHENTICATION AND LOCATION   ############################################
#######################################################################################################################
"""

# Define Github authentication credentials and file retrieval info to complete a request
OWNER = 'ShounakRay'
REPO = 'AutoAnalytics'
BRANCH = 'main'
REPO_TYPE = 'private'
FILE_PATH = 'generic_data_prep.py'
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


def pip_install_confirm(mod: str, ver: bool = False):
    """Install a package using pip and confirm that it was installed.

    Parameters
    ----------
    mod : str
        The name of the module to be installed.
    ver : bool
        Whether the output of command installing the module should be printed to the screen

    Returns
    -------
    None
        Nothing. Only a command is executed and an output is – potentially – printed.

    """

    # To ensure consistency between requirement.txt files
    mod = mod.replace('_', '-')
    if(mod not in sys.modules):
        output = cmd_runprint('pip3 install {module}'.format(module=mod), prnt_file=False, ret=True)
        if(ver):
            print(output)
        pip_installed = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze',
                                                                                        ret=True).split()}
        if(mod not in pip_installed):
            print("WARNING: {module} not installed properly by pip.".format(module=mod))


_ = """
#######################################################################################################################
###################################################   RETRIEVAL   #####################################################
#######################################################################################################################
"""
# Retrieve user's file from specific repository, given authentication token
# out = _github('https://api.github.com/{OWNER}/{REPO}/{BRANCH}/{FILE_PATH}'.format(OWNER=OWNER,
#                                                                                   REPO=REPO,
#                                                                                   BRANCH=BRANCH,
#                                                                                   FILE_PATH=FILE_PATH),
#               mode=REPO_TYPE)

# send a request
out = requests.get('https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}'.format(OWNER=OWNER,
                                                                                             REPO=REPO,
                                                                                             FILE_PATH=FILE_PATH),
                   headers={'accept': 'application/vnd.github.v3.raw',
                            'authorization': 'token {}'.format(token)})

print('STATUS: Request Completed.\n')

# Alert User w/ Status Code, request success or failure (waterfall)
if(out.status_code != 200):
    raise Exception('ERROR: Status Code: ' + str(out.status_code) +
                    '\nUnable to retrieve {FPATH}, check parameters!'.format(FPATH=FILE_PATH))
else:
    print('STATUS: {FPATH} Successfully Retrieved!\n'.format(FPATH=FILE_PATH))

# Check if "remote to local" storage directory exists and save python file
if not os.path.exists(REM_TO_LOC_PATH):
    os.makedirs(REM_TO_LOC_PATH)
print(out.text, file=open(REM_TO_LOC_PATH + '/' + 'remote_' + FILE_PATH, 'w'))

_ = """
#######################################################################################################################
############################################   IDENTIFYING DEPENDENCIES   #############################################
#######################################################################################################################
"""

# Add specific folder to system path (to ensure that requirements.txt searches everywhere for dependencies)
sys.path.append(os.getcwd() + '/' + REM_TO_LOC_PATH)

# Any base packages need for run_remote.py to work
# NOTE: pipreqs is required to generate requirements.txt
_modules = ['pipreqs']
for mod in _modules:
    pip_install_confirm(mod, ver=False)

# Generate requirements.txt file
cmd_runprint('pipreqs --force')


_ = """
#######################################################################################################################
########################################   INSTALLING MISSING DEPENDENCIES   ##########################################
#######################################################################################################################
"""

# Determine modules already installed via pip
installed_mods = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip3 freeze', ret=True).split()}
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
        pip_install_confirm(r_mod, ver=False)
        print('STATUS: {module} installed successfully.'.format(module=r_mod))

print('\nSTATUS: Imported and Checked All Requirements.')
_ = """
#######################################################################################################################
#########################################   IMPORTING FILE.PY IN QUESTION   ###########################################
#######################################################################################################################
"""
# Import the necessary package
# NOTE: Exec statement is used since import statement should be at the top of the file (PEP)
exec('from remote_{FPATH} import *'.format(FPATH=FILE_PATH.replace('.py', '')))

papa_johns(pd.DataFrame([1, 2, 3]))

time.sleep(1)

shutil.rmtree(REM_TO_LOC_PATH)
os.remove('remote_shell_output.txt')
os.remove('requirements.txt')


# EOF

# EOF
