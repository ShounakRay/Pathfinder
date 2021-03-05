# @Author: Shounak Ray <Ray>
# @Date:   05-Mar-2021 10:03:92:926  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: run_remote.py
# @Last modified by:   Ray
# @Last modified time: 05-Mar-2021 15:03:76:760  GMT-0700
# @License: [Private IP]

import os
import subprocess
import sys

import requests

# define parameters for a request
OWNER = 'ShounakRay'
REPO = 'AnomalyDetection'
BRANCH = 'main'
REPO_TYPE = 'private'
FILE_PATH = 'Anomaly_Detection_PKG.py'
REM_TO_LOC_PATH = 'ReferencePY'
GITHUB_TOKEN = token = open('Access Tokens/{owner}_TOKEN.txt'.format(owner=OWNER)).read().strip()


def _github(url: str, mode: str = "private", token: str = GITHUB_TOKEN):
    if mode == "public":
        return requests.get(url)
    else:
        headers = {'accept': 'application/vnd.github.v3.raw',
                   'authorization': 'token {}'.format(token)}
        return requests.get(url, headers=headers)


def cmd_runprint(command, prnt_file=True, prnt_scrn=False, ret=False):
    exec_output = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    if(prnt_file):
        print(exec_output, file=open('remote_shell_output.txt', 'w'))
    if(prnt_scrn):
        print(exec_output)
    if(ret):
        return exec_output


out = _github('https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{FILE_PATH}'.format(OWNER=OWNER,
                                                                                             REPO=REPO,
                                                                                             BRANCH=BRANCH,
                                                                                             FILE_PATH=FILE_PATH),
              mode=REPO_TYPE)
print('STATUS: Request Completed.')

# Alert User w/ Status Code
if(out.status_code != 200):
    raise Exception('ERROR: Status Code: ' + str(out.status_code) +
                    '\nUnable to retrieve {FPATH}, check parameters!'.format(FPATH=FILE_PATH))
else:
    print('STATUS: {FPATH} Successfully Retrieved!'.format(FPATH=FILE_PATH))

# Check if remote to local directory exists and save python file
if not os.path.exists(REM_TO_LOC_PATH):
    os.makedirs(REM_TO_LOC_PATH)
print(out.text, file=open(REM_TO_LOC_PATH + '/' + 'remote_' + FILE_PATH, 'w'))

# Use this when uploaded to DSP
# sys.path.append('/ReferencePY')

sys.path.append(os.getcwd() + '/' + REM_TO_LOC_PATH)

_modules = ['pipreqs']
for mod in _modules:
    if(mod not in sys.modules):
        cmd_runprint('pip3 install {module}'.format(module=mod), prnt_file=False)
        pip_installed = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze',
                                                                                        ret=True).split()}
        if(mod not in pip_installed):
            raise ValueError('ERROR: {module} not installed properly by pip'.format(module=mod))

# [x for x in sys.modules if 'pi' in x]

cmd_runprint('pipreqs')

installed_mods = {mod.split("==")[0]: mod.split("==")[1] for mod in cmd_runprint('pip freeze', ret=True).split()}
required_mods = {mod.split("==")[0]: mod.split("==")[1] for mod in open('requirements.txt').read().split()}

for r_mod in required_mods:
    if(r_mod in installed_mods):
        print('STATUS: {module} already installed. Version {version}'.format(module=r_mod,
                                                                             version=required_mods.get(r_mod)))
    else:  # Required module not installed, install it
        print('> STATUS: {module} not installed. Version {version} required.'.format(module=r_mod,
                                                                                     version=required_mods.get(r_mod)))
        cmd_runprint('pip3 install {module}'.format(module=r_mod))
        print('STATUS: {module} installed successfully.'.format(module=r_mod))

# Exec statement is used since import statement should be at the top of the file (PEP)
exec('from remote_{FPATH} import *'.format(FPATH=FILE_PATH.replace('.py', '')))

# EOF

# EOF
