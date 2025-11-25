First run 'poetry shell' to activate the venv and 'poetry install' to install the packages.
Then run 'python3 main.py -m DATA.N=20,40 non_equivariance=0,1,2'

To install new packages within the poetry venv and automatically update the toml:
1. activate the venv with 'poetry shell'
2. run 'poetry add package_name'

To install new packages within the poetry venv without updating the toml:
1. activate the venv with 'poetry shell'
2. run 'pip install package_name'
