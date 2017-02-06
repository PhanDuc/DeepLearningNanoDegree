# Anaconda package manager

**Install Anaconda first**

0. you originally at the default conda environment
	- you can see all conda list by `conda list`
	- you can upgrade conda list by `conda upgrade --all`

1. install packages
	- e.g.`conda install numpy scipy pandas`
	- with version number `conda install numpy=1.0`
	- conda will automatically install dependencies! 
2. uninstall packages
	- `conda remove package_name`

3. update packages
	- `conda update package_name`
	-	`conda update --all`

# Anaconda setup an environment

1.Create An Environment: `conda create -n env_name list of packages`
	- e.g. `conda create -n tea_facts python=3` 
	# create a new environment
2.Entering An Environment: `source activate my_env`
	- e.g. `source activate tea_facts`   
	# activate the new environment
3. `conda list`
	# see all the available packages in the current environment
4. `conda install numpy pandas matplotlib`
	# install packages
5. `conda install jupyter notebook`
	# install notebooks for 
6. now we have setup a new environment
7. Leave an Environment: `source deactivate`

# Save and Load Anacoda Environments

1. save an environment to **YAML** file: `conda env export >environment.yaml`
2. create an environment from a YAML file: `conda env create -f environment.yaml`
3. Default environment: `root`
4. Remove Environments: `conda env remove -n env_name`

# Recommended Practices

1. Create two environments for python 2 and python 3 respectively:
	- `conda create -n py2 python=2`
	- `conda create -n py3 python=3`
2. keep a **YAML** configuration file in the repository

# jupytr notebook Magic keywords

## Timing Code

`%timeit + <python code>`
- `%timeit fibol(20)`

`%%timeit` (at the beginning of the cell)
- get the running time of a cell

## Visualization

`%matplotlib inline`
- rendering inline plot

`%config InlineBackend.figure_format = 'retina'`
- after the line above, configure for retina display

## Debugging

`%pdb`
- if problem, will allow debugging
- [documentation of pdf](https://docs.python.org/3/library/pdb.html)


# Resources

- [Numpy Official Tutorial]()
- [Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)
- [Scipy Tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
- [10 minutes Pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html#min)
- [Scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)
- [matplotlib](http://matplotlib.org/users/pyplot_tutorial.html)