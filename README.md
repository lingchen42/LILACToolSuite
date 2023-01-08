## Installation

### Prerequisite
* Install Anaconda (https://docs.anaconda.com/anaconda/install/index.html)


### Install LILACToolSuite Web Application
* Download LILACToolSuite Code
```
git clone https://github.com/lingchen42/LILACToolSuite.git
```

* Install LILACToolSuite on your local computer through terminal
```
# cd into the source code
cd LILACToolSuite 

# create software enviroment for running LILACToolSuite
conda create -n lilacsuite python=3.10.6
conda activate lilacsuite
pip install -r requirements.txt
```

### Run LILACToolSuite on your local machine
```
# cd into the source code
cd LILACToolSuite/ui
flask run --port 5001
```
Got to `http://127.0.0.1:5001` to use the LILACToolSuite Web Application