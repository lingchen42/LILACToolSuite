## App Preview

![preview](https://github.com/lingchen42/LILACToolSuite/blob/main/assets/AppPreview.png)

## Installation

### Prerequisite
* Install Miniconda (https://docs.conda.io/en/latest/miniconda.html)


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


## FAQ

### Errors
1. On Mac, you may run into `invalid active developer path .. missing xcrun`, in this case, you might need to install `XCODE`. Try running this in your terminal to install it.

```
xcode-select --install
```
If this fails, download xcode from Apply official website:https://developer.apple.com/xcode/


### Production deployment with gunicorn
The default app running method (`flask run`) is good enough for a few users. But if you have multiple users, you may consider to use `gunicorn` to deploy this web app.
```
cd LILACToolSuite/ui
gunicorn -w 4 'app:app'
```


## Cite this work
* APA
```
Chen, L., & Su, P. (2023). LILAC Lab Tool Suites (Version 1.0.0) [Computer software]. https://github.com/lingchen42/LILACToolSuite
```

* BibTex
```
@software{Chen_LILAC_Lab_Tool_2023,
author = {Chen, Ling and Su, Pumpki},
month = {1},
title = {{LILAC Lab Tool Suites}},
url = {https://github.com/lingchen42/LILACToolSuite},
version = {1.0.0},
year = {2023}
}
```
