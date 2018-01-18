# Google ML engine example model

Code comes with bash script `train.sh` which manages all the work with google cloud

Presentation: https://docs.google.com/presentation/d/14ZCqCpbwKfoatNhfYfw8iMhHPQjrbmNd_XAdVzJEHCE/edit?usp=sharing

# Install

Setup uses virtual env to handle dependencies

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

# Run

You need to be in virtualenv

Running the script requires google sdk: https://cloud.google.com/sdk/docs/quickstart-macos

Authenticate

```
gcloud auth login
```

Set default project

```
gcloud config set project example_project
```

Then you run first time, you will be promted to allow APIs for the project.

```
source env/bin/activate
bash train.sh
```
