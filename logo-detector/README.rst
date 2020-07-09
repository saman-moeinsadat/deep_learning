# Logo Detector
This app gets an input in the form of url_path to an image and returns the predicted logo

## Setup
Build and run a local python environment:
```
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
source $HOME/.poetry/env
poetry install
```

## Test
To run the tests
source /home/saman/.cache/pypoetry/virtualenvs/logo-detector-py3.6```
pytest
```

## Run Locally
To run the flask app localy run the following
```
source /home/saman/.cache/pypoetry/virtualenvs/logo-detector-py3.6/bin/activate
python app.py
```

Open your browser and go to [your localhost](http://localhost:8000)
Copy the url for the image containing log and enter it into the text field and press submit.
