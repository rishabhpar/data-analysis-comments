import flask
import pytest

import main


# Create a fake "app" for generating test request contexts.
@pytest.fixture(scope="module")
def app():
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    return app


def test_hello_get(app):
    with app.test_request_context(json={"comment_text": "I fucking hate data science."}):
        res = main.predict_comment(flask.request)
        assert type(res) is str
        print("\n\nResult: ", res)

