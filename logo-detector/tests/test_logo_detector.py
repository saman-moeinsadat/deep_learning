import pytest

from logo_detector import __version__ # pylint: disable=maybe-no-member


def test_version():
    assert __version__ == "0.1.0"

MILKA = "https://www.ocado.com/productImages/142/14286011_0_640x640.jpg?identifier=782aa512dd45c9d983216984442982a9"
# ADAC = "https://en.wikipedia.org/wiki/File:ADAC-Logo.svg"

@pytest.mark.parametrize("image_url, label", [(MILKA, "milka")])
def test_detect_resnet(image_url, label):
    from logo_detector.detector import detect_resnet
    assert detect_resnet(image_url) == label

def test_detect_cnn():
    from logo_detector.detector import detect_cnn
    image_url = "https://www.ocado.com/productImages/142/14286011_0_640x640.jpg?identifier=782aa512dd45c9d983216984442982a9"
    assert detect_cnn(image_url) == "fedex"

