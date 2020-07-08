# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.request_parameters import RequestParameters  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_detection_post(self):
        """Test case for detection_post

        Detection of Desired Logos.
        """
        requestParameters = RequestParameters()
        response = self.client.open(
            '/detection',
            method='POST',
            data=json.dumps(requestParameters),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
