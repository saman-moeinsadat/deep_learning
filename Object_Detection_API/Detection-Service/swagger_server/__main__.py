#!/usr/bin/env python3

import connexion

from swagger_server import encoder
import logging


def main():
    # loading the YAML file and initialize the api
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Object Detection api(prototype)'})
    app.run(port=8080)


if __name__ == '__main__':
    # initializing the logger
    logging.basicConfig(
        filename='detection_service.log', filemode='a',
        format='%(asctime)s-%(funcName)s-%(threadName)s-%(levelname)s-%(message)s',
        level=logging.INFO
    )
    main()
