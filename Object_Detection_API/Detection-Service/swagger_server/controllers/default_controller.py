import connexion
import six
import json
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.request_parameters import RequestParameters  # noqa: E501
from swagger_server import util
from Object_Detection_API.Library.detect import *
from Object_Detection_API.Library.classification_network.detector import *
import base64
import time
from swagger_server import encoder
import io
import cv2
from imageio import imread
import logging
import os


def detection_post(requestParameters):  # noqa: E501
    """Detection of Desired Logos.

    Returns the Coordinates and Confidance of Each Desired Logo in an Image. # noqa: E501

    :param requestParameters: # Parameters:   * imageData:      The image to be proccessed   * mode:      The modes of proccess:       1. just classification.        2. just detection.       3. classification and detection.       4. detection just if positive classification.   * nmsThres:     The threshold for NMS.   * confThres:     The threshold of confidence.
    :type requestParameters: dict | bytes

    :rtype: InlineResponse200
    """
    # Load request parameters in request-dictionary
    t0_method = time.time()
    request_dict = json.loads(requestParameters)
    # Check the validity of parameters
    if 'networkID' not in request_dict:
        logging.warning("Bad Request, The Request Must Contain 'networkID' Parameter.")
        return "Bad Request, The Request Must Contain valid 'networkID' Parameter. ", 400
    if os.path.isfile(DetectionNetwork.return_config()['weights']+str(request_dict['networkID'])+'_det.pt'):
        network_id = request_dict['networkID']
    else:
        logging.warning("Bad Request, the network_id is not valid")
        return "Bad Request, the network_id is not valid", 400
    if 'imageData' not in request_dict or not isinstance(request_dict['imageData'], list):
        logging.warning('Bad Request, No Image Data Provided (network_id = %s).' % request_dict['networkID'])
        return "Bad Request, No Image Data Provided", 400
    # Check image data for validity
    try:
        image0s = [
            cv2.cvtColor(imread(io.BytesIO(base64.b64decode(img64))), cv2.COLOR_RGB2BGR) for
            img64 in request_dict['imageData']
        ]
    except ValueError as e:
        logging.warning('Bad Request, Atleast One of Image Data provided, is Corrupted (network_id = %s).' % request_dict['networkID'], exc_info=True)
        return 'Bad Request, Atleast One of Image Data provided, is Corrupted', 400
    if 'mode' not in request_dict:
        logging.warning("Bad Request, The Request Must Contain 'mode' Parameter (network_id = %s)." % request_dict['networkID'])
        return "Bad Request, The Request Must Contain 'mode' Parameter. ", 400
    mode = request_dict['mode']
    # Fetch customized NMS and confidance thresholds
    nmst = DetectionNetwork.return_config()['nmst']
    conft = DetectionNetwork.return_config()['conft']
    logging.info("Detection method with mode: %s and Network_id: %s requested." % (mode, network_id))
    if 'nmsThres' in request_dict:
        nmst = request_dict['nmsThres']
    if 'confThres' in request_dict:
        conft = request_dict['confThres']
    if mode == '1':
        try:
            # Just classification is performed: Resnet network
            t0_1 = time.time()
            model_cls = DetectionNetwork.customized_cls(network_id)
            results_class = detect_resnet(model_cls, image0s, network_id)
            t_1_cls = time.time() - t0_1
            result_batch = []
            for rslt in results_class:
                result_per_image = {}
                result_per_image["classification_performed"] = True
                result_per_image["classification"] = rslt
                result_per_image["detection_performed"] = False
                result_per_image["detection"] = []
                result_batch.append(result_per_image)

            results = {
                "classification_network_time": round(float(t_1_cls), 3), "detection_network_time": 0,
                "detection_post_proccessing_time": 0, 'dtc_network_loading_time': 0,
                "results_batch": result_batch
            }
        except Exception as e:
            logging.error("An Unexpected Error happened. ", exc_info=True)
            return 'An Unexpected Error happened.', 500

    elif mode == '2':
        try:
            # Just Detection is performed, YOLO object decetion network version 3
            t0_2 = time.time()
            model = DetectionNetwork.customized_network(network_id)
            t2_load = time.time()
            detection_results = logo_detection(model, image0s, network_id, nmsThres=nmst, confThres=conft)
            t_2_det = time.time() - t2_load
            detection_time_pp = detection_results["post_processing_time"]
            detections = detection_results["detection_results_batch"]
            result_batch = []
            for dct in detections:
                result_per_image = {}
                result_per_image["classification_performed"] = False
                result_per_image["classification"] = ''
                result_per_image["detection_performed"] = True
                result_per_image["detection"] = dct
                result_batch.append(result_per_image)
            results = {
                "classification_network_time": 0, "detection_network_time": round(float(t_2_det - detection_time_pp), 3),
                "detection_post_proccessing_time": round(float(detection_time_pp), 3), 'dtc_network_loading_time': round(float(t2_load - t0_2), 3),
                "results_batch": result_batch
            }
        except Exception as e:
            logging.error("An Unexpected Error happened. ", exc_info=True)
            return 'An Unexpected Error happened.', 500
    elif mode == '3':
        try:
            # Both classification and detection is performed
            t0_3 = time.time()
            model_cls = DetectionNetwork.customized_cls(network_id)
            results_class = detect_resnet(model_cls, image0s, network_id)
            t_3_cls = time.time() - t0_3
            t0_3_dtc = time.time()
            model = DetectionNetwork.customized_network(network_id)
            t3_load = time.time()
            detection_results = logo_detection(model, image0s, network_id, nmsThres=nmst, confThres=conft)
            t_3_det = time.time() - t3_load
            detection_time_pp = detection_results["post_processing_time"]
            detections = detection_results["detection_results_batch"]
            result_batch = []
            for idx in range(len(results_class)):
                result_per_image = {}
                result_per_image["classification_performed"] = True
                result_per_image["classification"] = results_class[idx]
                result_per_image["detection_performed"] = True
                result_per_image["detection"] = detections[idx]
                result_batch.append(result_per_image)
            results = {
                "classification_network_time": round(float(t_3_cls), 3), "detection_network_time": round(float(t_3_det - detection_time_pp), 3),
                "detection_post_proccessing_time": round(float(detection_time_pp), 3), 'dtc_network_loading_time': round(float(t3_load - t0_3_dtc), 3),
                "results_batch": result_batch
            }
        except Exception as e:
            logging.error("An Unexpected Error happened. ", exc_info=True)
            return 'An Unexpected Error happened.', 500
    elif mode == '4':
        try:
            # Detection is performed just for positive classification
            t0_4_cls = time.time()
            model_cls = DetectionNetwork.customized_cls(network_id)
            results_class = detect_resnet(model_cls, image0s, network_id)
            t_4_cls = time.time() - t0_4_cls
            idx_nologos = [] # fetching indices for nologos label
            image0s_copy = []
            for i, det in enumerate(results_class):
                if det == 'no_logos':
                    idx_nologos.append(i)
                else:
                    image0s_copy.append(image0s[i])
            t0_4_det = time.time()
            model = DetectionNetwork.customized_network(network_id)
            t4_load = time.time()
            detection_results = logo_detection(model, image0s_copy, network_id, nmsThres=nmst, confThres=conft)
            t_4_det = time.time() - t4_load
            for idx in sorted(idx_nologos, reverse=False):
                # Adding not-performed label for images, haven't been processed with detection network
                detection_results["detection_results_batch"].insert(idx, [])
            detection_time_pp = detection_results["post_processing_time"]
            detections = detection_results["detection_results_batch"]
            result_batch = []
            for idx in range(len(results_class)):
                result_per_image = {}
                result_per_image["classification_performed"] = True
                result_per_image["classification"] = results_class[idx]
                if results_class[idx] == 'no_logos':
                    result_per_image["detection_performed"] = False
                else:
                    result_per_image["detection_performed"] = True
                result_per_image["detection"] = detections[idx]
                result_batch.append(result_per_image)
            results = {
                "classification_network_time": round(float(t_4_cls), 3), "detection_network_time": round(float(t_4_det - detection_time_pp), 3),
                "detection_post_proccessing_time": round(float(detection_time_pp), 3), 'dtc_network_loading_time': round(float(t4_load - t0_4_det), 3),
                "results_batch": result_batch
            }
        except Exception as e:
            logging.error("An Unexpected Error happened. ", exc_info=True)
            return 'An Unexpected Error happened.', 500
    else:
        logging.warning("Bad Request, Invalid Mode, Mode Must Be One Of The 1, 2, 3 or 4 Integers (network_id = %s)." % network_id)
        return 'Bad Request, Invalid Mode, Mode Must Be One Of The 1, 2, 3 or 4 Integers.', 400
    logging.info("Detection request is completed (network_id = %s)." % network_id)
    # dump results to json object
    if results:
        results['method_total_time'] = round(float(time.time() - t0_method), 3)
        return json.dumps(results), 200
    else:
        return "An Unexpected Error happened.", 500
