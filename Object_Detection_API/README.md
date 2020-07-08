# Object Detection API

The Object Detection API enables the user to train a neural network, which can detect objects in images. The user can train the network and query images for objects.

Every user can train their own network(s) for a limited amount of object classes.

The API is splitted in 3 parts.

## Detection Service

The only task of the detection service is to find objects in a given image with a given network defined by the network-id. The neural network will be loaded in a lazy manner. This service is not public and will be used by the Proxy Service.

####  Provided Methods:
- **POST /networks/{network_id}/detect(image)**

	Executes the object detection on the given image with the given network.

	*Parameters*

	image : Binary or base64 representation of the image under detection\
network_id	: Id of the network to use for the detection

	*Returns*

	A Json-Object containing all detected objects with its bounding boxes plus some meta informations

## Training Service

The networks used by the Detection Service are trained by the service. For this a bulk of images is uploaded. When enough images for a training are in the system the training is triggered. After the training is finished the network can be used for detection. Every new training will generate a unique network-id which can be used in the Detection Service.  This service is not public and will be used by the Proxy Service

####  Provided Methods:

- **POST /trainingset/add_image(image, labeled_boxes[])**

	Adds an image to the current training set.

	*Parameters*

	image : Binary or base64 representation of a training image\
labeled_boxes	: An array of labels with its bounding boxes

	*Returns*

	A unique image id for that image

- **DELETE /trainingset/{image_id}**

	Deletes an image from the current training set

	*Parameters*

	image_id: id of the image, which should be deleted from the training set

- **POST /networks/train**

	Trains a new network with all current images in the training set. 

	*Returns*

	A unique network-id for the new network

- **GET /networks/{network-id}/get_training_state**

	Returns infos about the current training state. This will be used to make sure the training of the network has finished.

	*Parameters*

	network-id: Id of the network to get the training status of

	*Returns*

	The current training progress

- **POST /networks/{network-id}/_stop_training**

	The training can be stopped early, if the learning curve is not satisfying or will probably not improve anymore.

	*Parameters*

	network-id: Id of the network currently being trained

	*Returns*

	The current training state

 - **DELETE /networks/{network-id}/delete_network**

	Older network versions or networks resulting from unsatisfying trainings can be deleted.

	*Parameters*

	network-id: Id of the network to be deleted


## Proxy Service

All client requests will be handled by this service. Including labeling/adding/removing images, training and deleting networks and administrative tasks. The service is visible to the internet.

####  Provided Methods:

- **POST /networks/{network_id}/detect(image)**

	Executes the object detection on the given image with the given network. This calls the Detection Service in backend. 

	*Security*

	API_key

	*Parameters*

	image : Binary or base64 representation of the image under detection\
	network_id	: Id of the network to use for the detection

	*Returns*

	A Json-Object containing all detected object with its bounding boxes plus some meta informations


- **POST /labels/(label_name)**

	Creates a new object class with the given label name.

	*Security*

	API_key

	*Parameters*

	class_id: Name of the object class to detect (e.g. ADAC-Logo)

- **POST /trainingset/images(image, labeled_boxes[])**

	Adds an image to the current training set. This calls the add_image method of the Training Service in backend.

	*Security*

	API_key

	*Parameters*

	image : Binary or base64 representation of the training image\
labeled_boxes : (optional) An array of labels with its bounding boxes. If labeled_boxes is not provided, the user must add labels with the label method in order to add this image to the training set. Only labels previously created with the /labels/(label_name) method will be accepted.

	*Returns*
A unique image id for that image


- **GET /trainingset/images/unlabeled**

	Returns a list of image ids, which are unlabeled.

	*Security*

	API_key

	*Returns*

	A list of image ids

- **GET /trainingset/images/{image_id}**

	Returns the image data of the training image with the given id.

	*Security*

	API_key

	*Parameters*

	image_id: The id of the training image to be returned

	*Returns*

	The image data for the given id


- **POST /trainingset/images/{image_id}/labels(labeled_boxes[])**

	Adds labeled bounding boxes for objects in a given training image.

	*Security*

	API_key

	*Parameters*

	image_id: The id of the training image to be labeled\
labeled_boxes: An array of labels with its bounding boxes. Only labels previously created with the /labels/(label_name) method will be accepted. 


- **DELETE /trainingset/images/{image_id}**

	Deletes an image from the current training set. This calls the delete_image method of the Training Service in backend.

	*Security*

	API_key

	*Parameters*

	image_id: id of the image which should be deleted from the training set

- **POST /networks/train**

	Trains a new network with all current images in the training set. Calls the train method of Training Service in backend.

	*Security*

	API_key

	*Returns*

	A unique network-id for the new network

- **GET /networks/{network_id}/training_state**

	Returns infos about the current training state. This will be used to make sure the training of the network has finished. This calls the get_training_state method of Training Service in backend

	*Security*

	API_key

	*Parameters*

	network-id: Id of the network to get the training state of

	*Returns*

	The current training progress


- **POST /networks/{network_id}/stop_training**

	The training can be stopped early, if the learning curve is not satisfying or will probably not improve anymore. This calls the stop_training method of Training Service in backend

	*Parameters*

	network_id: Id of the network currently being trained

	*Returns*

	The current training state

- **DELETE /networks/{network_id}**

	Older network versions or networks resulting from unsatisfying trainings can be deleted. This calls the delete_network method of Training Service in backend

	*Parameters*

	network-id: Id of the network to be deleted



