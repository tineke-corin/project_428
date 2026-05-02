### COSC428 Computer Vision project

## Image anonymiser for faces and vehicle licence plates

To run the demo:

* Ensure you have Python >= 3.12.4 installed
* Ensure you have Poetry >= 1.8.2 installed
* Run `poetry install` to install depenencies
* You will need pre-trained model weights for the two detection models. Place these in a `models` directory in the project root. They should be called `yolov11l-face.pt` and `licence_plate_detector.pt`.
* The face model weights used for the project can be found at https://huggingface.co/AlekseyKorshuk/yolov11-face/tree/main
* The licence plate model weights used for the project can be found at https://github.com/bhaskrr/number-plate-recognition-using-yolov11/tree/main
* Create a local folder of test images (png, jpg or jpeg)
* Run the demo with `poetry run demo --directory=<your_local_image_dir>`
