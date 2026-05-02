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

## Run the API with Docker

1. You will need pre-trained model weights for the two detection models. Place these in a `models` directory in the project root. They should be called `yolov11l-face.pt` and `licence_plate_detector.pt`. Make sure these are in place before building the docker image.

2. Build the Docker image:
   ```bash
   docker build -t anonymiser-api .
   ```
This can take a while. Go make a cuppa while you wait :coffee:

2. Run the container:
   ```bash
   docker run -p 8000:8000 anonymiser-api
   ```

The API will be available at `http://localhost:8000/api/v1/images/anonymize`

To test the API with `cUrl`:
```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/api/v1/images/anonymize --output anonymised.jpg
```

To test from Postman, specify the body as form-data and ensure the value type is set to 'File'.

![Postman Example](postman-example.png)

