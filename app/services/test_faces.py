from app.api.services.faces import contains_people, detect_faces_retinaface
from pathlib import Path, PosixPath

test_images_dir = Path(__file__).parent.parent.parent.parent

# TODO image 2008_000562_jpg.rf.fa04eee65f83d9df5ca50ef0be633a17.jpg (tandem) does not
# detect the people

# TODO 2007_000346_jpg.rf.3869abb347c0b1e71d40ed0b7c32237b.jpg does not detect the person
 
# some data is from:
# https://www.kaggle.com/datasets/sylshaw/streetview-by-country (NZ subset)
#

def is_jpg(filename: PosixPath) -> bool:
    return (filename.suffix == '.jpg')

"""
   People finding 
"""
def test_finds_people():
    # For images in the 'test_images/with_people' folder, make sure that finds_people
    # returns True
    image_folder = Path(test_images_dir / "test_images" / "with_people")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            has_people = contains_people(file_path)
            assert has_people == True

def test_does_not_find_people():
    # For images in the 'test_images/without_people' folder, make sure that finds_people
    # returns False
    image_folder = Path(test_images_dir / "test_images" / "without_people")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            has_people = contains_people(file_path)
            assert has_people == False

"""
   Face finding 
"""
def test_detects_one_face():
    # For images in the 'test_images/with_people/one_face' folder, make sure that detect_faces
    # returns one face detection
    image_folder = Path(test_images_dir / "test_images" / "one_face")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            detections = detect_faces_retinaface(file_path)
            assert len(detections) == 1
    return

def test_detects_many_faces():
    # For images in the 'test_images/with_people/many_faces' folder, make sure that detect_faces
    # returns multiple face detections
    image_folder = Path(test_images_dir / "test_images" / "many_faces")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            detections = detect_faces_retinaface(file_path)
            assert len(detections) > 1

def test_detects_no_faces():
    # For images in the 'test_images/with_people/no_faces' folder, make sure that detect_faces
    # returns no face detections
    image_folder = Path(test_images_dir / "test_images" / "no_faces")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            detections = detect_faces_retinaface(file_path)
            assert len(detections) == 0
