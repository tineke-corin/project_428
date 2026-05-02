from pathlib import Path, PosixPath
from app.api.services.vlps import contains_vehicles

test_images_dir = Path(__file__).parent.parent.parent.parent

def is_jpg(filename: PosixPath) -> bool:
    return (filename.suffix == '.jpg')

"""
   Vehicle finding 
"""
def test_contains_vehicles():
    # For images in the 'test_images/with_vehicles' folder, make sure that contains_vehicles
    # returns True
    image_folder = Path(test_images_dir / "test_images" / "with_vehicles")
    for file_path in image_folder.iterdir():
        if file_path.is_file() and is_jpg(file_path):
            has_vehicles = contains_vehicles(file_path)
            assert has_vehicles == True

