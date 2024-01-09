import cv2
import json
import os 

# scaling points relative to 512 * 512 
def scale_points(points, image_width, image_height, scaled_image_width, scaled_image_height):
    scaled_points = [(float(point[0] / image_width * scaled_image_width),
                      float(point[1] / image_height * scaled_image_height)) for point in points]
    return scaled_points

def draw_bounding_boxes(image_path, annotations):
    # Read the image
    image = cv2.imread(image_path)

    # Loop through shapes in annotations and draw bounding boxes
    for shape in annotations['shapes']:
        label = shape['label']
        points = shape['points']

        # Get image dimensions from JSON
        image_height = annotations['imageHeight']
        image_width = annotations['imageWidth']

        scaled_image_width , scaled_image_height = 512 , 512

        # Resize the image to match the dimensions from JSON
        image = cv2.resize(image, (image_width, image_height))

        # Convert points to integers
        points = [(point[0], point[1]) for point in points]

        points = scale_points(points,image_width,image_height,scaled_image_width,scaled_image_height)
        image = cv2.resize(image, (scaled_image_width, scaled_image_height))

        # Draw a rectangle on the image
        cv2.rectangle(image, points[0], points[1], (0, 255, 0), 1)
        cv2.putText(image, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the image with bounding boxes
    if image is None:
        print(f"Error reading the image from {image_path}")
        exit()

    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale_json_values(json_data):
    image_width = json_data['imageWidth']
    image_height = json_data['imageHeight']

    scaled_image_width = 512 
    scaled_image_height = 512

    for shape in json_data['shapes']:
        points = shape['points']
        scaled_points = scale_points(points, image_width, image_height, scaled_image_width, scaled_image_height)
        shape['points'] = scaled_points

    json_data['imageWidth'] = scaled_image_width
    json_data['imageHeight'] = scaled_image_height

    return json_data

def resize_image(image_path):
    scaled_image_width = 512
    scaled_image_height = 512

    image = cv2.imread(image_path)
    image = cv2.resize(image, (scaled_image_width, scaled_image_height))

    cv2.imwrite(image_path,image)

def save_new_json(json_path):
    with open(json_path,'r') as f:
        data = json.load(f)

    with open(json_path,'w') as f:
        f.write(json.dumps(scale_json_values(data)))

def process_images_in_folder(folder_path):

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG"):
            image_path = os.path.join(folder_path, filename)

            # Check if there is a corresponding JSON file
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(folder_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    annotations = json.load(json_file)

                #draw_bounding_boxes(image_path, annotations)
                    
                resize_image(image_path)
                save_new_json(json_path)
                              
            else:
                print(f"No JSON file found for {filename} where json needed is {json_filename}")

if __name__ == "__main__":
    # Load JSON data

    file_folder_path = 'file'
    process_images_in_folder(file_folder_path)


    
    