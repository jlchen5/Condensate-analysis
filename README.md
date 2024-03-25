# Condensate-analysis

This project utilizes Python and OpenCV to detect circular objects within an image. It can be particularly useful in scenarios like biological imaging where circular features such as cells are common.

## Prerequisites
Before running this project, ensure you have the following installed:
~~~
Python 3.x
OpenCV (cv2)
NumPy
Pillow (PIL)
Matplotlib
~~~
You can install the necessary libraries using pip:
~~~
pip install numpy opencv-python Pillow matplotlib
~~~

## Usage
To use this script, replace the image_path variable with the path to your image. The script performs the following operations:

- Loads the image.
- Converts the image to grayscale.
- Applies Gaussian blur to reduce noise.
- Performs Hough Circle Transform to detect circles.
- Draws the detected circles and their centers on the original image.
- Displays the final image with the detected circles highlighted.

## Example
Below is an example of how to execute the script:
~~~
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = 'path_to_your_image.tif'  
image = Image.open(image_path)
image_np = np.array(image)

# Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=5, minDist=20,
                           param1=30, param2=30, minRadius=10, maxRadius=20)

# Draw circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    output = image_np.copy()
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (255, 0, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Show the result
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Circles: {len(circles) if circles is not None else 0}')
plt.axis('off')
plt.show()

~~~
![image](https://github.com/jlchen5/Condensate-analysis/blob/main/pics/urea-0.1M-120min%20copy.png)

## Contributing
Feel free to fork this project and submit a pull request if you have suggestions for improvements or new features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


