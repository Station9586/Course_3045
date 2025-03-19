import numpy as np
import cv2

# setting the size of the canvas
height, width = 600, 800
num_stars = 100

# generating the positions and radius of the stars
stars_x = np.random.randint(0, width, num_stars)
stars_y = np.random.randint(0, height, num_stars)
stars_radius = np.random.randint(3, 7, num_stars) # random radius between 3 and 6

while True: 
    # creating a black canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate the brightness of the stars (100 ~ 255)
    brightness = np.random.randint(100, 256)

    # Draw the stars
    for i in range(num_stars): 
        cv2.circle(canvas, (int(stars_x[i]), int(stars_y[i])), int(stars_radius[i]), (brightness, brightness, brightness), -1)

    cv2.imshow('Canvas', canvas)

    # press 'ESC' to exit
    if cv2.waitKey(100) == 27: break

cv2.destroyAllWindows()