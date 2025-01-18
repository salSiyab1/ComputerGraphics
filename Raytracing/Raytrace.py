import numpy as np
from PIL import Image
O = np.array((0, 0, 0))  # Camera position
canvasWidth, canvasHeight = 500, 500
viewWidth, viewHeight = 1, 1
background_color = (255, 255, 255)  # White background

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = color

class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)

    def PutPixel(self, x, y, color):
        self.pixels[y, x] = color

def CanvasToViewport(x, y):
    return np.array((x * viewWidth / canvasWidth, y * viewHeight / canvasHeight, 1))

def IntersectRaySphere(O, D, sphere):
    CO = O - sphere.center
    a = np.dot(D, D)
    b = 2 * np.dot(CO, D)
    c = np.dot(CO, CO) - sphere.radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return np.inf, np.inf

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return t1, t2

def TraceRay(O, D, t_min, t_max, spheres):
    closest_t = np.inf
    closest_sphere = None
    for sphere in spheres:
        t1, t2 = IntersectRaySphere(O, D, sphere)
        if t_min < t1 < t_max and t1 < closest_t:
            closest_t = t1
            closest_sphere = sphere
        if t_min < t2 < t_max and t2 < closest_t:
            closest_t = t2
            closest_sphere = sphere

    if closest_sphere is None:
        return background_color
    return closest_sphere.color

# Initialize canvas and spheres
canvas = Canvas(canvasWidth, canvasHeight)
spheres = [
    Sphere((0, 1, 3), 1, (255, 0, 0)),  # Red sphere
    Sphere((2, 0, 4), 1, (0, 0, 255)),  # Blue sphere
    Sphere((-2, 0, 4), 1, (0, 255, 0))  # Green sphere
]

# Render scene
for x in range(-canvasWidth // 2, canvasWidth // 2):
    for y in range(-canvasHeight // 2, canvasHeight // 2):
        D = CanvasToViewport(x, y)
        color = TraceRay(O, D, 1, np.inf, spheres)
        canvas.PutPixel(x + canvasWidth // 2, y + canvasHeight // 2, color)

img = Image.fromarray(canvas.pixels, 'RGB')
img.save("image.png")

