import numpy as np
from PIL import Image

# Camera and viewport settings
O = np.array((0, 0, 0))  # Camera position
canvasWidth, canvasHeight = 500, 500
viewWidth, viewHeight = 1, 1
background_color = np.array((255, 255, 255))  # White background

class Sphere:
    def __init__(self, center, radius, color, specular):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular

class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
    def PutPixel(self, x, y, color):
        self.pixels[y, x] = np.clip(color, 0, 255)

class Light:
    def __init__(self, type, intensity, position=None):
        self.type = type
        self.intensity = intensity
        self.position = np.array(position) if position else None

def vecLength(vector):
    return np.sqrt(np.dot(vector, vector))

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

def ComputeLighting(P, N, V, s, lights):
    i = 0.0
    for light in lights:
        if light.type == "ambient":
            i += light.intensity
            continue
        else:
            if light.type == "point":
                L = light.position - P
            elif light.type == "directional":
                L = light.position
            else:
                continue
            L = L / vecLength(L)

            # Diffuse lighting
            n_dot_l = np.dot(N, L)
            if n_dot_l > 0:
                i += light.intensity * n_dot_l

            if s != -1:
                R = 2 * N * np.dot(N, L) - L
                r_dot_v = np.dot(R, V)
                if r_dot_v > 0:
                    i += light.intensity * pow(r_dot_v / (vecLength(R) * vecLength(V)), s)

    return i

def TraceRay(O, D, t_min, t_max, spheres, lights):
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

    # Compute intersection point and normal
    P = O + closest_t * D
    N = P - closest_sphere.center
    N = N / vecLength(N)

    # Compute lighting
    intensity = ComputeLighting(P, N, -D, closest_sphere.specular, lights)
    return closest_sphere.color * intensity

# Initialize canvas and spheres
canvas = Canvas(canvasWidth, canvasHeight)
spheres = [
    Sphere((0, 1, 3), 1, (255, 0, 0), 500),  # Red sphere
    Sphere((2, 0, 4), 1, (0, 0, 255), 500),  # Blue sphere
    Sphere((-2, 0, 4), 1, (0, 255, 0), 10),  # Green sphere
    Sphere((0, 5001, 0), 5000, (255, 255, 0), 1000)  # Yellow ground
]

lights = [
    Light(type="ambient", intensity=0.2),
    Light(type="point", intensity=0.6, position=(2, -1, 0)),
    Light(type="directional", intensity=0.2, position=(1, -4, 4))
]

# Render scene
for x in range(-canvasWidth // 2, canvasWidth // 2):
    for y in range(-canvasHeight // 2, canvasHeight // 2):
        D = CanvasToViewport(x, y)
        D = D / vecLength(D)
        color = TraceRay(O, D, 1, np.inf, spheres, lights)
        canvas.PutPixel(x + canvasWidth // 2, y + canvasHeight // 2, color)

img = Image.fromarray(canvas.pixels, 'RGB')
img.save("image.png")
