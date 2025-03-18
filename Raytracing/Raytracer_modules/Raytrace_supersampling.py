from multiprocessing.pool import Pool
import numpy as np
from PIL import Image
import time

# Camera and viewport settings
O = np.array((0, 0, 0))  # Camera position
canvasWidth, canvasHeight = 500, 500
viewWidth, viewHeight = 1, 1
# background_color = np.array((255, 255, 255))  # White background
background_color = np.array((0, 0, 0))  # Black background

# Supersampling factor (e.g., 2 means 2x2 = 4 samples per pixel)
supersample_factor = 2

class Sphere:
    def __init__(self, center, radius, color, specular, reflective):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

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

def compute_pixel(args):
    x, y = args
    offset= 0.5
    color_sum = np.zeros(3, dtype=np.float64)
    # For supersampling, we take several samples inside each pixel.
    # The pixel in canvas coordinates has a width and height of 1.
    # We add an offset within [-0.5, 0.5) in both directions.
    for i in range(supersample_factor):
        for j in range(supersample_factor):
            # Compute a uniform grid of subpixel offsets
            offset_x = (i + offset) / supersample_factor - offset
            offset_y = (j + offset) / supersample_factor - offset
            sample_x = x + offset_x
            sample_y = y + offset_y

            D = CanvasToViewport(sample_x, sample_y)
            D = D / vecLength(D)
            sample_color = TraceRay(O, D, 1, np.inf, spheres, lights, 3)
            color_sum += sample_color
    avg_color = color_sum / (supersample_factor ** 2)
    # Convert canvas coordinates (centered) to array indices.
    return x + canvasWidth // 2, y + canvasHeight // 2, avg_color

def vecLength(vector):
    return np.sqrt(np.dot(vector, vector))

def CanvasToViewport(x, y):
    # Map the canvas (which is centered) to the viewport.
    return np.array((x * viewWidth / canvasWidth, y * viewHeight / canvasHeight, 1))

def ReflectRay(R, N):
    return 2 * N * np.dot(N, R) - R

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

def ClosestIntersection(O, D, t_min, t_max, spheres):
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

    return closest_sphere, closest_t

def ComputeLighting(P, N, V, s, lights):
    i = 0.0
    for light in lights:
        if light.type == "ambient":
            i += light.intensity
            continue
        else:
            if light.type == "point":
                L = light.position - P
                t_max = 1
            elif light.type == "directional":
                L = light.position
                t_max = np.inf
            else:
                continue
            L = L / vecLength(L)

            # Shadow check: if any object blocks the light.
            shadow_sphere, shadow_t = ClosestIntersection(P, L, 0.001, t_max, spheres)
            if shadow_sphere is not None:
                continue

            # Diffuse lighting
            n_dot_l = np.dot(N, L)
            if n_dot_l > 0:
                i += light.intensity * n_dot_l

            # Specular lighting
            if s != -1:
                R = ReflectRay(N, L)
                r_dot_v = np.dot(R, V)
                if r_dot_v > 0:
                    i += light.intensity * pow(r_dot_v / (vecLength(R) * vecLength(V)), s)

    return i

def TraceRay(O, D, t_min, t_max, spheres, lights, recursion_depth):
    closest_sphere, closest_t = ClosestIntersection(O, D, t_min, t_max, spheres)
    if closest_sphere is None:  # No sphere intersected
        return background_color

    # Compute intersection point and normal.
    P = O + closest_t * D
    N = P - closest_sphere.center
    N = N / vecLength(N)
    local_color = closest_sphere.color * ComputeLighting(P, N, -D, closest_sphere.specular, lights)

    r = closest_sphere.reflective
    if recursion_depth <= 0 or r <= 0:
        return local_color

    R = ReflectRay(-D, N)
    reflected_color = TraceRay(P, R, 0.001, np.inf, spheres, lights, recursion_depth - 1)
    return local_color * (1 - r) + reflected_color * r

# Initialize canvas, spheres, and lights.
canvas = Canvas(canvasWidth, canvasHeight)
spheres = [
    Sphere((0, 1, 3), 1, (255, 0, 0), 500, 0.2),      # Red sphere
    Sphere((2, 0, 4), 1, (0, 0, 255), 500, 0.3),      # Blue sphere
    Sphere((-2, 0, 4), 1, (0, 255, 0), 10, 0.4),      # Green sphere
    Sphere((0, -1, 4), 1, (255, 0, 255), 500, 0.2),   # Purple sphere
    Sphere((0, 5001, 0), 5000, (255, 255, 0), 1000, 0.5),  # Yellow ground
]

lights = [
    Light(type="ambient", intensity=0.2),
    Light(type="point", intensity=0.6, position=(2, -1, 0)),
    Light(type="directional", intensity=0.2, position=(1, -4, 4))
]

if __name__ == "__main__":
    start_time = time.time()

    # Create a list of pixel coordinates (centered)
    pixel_coords = [
        (x, y)
        for x in range(-canvasWidth // 2, canvasWidth // 2)
        for y in range(-canvasHeight // 2, canvasHeight // 2)
    ]

    with Pool() as pool:
        results = pool.map(compute_pixel, pixel_coords)

    # Apply the computed pixel colors to the canvas.
    for x, y, color in results:
        canvas.PutPixel(x, y, color)
    img = Image.fromarray(canvas.pixels, 'RGB')
    img.save("image.png")

    end = time.time()
    print(round(end - start_time), "seconds")
