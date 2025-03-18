from multiprocessing.pool import Pool
import numpy as np
from PIL import Image
import time

# Camera and viewport settings
O = np.array((0, 0, 0))  # Camera position
canvasWidth, canvasHeight = 100, 100
viewWidth, viewHeight = 1, 1
background_color = np.array((0, 0, 0))  # Black background

# Supersampling factor (e.g., 1 means one sample per pixel)
supersample_factor = 1

# -----------------------------------------------------------------------------
# Object Classes
# -----------------------------------------------------------------------------
class Sphere:
    def __init__(self, center, radius, color, specular, reflective, transparent=0, refractive_index=1):
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.color = np.array(color, dtype=np.float64)
        self.specular = specular
        self.reflective = reflective
        self.transparent = transparent
        self.refractive_index = refractive_index

class Triangle:
    def __init__(self, v0, v1, v2, color, specular, reflective, transparent=0, refractive_index=1):
        self.v0 = np.array(v0, dtype=np.float64)
        self.v1 = np.array(v1, dtype=np.float64)
        self.v2 = np.array(v2, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.specular = specular
        self.reflective = reflective
        self.transparent = transparent
        self.refractive_index = refractive_index

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
        self.position = np.array(position, dtype=np.float64) if position else None

# -----------------------------------------------------------------------------
# Intersection and Utility Functions
# -----------------------------------------------------------------------------
def vecLength(vector):
    return np.sqrt(np.dot(vector, vector))

def CanvasToViewport(x, y):
    return np.array((x * viewWidth / canvasWidth, y * viewHeight / canvasHeight, 1), dtype=np.float64)

def ReflectRay(R, N):
    return 2 * N * np.dot(N, R) - R

def RefractRay(I, N, n1, n2):
    eta = n1 / n2
    cos_i = -np.dot(I, N)
    sin_t2 = eta ** 2 * (1 - cos_i ** 2)
    if sin_t2 > 1:
        return None  # Total internal reflection
    cos_t = np.sqrt(1 - sin_t2)
    T = eta * I + (eta * cos_i - cos_t) * N
    return T

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

def IntersectRayTriangle(O, D, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle.v0, triangle.v1, triangle.v2
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(D, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return np.inf  # Ray is parallel to the triangle.
    f = 1.0 / a
    s = O - v0
    u = f * np.dot(s, h)
    if u < 0 or u > 1:
        return np.inf
    q = np.cross(s, edge1)
    v = f * np.dot(D, q)
    if v < 0 or (u + v) > 1:
        return np.inf
    t = f * np.dot(edge2, q)
    if t > epsilon:
        return t
    else:
        return np.inf

def ClosestIntersection(O, D, t_min, t_max, spheres, triangles):
    closest_t = np.inf
    closest_obj = None
    for sphere in spheres:
        t1, t2 = IntersectRaySphere(O, D, sphere)
        if t_min < t1 < t_max and t1 < closest_t:
            closest_t = t1
            closest_obj = sphere
        if t_min < t2 < t_max and t2 < closest_t:
            closest_t = t2
            closest_obj = sphere
    for triangle in triangles:
        t = IntersectRayTriangle(O, D, triangle)
        if t_min < t < t_max and t < closest_t:
            closest_t = t
            closest_obj = triangle
    return closest_obj, closest_t

def ComputeLighting(P, N, V, s, lights):
    intensity = 0.0
    for light in lights:
        if light.type == "ambient":
            intensity += light.intensity
            continue
        if light.type == "point":
            L = light.position - P
            t_max = 1
        elif light.type == "directional":
            L = light.position
            t_max = np.inf
        else:
            continue
        L = L / vecLength(L)
        shadow_obj, shadow_t = ClosestIntersection(P, L, 0.001, t_max, spheres, triangles)
        if shadow_obj is not None:
            continue
        n_dot_l = np.dot(N, L)
        if n_dot_l > 0:
            intensity += light.intensity * n_dot_l
        if s != -1:
            R = ReflectRay(N, L)
            r_dot_v = np.dot(R, V)
            if r_dot_v > 0:
                intensity += light.intensity * pow(r_dot_v / (vecLength(R) * vecLength(V)), s)
    return intensity

# -----------------------------------------------------------------------------
# Ray Tracing Core Function
# -----------------------------------------------------------------------------
def TraceRay(O, D, t_min, t_max, spheres, triangles, lights, recursion_depth):
    closest_obj, closest_t = ClosestIntersection(O, D, t_min, t_max, spheres, triangles)
    if closest_obj is None:
        return background_color

    P = O + closest_t * D

    if isinstance(closest_obj, Sphere):
        N = P - closest_obj.center
        N = N / vecLength(N)
        local_intensity = ComputeLighting(P, N, -D, closest_obj.specular, lights)
        local_color = closest_obj.color * local_intensity
        r = closest_obj.reflective
        t_coef = closest_obj.transparent
        refractive_index = closest_obj.refractive_index
    elif isinstance(closest_obj, Triangle):
        N = np.cross(closest_obj.v1 - closest_obj.v0, closest_obj.v2 - closest_obj.v0)
        N = N / vecLength(N)
        local_intensity = ComputeLighting(P, N, -D, closest_obj.specular, lights)
        local_color = closest_obj.color * local_intensity
        r = closest_obj.reflective
        t_coef = closest_obj.transparent
        refractive_index = closest_obj.refractive_index
    else:
        return background_color

    if recursion_depth <= 0 or (r == 0 and t_coef == 0):
        return local_color

    reflected_color = np.zeros(3)
    if r > 0:
        R = ReflectRay(-D, N)
        reflected_color = TraceRay(P, R, 0.001, np.inf, spheres, triangles, lights, recursion_depth - 1)

    transmitted_color = np.zeros(3)
    if t_coef > 0:
        if np.dot(D, N) > 0:
            n1, n2 = refractive_index, 1
            N_transmit = -N
        else:
            n1, n2 = 1, refractive_index
            N_transmit = N
        T = RefractRay(D, N_transmit, n1, n2)
        if T is None:
            transmitted_color = reflected_color
        else:
            transmitted_color = TraceRay(P, T, 0.001, np.inf, spheres, triangles, lights, recursion_depth - 1)

    final_color = local_color * (1 - r - t_coef) + reflected_color * r + transmitted_color * t_coef
    return final_color

# -----------------------------------------------------------------------------
# Pixel Computation (with Supersampling)
# -----------------------------------------------------------------------------
def compute_pixel(args):
    x, y = args
    color_sum = np.zeros(3, dtype=np.float64)
    for i in range(supersample_factor):
        for j in range(supersample_factor):
            offset_x = (i + 0.5) / supersample_factor - 0.5
            offset_y = (j + 0.5) / supersample_factor - 0.5
            sample_x = x + offset_x
            sample_y = y + offset_y

            D = CanvasToViewport(sample_x, sample_y)
            D = D / vecLength(D)
            # Use a lower t_min to catch near intersections
            sample_color = TraceRay(O, D, 0.00001, np.inf, spheres, triangles, lights, 3)
            color_sum += sample_color
    avg_color = color_sum / (supersample_factor ** 2)
    return x + canvasWidth // 2, y + canvasHeight // 2, avg_color

# -----------------------------------------------------------------------------
# Scene Setup
# -----------------------------------------------------------------------------
canvas = Canvas(canvasWidth, canvasHeight)

# List of spheres.
spheres = [
    # Sphere((0, 1, 3), 1, (255, 0, 0), 500, 0.2),
    # Sphere((2, 0, 4), 1, (0, 0, 255), 500, 0.3),
    # Sphere((-2, 0, 4), 1, (0, 255, 0), 10, 0.1, transparent=0.5, refractive_index=1.5),
    Sphere((0, 5001, 0), 5000, (255, 255, 0), 1000, 0.5),
]

# Process OBJ file and populate triangle list.
# --- Transform the bunny --- #
# Define a scale and translation for the bunny.
scale = 20.0
translation = np.array([0, 0, 4])  # Move the bunny 3 units along the z-axis

obj_vertices = []
raw_triangles = []
with open("bunny.obj", "r") as file:
    for line in file:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "v":
            # Transform the vertex: scale and then translate.
            v = np.array(tuple(map(float, parts[1:4])))
            transformed_v = v * -scale + translation
            obj_vertices.append(tuple(transformed_v))
        elif parts[0] == "f":
            indices = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
            raw_triangles.append(tuple(indices))



obj_triangles = []
for indices in raw_triangles:
    v0 = obj_vertices[indices[0]]
    v1 = obj_vertices[indices[1]]
    v2 = obj_vertices[indices[2]]
    obj_triangles.append(Triangle(v0, v1, v2, (250, 0, 0), specular=300, reflective=0.0))

triangles = obj_triangles

lights = [
    Light(type="ambient", intensity=0.2),
    Light(type="point", intensity=0.6, position=(2, -1, 0)),
    Light(type="directional", intensity=0.2, position=(1, -4, 4))
]

# -----------------------------------------------------------------------------
# Main Rendering Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    pixel_coords = [(x, y) for x in range(-canvasWidth // 2, canvasWidth // 2)
                              for y in range(-canvasHeight // 2, canvasHeight // 2)]

    with Pool() as pool:
        results = pool.map(compute_pixel, pixel_coords)

    for x, y, color in results:
        canvas.PutPixel(x, y, color)

    img = Image.fromarray(canvas.pixels, 'RGB')
    img.save("image.png")

    end = time.time()
    print(round(end - start_time), "seconds")
