from multiprocessing.pool import Pool
import numpy as np
from PIL import Image
import time

# Camera and viewport settings
O = np.array((0, 0, 0))  # Camera position
canvasWidth, canvasHeight = 500, 500
viewWidth, viewHeight = 1, 1
background_color = np.array((0, 0, 0))  # Black background

# Supersampling factor (e.g., 2 means 2x2 = 4 samples per pixel)
supersample_factor = 2

class Sphere:
    def __init__(self, center, radius, color, specular, reflective, transparent=0, refractive_index=1):
        """
        Parameters:
         - reflective: reflection coefficient (0 = non-reflective)
         - transparent: transparency coefficient (0 = opaque)
         - refractive_index: refractive index (only used if transparent > 0)
        """
        self.center = np.array(center)
        self.radius = radius
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
        self.position = np.array(position) if position else None

def compute_pixel(args):
    x, y = args
    color_sum = np.zeros(3, dtype=np.float64)
    # Supersampling: take several samples inside each pixel.
    for i in range(supersample_factor):
        for j in range(supersample_factor):
            # Compute a uniform grid of subpixel offsets in [-0.5, 0.5)
            offset_x = (i + 0.5) / supersample_factor - 0.5
            offset_y = (j + 0.5) / supersample_factor - 0.5
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

def RefractRay(I, N, n1, n2):
    """
    Computes the refraction (transmitted) ray using Snell's law.
    I and N should be normalized. n1 is the refractive index of the medium the ray is coming from,
    and n2 is that of the medium it is entering.
    Returns None if total internal reflection occurs.
    """
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
        # Shadow check: if any object blocks the light.
        shadow_sphere, shadow_t = ClosestIntersection(P, L, 0.001, t_max, spheres)
        if shadow_sphere is not None:
            continue
        # Diffuse
        n_dot_l = np.dot(N, L)
        if n_dot_l > 0:
            intensity += light.intensity * n_dot_l
        # Specular
        if s != -1:
            R = ReflectRay(N, L)
            r_dot_v = np.dot(R, V)
            if r_dot_v > 0:
                intensity += light.intensity * pow(r_dot_v / (vecLength(R) * vecLength(V)), s)
    return intensity

def TraceRay(O, D, t_min, t_max, spheres, lights, recursion_depth):
    closest_sphere, closest_t = ClosestIntersection(O, D, t_min, t_max, spheres)
    if closest_sphere is None:
        return background_color

    # Compute the hit point and normal.
    P = O + closest_t * D
    N = P - closest_sphere.center
    N = N / vecLength(N)

    # Local lighting (diffuse and specular)
    local_intensity = ComputeLighting(P, N, -D, closest_sphere.specular, lights)
    local_color = closest_sphere.color * local_intensity

    # Prepare coefficients.
    r = closest_sphere.reflective
    t_coef = closest_sphere.transparent  # Transparency coefficient

    # If recursion has ended or the material is purely opaque with no reflection/transparency, return local.
    if recursion_depth <= 0 or (r == 0 and t_coef == 0):
        return local_color

    # Compute reflection contribution.
    reflected_color = np.zeros(3)
    if r > 0:
        R = ReflectRay(-D, N)
        reflected_color = TraceRay(P, R, 0.001, np.inf, spheres, lights, recursion_depth - 1)

    # Compute refraction (transparency) contribution.
    transmitted_color = np.zeros(3)
    if t_coef > 0:
        # Determine if the ray is inside the sphere.
        # If the ray is inside, then the normal should be inverted.
        if np.dot(D, N) > 0:
            # The ray is inside the sphere.
            n1 = closest_sphere.refractive_index
            n2 = 1  # Going from sphere to air
            N_transmit = -N
        else:
            n1 = 1
            n2 = closest_sphere.refractive_index
            N_transmit = N

        T = RefractRay(D, N_transmit, n1, n2)
        if T is None:
            # Total internal reflection; treat it as reflection.
            transmitted_color = reflected_color
        else:
            transmitted_color = TraceRay(P, T, 0.001, np.inf, spheres, lights, recursion_depth - 1)

    # Combine contributions.
    # The weights for local, reflection, and transmission should sum to 1.
    # Here we assume:
    #   local weight = (1 - r - t_coef)
    #   reflection weight = r
    #   transparency weight = t_coef
    final_color = local_color * (1 - r - t_coef) + reflected_color * r + transmitted_color * t_coef
    return final_color

# Initialize canvas, spheres, and lights.
canvas = Canvas(canvasWidth, canvasHeight)
spheres = [
    Sphere((0, 1, 3), 1, (255, 0, 0), 500, 0.2, transparent=0.5, refractive_index=1.5),  # Red sphere (opaque)
    Sphere((2, 0, 4), 1, (0, 0, 255), 500, 0.3, transparent=0.5, refractive_index=1.5),  # Blue sphere (opaque)
    # Green sphere is now made partially transparent (e.g., glass-like) with a refractive index of 1.5.
    Sphere((-2, 0, 4), 1, (0, 255, 0), 10, 0.1, transparent=0.5, refractive_index=1.5),
    Sphere((0, -1, 4), 1, (255, 0, 255), 500, 0.2, transparent=0.5, refractive_index=1.5),  # Purple sphere (opaque)
    Sphere((0, 5001, 0), 5000, (255, 255, 0), 1000, 0.5),  # Yellow ground (opaque)
]

lights = [
    Light(type="ambient", intensity=0.2),
    Light(type="point", intensity=0.6, position=(2, -1, 0)),
    Light(type="directional", intensity=0.2, position=(1, -4, 4))
]

if __name__ == "__main__":
    start_time = time.time()

    # Create a list of pixel coordinates (centered).
    pixel_coords = [
        (x, y)
        for x in range(-canvasWidth // 2, canvasWidth // 2)
        for y in range(-canvasHeight // 2, canvasHeight // 2)
    ]

    with Pool() as pool:
        results = pool.map(compute_pixel, pixel_coords)

    # Apply computed pixel colors to the canvas.
    for x, y, color in results:
        canvas.PutPixel(x, y, color)
    img = Image.fromarray(canvas.pixels, 'RGB')
    img.save("image.png")

    end = time.time()
    print(round(end - start_time), "seconds")
