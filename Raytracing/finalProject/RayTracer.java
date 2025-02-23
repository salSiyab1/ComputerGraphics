import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;

public class RayTracer {
    // Constants
    public static final double PI = Math.PI;
    public static final double INFINITY = Double.POSITIVE_INFINITY;

    // Utility functions
    public static double degreesToRadians(double degrees) {
        return degrees * PI / 180.0;
    }

    public static double clamp(double x, double min, double max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    public static double randomDouble() {
        return Math.random(); // returns in [0, 1)
    }

    public static double randomDouble(double min, double max) {
        return min + (max - min) * randomDouble();
    }

    // Vec3: 3D vector class used for points and colors.
    public static class Vec3 {
        public double x, y, z;

        public Vec3() {
            this(0, 0, 0);
        }

        public Vec3(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public Vec3 add(Vec3 v) {
            return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z);
        }

        public Vec3 subtract(Vec3 v) {
            return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z);
        }

        public Vec3 multiply(double t) {
            return new Vec3(this.x * t, this.y * t, this.z * t);
        }

        // Component-wise multiplication
        public Vec3 multiply(Vec3 v) {
            return new Vec3(this.x * v.x, this.y * v.y, this.z * v.z);
        }

        public Vec3 divide(double t) {
            return this.multiply(1 / t);
        }

        public double length() {
            return Math.sqrt(lengthSquared());
        }

        public double lengthSquared() {
            return x * x + y * y + z * z;
        }

        public Vec3 negate() {
            return new Vec3(-x, -y, -z);
        }

        public Vec3 unitVector() {
            return this.divide(length());
        }

        public static double dot(Vec3 u, Vec3 v) {
            return u.x * v.x + u.y * v.y + u.z * v.z;
        }

        public static Vec3 cross(Vec3 u, Vec3 v) {
            return new Vec3(
                    u.y * v.z - u.z * v.y,
                    u.z * v.x - u.x * v.z,
                    u.x * v.y - u.y * v.x
            );
        }

        public static Vec3 random() {
            return new Vec3(randomDouble(), randomDouble(), randomDouble());
        }

        public static Vec3 random(double min, double max) {
            return new Vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
        }

        @Override
        public String toString() {
            return x + " " + y + " " + z;
        }
    }

    // Ray class
    public static class Ray {
        public Vec3 orig;
        public Vec3 dir;

        public Ray(Vec3 orig, Vec3 dir) {
            this.orig = orig;
            this.dir = dir;
        }

        public Vec3 at(double t) {
            return orig.add(dir.multiply(t));
        }
    }

    // Hit record for storing intersection details
    public static class HitRecord {
        public Vec3 p;
        public Vec3 normal;
        public Material mat;
        public double t;
        public boolean frontFace;

        public void setFaceNormal(Ray r, Vec3 outwardNormal) {
            frontFace = Vec3.dot(r.dir, outwardNormal) < 0;
            normal = frontFace ? outwardNormal : outwardNormal.negate();
        }
    }

    // Hittable interface for objects that can be hit by rays
    public static interface Hittable {
        boolean hit(Ray r, double tMin, double tMax, HitRecord rec);
    }

    // Sphere class implementing Hittable
    public static class Sphere implements Hittable {
        public Vec3 center;
        public double radius;
        public Material mat;

        public Sphere(Vec3 center, double radius, Material mat) {
            this.center = center;
            this.radius = radius;
            this.mat = mat;
        }

        @Override
        public boolean hit(Ray r, double tMin, double tMax, HitRecord rec) {
            Vec3 oc = r.orig.subtract(center);
            double a = r.dir.lengthSquared();
            double half_b = Vec3.dot(oc, r.dir);
            double c = oc.lengthSquared() - radius * radius;
            double discriminant = half_b * half_b - a * c;
            if (discriminant < 0) return false;
            double sqrtd = Math.sqrt(discriminant);

            double root = (-half_b - sqrtd) / a;
            if (root < tMin || root > tMax) {
                root = (-half_b + sqrtd) / a;
                if (root < tMin || root > tMax)
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            Vec3 outwardNormal = rec.p.subtract(center).divide(radius);
            rec.setFaceNormal(r, outwardNormal);
            rec.mat = mat;
            return true;
        }
    }

    // HittableList: a list of hittable objects.
    public static class HittableList implements Hittable {
        public List<Hittable> objects;

        public HittableList() {
            objects = new ArrayList<>();
        }

        public HittableList(Hittable object) {
            this();
            add(object);
        }

        public void clear() {
            objects.clear();
        }

        public void add(Hittable object) {
            objects.add(object);
        }

        @Override
        public boolean hit(Ray r, double tMin, double tMax, HitRecord rec) {
            HitRecord tempRec = new HitRecord();
            boolean hitAnything = false;
            double closestSoFar = tMax;
            for (Hittable object : objects) {
                if (object.hit(r, tMin, closestSoFar, tempRec)) {
                    hitAnything = true;
                    closestSoFar = tempRec.t;
                    rec.t = tempRec.t;
                    rec.p = tempRec.p;
                    rec.normal = tempRec.normal;
                    rec.mat = tempRec.mat;
                    rec.frontFace = tempRec.frontFace;
                }
            }
            return hitAnything;
        }
    }

    // Abstract material class and a helper record for scattering results.
    public static abstract class Material {
        // Scatter method: if the ray is scattered, set the scattered ray and attenuation, then return true.
        public abstract boolean scatter(Ray rIn, HitRecord rec, ScatterRecord sRec);
    }

    public static class ScatterRecord {
        public Ray scattered;
        public Vec3 attenuation;
    }

    // Lambertian (diffuse) material
    public static class Lambertian extends Material {
        public Vec3 albedo;

        public Lambertian(Vec3 albedo) {
            this.albedo = albedo;
        }

        @Override
        public boolean scatter(Ray rIn, HitRecord rec, ScatterRecord sRec) {
            Vec3 scatterDirection = rec.normal.add(randomUnitVector());
            // Catch degenerate scatter direction
            if (scatterDirection.lengthSquared() < 1e-8)
                scatterDirection = rec.normal;
            sRec.scattered = new Ray(rec.p, scatterDirection);
            sRec.attenuation = albedo;
            return true;
        }
    }

    // Metal (reflective) material
    public static class Metal extends Material {
        public Vec3 albedo;
        public double fuzz;

        public Metal(Vec3 albedo, double fuzz) {
            this.albedo = albedo;
            this.fuzz = fuzz < 1 ? fuzz : 1;
        }

        @Override
        public boolean scatter(Ray rIn, HitRecord rec, ScatterRecord sRec) {
            Vec3 reflected = reflect(rIn.dir.unitVector(), rec.normal);
            sRec.scattered = new Ray(rec.p, reflected.add(randomInUnitSphere().multiply(fuzz)));
            sRec.attenuation = albedo;
            return Vec3.dot(sRec.scattered.dir, rec.normal) > 0;
        }
    }

    // Dielectric (glass-like) material
    public static class Dielectric extends Material {
        public double ir; // Index of Refraction

        public Dielectric(double indexOfRefraction) {
            this.ir = indexOfRefraction;
        }

        @Override
        public boolean scatter(Ray rIn, HitRecord rec, ScatterRecord sRec) {
            sRec.attenuation = new Vec3(1.0, 1.0, 1.0);
            double refractionRatio = rec.frontFace ? (1.0 / ir) : ir;

            Vec3 unitDirection = rIn.dir.unitVector();
            double cosTheta = Math.min(Vec3.dot(unitDirection.negate(), rec.normal), 1.0);
            double sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);

            boolean cannotRefract = refractionRatio * sinTheta > 1.0;
            Vec3 direction;
            if (cannotRefract || reflectance(cosTheta, refractionRatio) > randomDouble())
                direction = reflect(unitDirection, rec.normal);
            else
                direction = refract(unitDirection, rec.normal, refractionRatio);
            sRec.scattered = new Ray(rec.p, direction);
            return true;
        }

        private double reflectance(double cosine, double refIdx) {
            double r0 = (1 - refIdx) / (1 + refIdx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * Math.pow((1 - cosine), 5);
        }
    }

    // Helper functions for reflection, refraction, and random vectors.
    public static Vec3 reflect(Vec3 v, Vec3 n) {
        return v.subtract(n.multiply(2 * Vec3.dot(v, n)));
    }

    public static Vec3 refract(Vec3 uv, Vec3 n, double etaiOverEtat) {
        double cosTheta = Math.min(Vec3.dot(uv.negate(), n), 1.0);
        Vec3 rOutPerp = uv.add(n.multiply(cosTheta)).multiply(etaiOverEtat);
        Vec3 rOutParallel = n.multiply(-Math.sqrt(Math.abs(1.0 - rOutPerp.lengthSquared())));
        return rOutPerp.add(rOutParallel);
    }

    public static Vec3 randomInUnitSphere() {
        while (true) {
            Vec3 p = Vec3.random(-1, 1);
            if (p.lengthSquared() >= 1) continue;
            return p;
        }
    }

    public static Vec3 randomUnitVector() {
        return randomInUnitSphere().unitVector();
    }

    // Camera class
    public static class Camera {
        public Vec3 origin;
        public Vec3 lowerLeftCorner;
        public Vec3 horizontal;
        public Vec3 vertical;
        public Vec3 u, v, w;
        public double lensRadius;

        public Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspectRatio, double aperture, double focusDist) {
            double theta = degreesToRadians(vfov);
            double h = Math.tan(theta / 2);
            double viewportHeight = 2.0 * h;
            double viewportWidth = aspectRatio * viewportHeight;

            w = lookfrom.subtract(lookat).unitVector();
            u = Vec3.cross(vup, w).unitVector();
            v = Vec3.cross(w, u);

            origin = lookfrom;
            horizontal = u.multiply(viewportWidth * focusDist);
            vertical = v.multiply(viewportHeight * focusDist);
            lowerLeftCorner = origin.subtract(horizontal.divide(2))
                    .subtract(vertical.divide(2))
                    .subtract(w.multiply(focusDist));

            lensRadius = aperture / 2;
        }

        public Ray getRay(double s, double t) {
            Vec3 rd = randomInUnitSphere().multiply(lensRadius);
            Vec3 offset = u.multiply(rd.x).add(v.multiply(rd.y));
            return new Ray(origin.add(offset),
                    lowerLeftCorner.add(horizontal.multiply(s))
                            .add(vertical.multiply(t))
                            .subtract(origin)
                            .subtract(offset));
        }
    }

    // Recursive function to compute the ray's color.
    public static Vec3 rayColor(Ray r, Hittable world, int depth) {
        if (depth <= 0)
            return new Vec3(0, 0, 0);

        HitRecord rec = new HitRecord();
        if (world.hit(r, 0.001, INFINITY, rec)) {
            ScatterRecord sRec = new ScatterRecord();
            if (rec.mat.scatter(r, rec, sRec))
                return sRec.attenuation.multiply(rayColor(sRec.scattered, world, depth - 1));
            return new Vec3(0, 0, 0);
        }

        Vec3 unitDirection = r.dir.unitVector();
        double t = 0.5 * (unitDirection.y + 1.0);
        return new Vec3(1.0, 1.0, 1.0).multiply(1.0 - t)
                .add(new Vec3(0.5, 0.7, 1.0).multiply(t));
    }

    // Main rendering function.
    public static void main(String[] args) throws Exception {
        // Image parameters
        double aspectRatio = 16.0 / 9.0;
        int imageWidth = 1200;
        int imageHeight = (int)(imageWidth / aspectRatio);
        int samplesPerPixel = 100;
        int maxDepth = 50;

        // World: build the scene with a ground plane, three large spheres, and many small spheres.
        HittableList world = new HittableList();

        Material groundMaterial = new Lambertian(new Vec3(0.5, 0.5, 0.5));
        world.add(new Sphere(new Vec3(0, -1000, 0), 1000, groundMaterial));

        // Three large spheres
        Material material1 = new Dielectric(1.5);
        world.add(new Sphere(new Vec3(0, 1, 0), 1.0, material1));

        Material material2 = new Lambertian(new Vec3(0.4, 0.2, 0.1));
        world.add(new Sphere(new Vec3(-4, 1, 0), 1.0, material2));

        Material material3 = new Metal(new Vec3(0.7, 0.6, 0.5), 0.0);
        world.add(new Sphere(new Vec3(4, 1, 0), 1.0, material3));

        // Random small spheres
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                double chooseMat = randomDouble();
                Vec3 center = new Vec3(a + 0.9 * randomDouble(), 0.2, b + 0.9 * randomDouble());
                if (center.subtract(new Vec3(4, 0.2, 0)).length() > 0.9) {
                    Material sphereMaterial;
                    if (chooseMat < 0.8) {
                        Vec3 albedo = Vec3.random().multiply(Vec3.random());
                        sphereMaterial = new Lambertian(albedo);
                        world.add(new Sphere(center, 0.2, sphereMaterial));
                    } else if (chooseMat < 0.95) {
                        Vec3 albedo = Vec3.random(0.5, 1);
                        double fuzz = randomDouble(0, 0.5);
                        sphereMaterial = new Metal(albedo, fuzz);
                        world.add(new Sphere(center, 0.2, sphereMaterial));
                    } else {
                        sphereMaterial = new Dielectric(1.5);
                        world.add(new Sphere(center, 0.2, sphereMaterial));
                    }
                }
            }
        }

        // Camera settings
        Vec3 lookfrom = new Vec3(13, 2, 3);
        Vec3 lookat = new Vec3(0, 0, 0);
        Vec3 vup = new Vec3(0, 1, 0);
        double distToFocus = 10.0;
        double aperture = 0.1;
        Camera cam = new Camera(lookfrom, lookat, vup, 20, aspectRatio, aperture, distToFocus);

        // Create a BufferedImage to store the rendered image.
        BufferedImage image = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);

        // Render loop: for each pixel, sample multiple rays for anti-aliasing.
        for (int j = imageHeight - 1; j >= 0; j--) {
            System.err.println("Scanlines remaining: " + j);
            for (int i = 0; i < imageWidth; i++) {
                Vec3 pixelColor = new Vec3(0, 0, 0);
                for (int s = 0; s < samplesPerPixel; s++) {
                    double u = (i + randomDouble()) / (imageWidth - 1);
                    double v = (j + randomDouble()) / (imageHeight - 1);
                    Ray r = cam.getRay(u, v);
                    pixelColor = pixelColor.add(rayColor(r, world, maxDepth));
                }

                // Convert the pixel's color to [0, 255] with gamma correction.
                double scale = 1.0 / samplesPerPixel;
                double r = Math.sqrt(scale * pixelColor.x);
                double g = Math.sqrt(scale * pixelColor.y);
                double b = Math.sqrt(scale * pixelColor.z);

                int ir = (int)(256 * clamp(r, 0.0, 0.999));
                int ig = (int)(256 * clamp(g, 0.0, 0.999));
                int ib = (int)(256 * clamp(b, 0.0, 0.999));

                // Compose the RGB value and set it in the BufferedImage.
                int rgb = (ir << 16) | (ig << 8) | ib;
                // Note: image coordinate y is inverted relative to our loop.
                image.setRGB(i, imageHeight - j - 1, rgb);
            }
        }
        System.err.println("Done rendering.");

        // Write the image as a JPEG file.
        ImageIO.write(image, "jpg", new File("image.jpg"));
        System.err.println("Image saved as image.jpg");
    }
}
