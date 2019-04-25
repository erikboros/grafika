//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Boros Erik
// Neptun : JREBRU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	bool rough;
	vec3 F0;
	Material(vec3 _kd, vec3 _ks, float _shininess, bool _rough) : ka(_kd * M_PI), kd(_kd), ks(_ks), rough(_rough) {
		shininess = _shininess; 
		F0 = vec3(0.9, 0.85, 0.8);
	}
};


struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Side : public Intersectable {
	std::vector<vec3> r;
	Side(Material* _material) {
		material = _material;
		r.push_back(vec3(-0.5, 0, 0));
		r.push_back(vec3(0, 0.1, 1));
		r.push_back(vec3(0.5,  0, 0));
		
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 n = normalize( cross((r[1] - r[0]),(r[2] - r[0])) );
		float t = dot((r[0] - ray.start), n) / dot(ray.dir, n);
		vec3 p= ray.start + ray.dir * t;
		
		if (dot(cross(r[1] - r[0], p - r[0]), n) > 0
			&& dot(cross(r[2] - r[1], p - r[1]), n) > 0
			&& dot(cross(r[0] - r[2], p - r[2]), n) > 0) {
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = n;
			hit.material = material;
			return hit;
		}

		return hit;
	}
};

struct Wall : public Intersectable {
	std::vector<vec3> r;
	Wall(vec3 a, vec3 b, vec3 c, vec3 d, Material* _material) {
		material = _material;
		r.push_back(a);
		r.push_back(b);
		r.push_back(c);
		r.push_back(d);

	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = 0;
		vec3 normal = normalize(cross((r[1] - r[0]), (r[2] - r[0])));
		float t = dot((r[0] - ray.start), normal) / dot(ray.dir, normal);
		vec3 p = ray.start + ray.dir * t;

		if (   dot(cross(r[1] - r[0], p - r[0]), normal) > 0
			&& dot(cross(r[0] - r[2], p - r[2]), normal) > 0
			&& dot(cross(r[2] - r[3], p - r[3]), normal) > 0
			&& dot(cross(r[3] - r[1], p - r[1]), normal) > 0) {
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normal;
			hit.material = material;
			return hit;
		}

		return hit;
	}
};

struct PolyTube{
	Material *material;
	int n;
	float d;
	PolyTube(Material* _material) {
		material = _material;
		n = 5;
		d = 0.4;
	}
	void addSide() {
		n++;
	}
	void putSides(std::vector<Intersectable *> &o) {
		for (int i = 0; i < n-1; i++) {
			vec3 t1 = (vec3(cos(i * 2 * M_PI / n + M_PI / 2), sin(i * 2 * M_PI / n + M_PI / 2), 0))*d;
			vec3 t3 = (vec3(cos((i+1) * 2 * M_PI / n + M_PI / 2), sin((i+1) * 2 * M_PI / n + M_PI / 2), 0))*d;
			o.push_back(new Wall(t1,vec3(t1.x,t1.y,2), t3, vec3(t3.x, t3.y, 2), material));
		}
		vec3 t3 = (vec3(cos( M_PI / 2), sin( M_PI / 2), 0))*d;
		vec3 t1 = (vec3(cos((n-1) * 2 * M_PI / n + M_PI / 2), sin((n-1) * 2 * M_PI / n + M_PI / 2), 0))*d;
		o.push_back(new Wall(t1, vec3(t1.x, t1.y, 2), t3, vec3(t3.x, t3.y, 2), material));
	}
	/*
	Hit intersect(const Ray& ray) {
		Hit hit;
		for (Wall w : r) {
			hit = w.intersect(ray);
			if (hit.t != 0) {
				return hit;
			}
		}

		/*
		for (int i = 0; i < n*2-2; i+=2){		//minden oldalra (kivéve utolsó)
			vec3 normal = normalize(cross((r[i+1] - r[i+0]), (r[i+2] - r[i+0])));
			float t = dot((r[i+0] - ray.start), normal) / dot(ray.dir, normal);
			vec3 p = ray.start + ray.dir * t;

			
			if (   dot(cross(r[i + 1] - r[i + 0], p - r[i + 0]), n) > 0
				&& dot(cross(r[i + 0] - r[i + 2], p - r[i + 2]), n) > 0
				&& dot(cross(r[i + 2] - r[i + 3], p - r[i + 3]), n) > 0
				&& dot(cross(r[i + 3] - r[i + 1], p - r[i + 1]), n) > 0) {
				hit.t = t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normal;
				hit.material = material;
				return hit;
			}
			if (dot(cross(r[i + 0] - r[i + 1], p - r[i + 0]), n) > 0
				&& dot(cross(r[i + 2] - r[i + 0], p - r[i + 2]), n) > 0
				&& dot(cross(r[i + 3] - r[i + 2], p - r[i + 3]), n) > 0
				&& dot(cross(r[i + 1] - r[i + 3], p - r[i + 1]), n) > 0) {
				hit.t = t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normal;
				hit.material = material;
				return hit;
			}
			

			/*
			if (   dot(cross(r[i + 1] - r[i + 0], p - r[i + 0]), n) > 0
				&& dot(cross(r[i + 2] - r[i + 1], p - r[i + 1]), n) > 0
				&& dot(cross(r[i + 0] - r[i + 2], p - r[i + 2]), n) > 0) {
				hit.t = t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = n;
				hit.material = material;
				return hit;
			}
			

			if (   dot(cross(r[i + 0] - r[i + 1], p - r[i + 1]), n) > 0
				&& dot(cross(r[i + 1] - r[i + 2], p - r[i + 2]), n) > 0
				&& dot(cross(r[i + 2] - r[i + 0], p - r[i + 0]), n) > 0) {
				hit.t = t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = n;
				hit.material = material;
				return hit;
			}
			
		}

		
		vec3 normal = normalize(cross((r[n-1] - r[n-2]), (r[0] - r[n-2])));
		float t = dot((r[n-2] - ray.start), normal) / dot(ray.dir, normal);
		vec3 p = ray.start + ray.dir * t;

		if (   dot(cross(r[n - 2] - r[n - 1], p - r[n - 2]), n) > 0
			&& dot(cross(r[0    ] - r[n - 2], p - r[0    ]), n) > 0
			&& dot(cross(r[1    ] - r[0    ], p - r[1    ]), n) > 0
			&& dot(cross(r[n - 1] - r[1    ], p - r[n - 1]), n) > 0) {
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normal;
			hit.material = material;
			return hit;
		}
		
		
		return hit;
	}
	*/
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
		La = vec3(0.4, 0.3, 0.3);
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 80 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(0, 0, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		//lightDirection = vec3(-1, 1, 1);
		//lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f); vec3 ks(2, 2, 2);
		Material * material = new Material(kd, ks, 40, true);
		/*
		for (int i = 0; i < 25; i++) {
			kd = vec3(rnd() / 3, rnd() / 3, rnd() / 3);
			ks= vec3(2, 2, 2);
			material = new Material(kd, ks, 40, true);
			
			objects.push_back(new Sphere(vec3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5), rnd() * 0.1, material));
		}
		*/
		objects.push_back(new Sphere(vec3(0, 0, 0), 0.2, material));

		kd=vec3(0.3f, 0.2f, 0.1f); ks=vec3(2, 2, 2);
		Material * smaterial = new Material(kd, ks, 50, false);
		objects.push_back(new Sphere(vec3(0, 0, 0), 0.2, smaterial));


		PolyTube mirror(smaterial);
		mirror.putSides(objects);

		/*
		vec3 a(-0.5, 0, 0);
		vec3 b(0.5, 0, 0);
		vec3 c(-0.5, -0.1, 1);
		vec3 d(0.5, -0.1, 1);
		objects.push_back(new Wall(a, c, b, d, material));*/
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) {
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	vec3 reflect(vec3 I, vec3 N) {
		return I - N * 2.0f * dot(N, I);
	}

	vec3 trace(Ray ray, int maxdepth=5) {
		/*
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
		*/		
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);

		for (int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return La;
			if (hit.material->rough) {
				vec3 outRadiance = hit.material->ka * La;
				for (Light * light : lights) {
					Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
					float cosTheta = dot(hit.normal, light->direction);
					if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
						outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
						vec3 halfway = normalize(-ray.dir + light->direction);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}
				return outRadiance;
			}
			if (!hit.material->rough) {
				weight = Fresnel(hit.material->F0, dot(-ray.dir, hit.normal)) * weight;
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
			else return outRadiance;
		}
		
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad


	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad.Create(image);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}