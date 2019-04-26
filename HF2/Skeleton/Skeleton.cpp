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
public:
	vec3 ka, kd, ks;
	float  shininess;
	bool rough;
	Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess, bool _rough) : ka(_ka), kd(_kd), ks(_ks), rough(_rough) {
		shininess = _shininess; 
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
		hit.t = -1;
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
	PolyTube() {
		n = 3;
		d = 0.4;
	}
	void setMaterial(Material* _material) {
		material = _material;
	}
	PolyTube(Material* _material) {
		material = _material;
		n = 3;
		d = 0.4;
	}
	void addSide() {
		n++;
		//printf("polytube: sides=%d\n", n);
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
	vec3 Le;// La;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
		//La = vec3(0.4, 0.3, 0.3);
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;
PolyTube mirror;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;

	vec3 F0gold;
	vec3 F0silver;
	
public:
	bool gold = true;
	void setGold(bool b) {
		gold = b;
	}
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		vec3 one(1, 1, 1);
		vec3 n(0.17, 0.35, 1.5); //arany
		vec3 kappa(3.1, 2.7, 1.9);
		F0gold.x = ((n.x - one.x)*(n.x - one.x) + kappa.x * kappa.x) /
			((n.x + one.x)*(n.x + one.x) + kappa.x * kappa.x);
		F0gold.y = ((n.y - one.y)*(n.y - one.y) + kappa.y * kappa.y) /
			((n.y + one.y)*(n.y + one.y) + kappa.y * kappa.y);
		F0gold.z = ((n.z - one.z)*(n.z - one.z) + kappa.z * kappa.z) /
			((n.z + one.z)*(n.z + one.z) + kappa.z * kappa.z);

		n=vec3(0.14, 0.16, 0.13); //ezüst
		kappa=vec3(4.1, 2.3, 3.1);
		F0silver.x = ((n.x - one.x)*(n.x - one.x) + kappa.x * kappa.x) /
			((n.x + one.x)*(n.x + one.x) + kappa.x * kappa.x);
		F0silver.y = ((n.y - one.y)*(n.y - one.y) + kappa.y * kappa.y) /
			((n.y + one.y)*(n.y + one.y) + kappa.y * kappa.y);
		F0silver.z = ((n.z - one.z)*(n.z - one.z) + kappa.z * kappa.z) /
			((n.z + one.z)*(n.z + one.z) + kappa.z * kappa.z);

		La = vec3(0.8, 0.8, 0.8);
		vec3 lightDirection(0, 0, 1);
		vec3 Le(0.8, 0.8, 0.8);
		lights.push_back(new Light(lightDirection, Le));

		vec3 ka(0.4, 0.5, 0.6);
		vec3 kd(0.5, 0.5, 0.9); 
		vec3 ks(0.1, 0.1, 0.9);
		Material * material = new Material(ka, kd, ks, 5, true);
		objects.push_back(new Sphere(vec3(0, 0.15, 0), 0.15, material));
		ka = vec3(0.4, 0.5, 0.6);
		kd = vec3(0.9, 0.5, 0.5);
		ks = vec3(0.9, 0.1, 0.9);
		material = new Material(ka, kd, ks, 50, true);
		objects.push_back(new Sphere(vec3(-0.1, -0.05, 0), 0.08, material));
		ka = vec3(0.4, 0.5, 0.6);
		kd = vec3(0.9, 0.5, 0.9);
		ks = vec3(0.1, 0.9, 0.1);
		material = new Material(ka, kd, ks, 20, true);
		objects.push_back(new Sphere(vec3(0.1, -0.05, 0), 0.1, material));


		Material * smaterial = new Material(ka, kd, ks, 0, false);
		mirror.setMaterial(smaterial);
		mirror.putSides(objects);

		ka= vec3(0.5, 0.5, 0.5);
		kd= vec3(0.3, 0.3, 0.3);
		ks= vec3(0.2, 0.2, 0.2);//0.1
		Material * white = new Material(ka, kd, ks, 2, true);
		objects.push_back(new Wall(vec3(-1, 1, 0), vec3(-1, -1, 0), vec3(1, 1,0), vec3(1, -1, 0), white)); //backdrop


	}

	void clear() {
		objects.clear();
		lights.clear();
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

	vec3 Fresnel(vec3 inDir, vec3 normal){//vec3 F0, float cosTheta) {
		//return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
		//printf("fresnel: F0: (%f,%f,%f)\n", F0.x, F0.y, F0.z);

		vec3 one(1, 1, 1);
		vec3 ret(0, 0, 0);
		float cosa = -dot(inDir, normal);
		if (gold) {
			ret = F0gold + (one - F0gold) * pow(1 - cosa, 5);
		}
		else {
			ret = F0silver + (one - F0silver) * pow(1 - cosa, 5);
		}
		
		//printf("fresnel: (%f,%f,%f)\n", ret.x, ret.y, ret.z);
		return ret;
	}

	vec3 reflect(vec3 I, vec3 N) {
		return I - N * 2.0f * dot(N, I);
	}

	vec3 trace(Ray ray, int maxdepth=10) {	
		vec3 weight(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		Light *light = lights[0];

		for (int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * La;
			if (hit.material->rough) {
				outRadiance = outRadiance + weight * hit.material->ka * La;
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + weight * light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + weight * light->Le * hit.material->ks * pow(cosDelta, hit.material->shininess);
				}
			}
			if (!hit.material->rough) { //reflective
				//printf("F0: (%f,%f,%f)\n", hit.material->F0.x, hit.material->F0.y, hit.material->F0.z);
				weight = weight * Fresnel(ray.dir, hit.normal);//Fresnel(hit.material->F0, dot(-ray.dir, hit.normal)) * weight;
				//printf("d: %d weight: (%f,%f,%f)\n",d,  weight.x, weight.y, weight.z);
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
			else {
				//printf("weight: (%f,%f,%f)\n", weight.x, weight.y, weight.z);
				return outRadiance;
			}
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
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void setImage(std::vector<vec4>& image) {
		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad


	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;
std::vector<vec4> image(windowWidth * windowHeight);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	printf("Rendering...\n");
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
	//printf("ondisp\n");
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	bool pressed = false;
	if (key == 'a') {
		mirror.addSide();
		printf("Mirror side added\n");
		scene.clear();
		pressed = true;
	}
	if (key == 's' && scene.gold) {
		scene.clear();
		scene.setGold(false);
		printf("Mirrors set to silver\n");
		pressed = true;

	}
	if (key == 'g' && !scene.gold) {
		scene.clear();
		scene.setGold(true);
		printf("Mirrors set to gold\n");
		pressed = true;
	}

	if (pressed){
		scene.build();
		printf("Rendering...\n");
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		scene.render(image);
		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
		fullScreenTexturedQuad.setImage(image);
		glutPostRedisplay();
	}
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