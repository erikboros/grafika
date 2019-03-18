//=============================================================================================
// 2D transformations with complex numbers
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	void main() {
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform vec3 color;			// uniform color
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
					0, wWy / 2, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};


Camera camera;			// 2D camera
GPUProgram gpuProgram;	// vertex and fragment shaders

struct Complex {
	float x, y;
	Complex(float x0 = 0, float y0 = 0) { x = x0, y = y0; }
	Complex operator+(Complex r) { return Complex(x + r.x, y + r.y); }
	Complex operator-(Complex r) { return Complex(x - r.x, y - r.y); }
	Complex operator*(Complex r) {
		return Complex(x * r.x - y * r.y, x * r.y + y * r.x);
	}
	Complex operator/(Complex r) {
		float l = r.x * r.x + r.y * r.y;
		return (*this) * Complex(r.x / l, -r.y / l);
	}
};

Complex Polar(float r, float phi) {
	return Complex(r * cos(phi), r * sin(phi));
}

class Object {
	unsigned int vao;	// vertex array object id
	unsigned int vbo;		// vertex buffer objects
	std::vector<Complex> points;
public:
	Object() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		points.push_back(Complex(0, 0));
		points.push_back(Complex(-1, -1));
		points.push_back(Complex(0, 1));
		points.push_back(Complex(1, -1));

		glBindVertexArray(vao);		// make it active
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed
		Animate(0);
	}

	void Animate(float t) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array

		std::vector<Complex> transPoints(points.size());
		for (unsigned int i = 0; i < points.size(); i++) {
			transPoints[i] = ((points[i] - Complex(1, -1)) * Polar(2, t) + Complex(1, -1) + Complex(2, 3)) * Polar(0.8f, -t / 2);
		}
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			transPoints.size() * 2 * sizeof(float), // number of the vbo in bytes
			&transPoints[0],		   // address of the data array on the CPU
			GL_DYNAMIC_DRAW);	   // copy to GPU
	}

	void Draw() {
		mat4 MVPTransform = camera.V() * camera.P();

		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		if (location >= 0) glUniform3f(location, 1, 0, 1);

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, points.size());	// draw a single triangle with vertices defined in vao
	}
};

// The virtual world: collection of two objects
Object * object;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	object = new Object;

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	object->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	object->Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}
