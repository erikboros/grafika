//=============================================================================================
// Bezier surface
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4  MVP; 
	uniform vec3  wEye;         // pos of eye
	uniform int phong;

	layout(location = 0) in vec3  vtxPos;            // pos in modeling space
	layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

	out vec3 wNormal;		    // normal in world space
	out vec3 wView;             // view in world space

	void main() {
		gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		if (phong == 1) {
			wView  = wEye - vtxPos;
			wNormal = vtxNorm;
		}
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	precision highp float;

	// Light
	const vec3 wLight = vec3(5, 5, 4);	// directional light source
	const vec3 La = vec3(2, 2, 2), Le = vec3(3, 3, 3);
	const vec3 kd = vec3(0, 0, 1), ks = vec3(2, 2, 2), ka = vec3(0, 0, 0.3f); // diffuse, specular, ambient ref
	const float shine = 100.0f;

	uniform vec3 color;
	uniform int phong;

	in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	out vec4 fragmentColor; // output goes to frame buffer

	void main() {
		if (phong == 1) {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);

			vec3 calcColor = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
			fragmentColor = vec4(calcColor, 1);
		} else {
			fragmentColor = vec4(color, 1);
		}
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
	bool perspective;
public:
	Camera() {
		asp = 1;
		fov = 60.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
		perspective = false;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0,   0,   0,   1);
	}
	mat4 P() { // projection matrix
		if (perspective) {
			return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
				0, 1 / tan(fov / 2), 0, 0,
				0, 0, -(fp + bp) / (bp - fp), -1,
				0, 0, -2 * fp*bp / (bp - fp), 0);
		}
		else {
			float scale = tan(fov / 2) * length(wEye - wLookat);
			return mat4(1 / scale, 0, 0, 0,
				0, 1 / scale, 0, 0,
				0, 0, -2/(bp - fp), 0,
				0, 0, -(fp+bp)/(bp - fp), 1);
		}
	}

	mat4 Vinv() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return mat4(u.x, u.y, u.z, 0,
			v.x, v.y, v.z, 0,
			w.x, w.y, w.z, 0,
			0, 0, 0, 1) * TranslateMatrix(wEye);
	}
	mat4 Pinv() { // view matrix: translates the center to the origin
		float scale = tan(fov / 2) * length(wEye - wLookat);
		return mat4(scale, 0, 0, 0,
			0, scale, 0, 0,
			0, 0, -(bp - fp) / 2, 0,
			0, 0, -(fp + bp) / 2, 1);
	}

	bool Pick(vec3& p, float cx, float cy) {
		float scale = tan(fov / 2) * length(wEye - wLookat);
		vec4 cp = vec4(p.x, p.y, p.z, 1) * V() * P();
		if ((cp.x - cx) *(cp.x  - cx) + (cp.y - cy) *(cp.y - cy) < 0.05)  return true;
		return false;
	}
	void Move(vec3& p, float cx, float cy) {
		float scale = tan(fov / 2) * length(wEye - wLookat);
		vec4 cp = vec4(p.x, p.y, p.z, 1) * V() * P() ;
		vec4 wp = vec4(cx, cy, cp.z, 1) * Pinv() * Vinv();
		p = vec3(wp.x, wp.y, wp.z);
	}
};

//---------------------------
struct VertexData {
//---------------------------
	vec3 position, normal;
};

//---------------------------
class Geometry {
//---------------------------
	unsigned int vao, type;        // vertex array object
protected: 
	int nVertices;
public:
	Geometry(unsigned int _type) {
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Load() {}
	virtual void Draw() {
		glBindVertexArray(vao);
		Load();
		glDrawArrays(type, 0, nVertices);
	}
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
	unsigned int vbo;
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create() {
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		Load();
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position)); 
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}
	void Load() {
		const int N = 20, M = 20;
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = N * M * 6;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
	}
};

// Bezier using Bernstein polynomials
const int nControlPoints = 4;
vec3 wCtrlPoints[nControlPoints][nControlPoints] = {
	vec3(-3, -3, 0), vec3(-3, -1, 0), vec3(-3, 1, 0), vec3(-3, 3, 0),
	vec3(-1, -3, 0), vec3(-1, -1, 3), vec3(-1, 1, 0), vec3(-1, 3, 0),
	vec3(1, -3, 0), vec3(1, -1, 0), vec3(1, 1, 0), vec3(1, 3, 0),
	vec3(3, -3, 0), vec3(3, -1, 0), vec3(3, 1, 0), vec3(3, 3, 0)
};

//---------------------------
class BezierSurface : public ParamSurface {
//---------------------------
	float B(int i, float t) {
		int n = nControlPoints - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
	float Bd(int i, float t) {
		int n = nControlPoints - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		if (i == 0)      return - choose * n * pow(1 - t, n - 1);
		else if (i == n) return choose * n * pow(t, n - 1);
		else             return choose * (i * pow(t, i - 1) * pow(1 - t, n - i) - pow(t, i) * (n - i) * pow(1 - t, n - i - 1));
	}
public:
	BezierSurface() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vec3(0, 0, 0);
		vec3 du(0, 0, 0), dv(0, 0, 0);
		for (unsigned int i = 0; i < nControlPoints; i++) {
			for (unsigned int j = 0; j < nControlPoints; j++) {
				vd.position = vd.position + wCtrlPoints[i][j] * B(i, u) * B(j, v);
				du = du + wCtrlPoints[i][j] * Bd(i, u) * B(j, v);
				dv = dv + wCtrlPoints[i][j] * B(i, u) * Bd(j, v);
			}
		}
		vd.normal = cross(du, dv);
		return vd;
	}
	void Render(Camera& camera) {
		mat4 MVP = camera.V() * camera.P();
		MVP.SetUniform(gpuProgram.getId(), "MVP");

		camera.wEye.SetUniform(gpuProgram.getId(), "wEye");

		int phongLocation = glGetUniformLocation(gpuProgram.getId(), "phong");
		if (phongLocation >= 0) glUniform1i(phongLocation, 1);
		glEnable(GL_DEPTH_TEST);
		Draw();
	}
};

//---------------------------
class ControlPoints : public Geometry {
//---------------------------
	unsigned int vbo;
public:
	ControlPoints() : Geometry(GL_POINTS) { 
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, NULL, NULL); // attribute array, components/attribute, component type, normalize?, stride, offset
	}
	void Load() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = nControlPoints * nControlPoints;
		glBufferData(GL_ARRAY_BUFFER,  nVertices * sizeof(vec3), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
	}
	void Render(Camera& camera) {
		glPointSize(10.0f);

		mat4 MVP = camera.V() * camera.P();
		MVP.SetUniform(gpuProgram.getId(), "MVP");

		camera.wEye.SetUniform(gpuProgram.getId(), "wEye");

		int phongLocation = glGetUniformLocation(gpuProgram.getId(), "phong");
		if (phongLocation >= 0) glUniform1i(phongLocation, 0);

		glDisable(GL_DEPTH_TEST);
		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);
		Draw();
	}
};

//---------------------------
class Frame : public Geometry {
//---------------------------
	unsigned int vbo;
public:
	Frame() : Geometry(GL_LINE_LOOP) {
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float frame[] = { -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0 };
		nVertices = 4;
		glBufferData(GL_ARRAY_BUFFER, sizeof(frame), &frame[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, NULL, NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	}
	void Render() {
		glPointSize(10.0f);

		int phongLocation = glGetUniformLocation(gpuProgram.getId(), "phong");
		if (phongLocation >= 0) glUniform1i(phongLocation, 0);

		glDisable(GL_DEPTH_TEST);
		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 1);

		mat4 MVP = mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		MVP.SetUniform(gpuProgram.getId(), "MVP");
		Draw();
	}
};

BezierSurface * pSurface;
ControlPoints * pControlPoints;
Frame * pFrame;
Camera camera[4];
 
// Initialization, create an OpenGL context
void onInitialization() {
	glDisable(GL_CULL_FACE);

	pSurface = new BezierSurface();
	pControlPoints = new ControlPoints();
	pFrame = new Frame();

	// Camera
	camera[0].wEye = vec3( 0, -6, 0); camera[0].wLookat = vec3(0, 0, 0); camera[0].wVup = vec3(0, 0, 1); camera[0].perspective = false;
	camera[1].wEye = vec3( 6,  0, 0); camera[1].wLookat = vec3(0, 0, 0); camera[1].wVup = vec3(0, 0, 1); camera[1].perspective = false;
	camera[2].wEye = vec3( 0,  0, 6); camera[2].wLookat = vec3(0, 0, 0); camera[2].wVup = vec3(0, 1, 0); camera[2].perspective = false;
	camera[3].wEye = vec3(-3, -3, 6); camera[3].wLookat = vec3(0, 0, 0); camera[3].wVup = vec3(0, 1, 0); camera[3].perspective = true;

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");

	printf("\nUsage: \n");
	printf("Mouse Any Button: Pick and move control point\n");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.0f, 0.2f, 0.2f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	const int boundary = 2;
	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	pSurface->Render(camera[0]);
	pControlPoints->Render(camera[0]);
	pFrame->Render();

	glViewport(windowWidth / 2, windowHeight / 2, windowWidth / 2, windowHeight / 2);
	pSurface->Render(camera[1]);
	pControlPoints->Render(camera[1]);
	pFrame->Render();

	glViewport(0, 0, windowWidth / 2, windowHeight / 2);
	pSurface->Render(camera[2]);
	pControlPoints->Render(camera[2]);
	pFrame->Render();

	glViewport(windowWidth / 2, 0, windowWidth / 2, windowHeight / 2);
	pSurface->Render(camera[3]);
	pControlPoints->Render(camera[3]);
	pFrame->Render();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

vec3 * pickedPoint = NULL;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
	if (state == GLUT_DOWN) {  
		float cX = 2.0f * pX / windowWidth - 1;	
		float cY = 1.0f - 2.0f * pY / windowHeight; // flip y axis
		int cameraIdx = 0;

		if (cX <= 0 && cY > 0) {
			cX = 2 * cX + 1;
			cY = 2 * cY - 1;
			cameraIdx = 0;
		} else if (cX > 0 && cY > 0) {
			cX = 2 * cX - 1;
			cY = 2 * cY - 1;
			cameraIdx = 1;
		} else if (cX <= 0 && cY <= 0) {
			cX = 2 * cX + 1;
			cY = 2 * cY + 1;
			cameraIdx = 2;
		}
		for (unsigned int i = 0; i < nControlPoints; i++) {
			for (unsigned int j = 0; j < nControlPoints; j++) {
				if (camera[cameraIdx].Pick(wCtrlPoints[i][j], cX, cY)) {
					pickedPoint = &wCtrlPoints[i][j];
					return;
				}
			}
		}
		glutPostRedisplay();     // redraw
	}
	else {
		pickedPoint = NULL;
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	int cameraIdx = 0;

	if (cX <= 0 && cY > 0) {
		cX = 2 * cX + 1;
		cY = 2 * cY - 1;
		cameraIdx = 0;
	} else if (cX > 0 && cY > 0) {
		cX = 2 * cX - 1;
		cY = 2 * cY - 1;
		cameraIdx = 1;
	} else if (cX <= 0 && cY <= 0) {
		cX = 2 * cX + 1;
		cY = 2 * cY + 1;
		cameraIdx = 2;
	}
	if (pickedPoint) {
		camera[cameraIdx].Move(*pickedPoint, cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
