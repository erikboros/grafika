//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
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
// Nev    : 
// Neptun : 
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao[2];	   // virtual world on the GPU


class CatmullRom {

public: 
	std::vector<vec2> points;	//control points
	std::vector<float> ts;		//knots

private:
	unsigned int vbo; // vertex buffer object

public: void init() {
	glGenVertexArrays(2, &vao[0]);	// get 1 vao id
	//glBindVertexArray(vao[0]);		// make it active
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glGenBuffers(1, &vbo);
}
public:
	void addPoint(vec2 cpt, float t) {
		points.push_back(cpt);
		ts.push_back(t);
	}
	
	vec2 hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		vec2 a2 = ((p1 - p0) * 3) * (1.0f / ((t1 - t0)*(t1 - t0))) - (v1 + v0 * 2) * (1.0f / (t1 - t0));
		vec2 a3 = ((p0 - p1) * 2) * (1.0f / ((t1 - t0)*(t1 - t0)*(t1 - t0))) + (v1 + v0) * (1.0f / ((t1 - t0)*(t1 - t0)));
		float tt = (t - t0);
		return a3 * (tt*tt*tt) + a2 * (tt*tt) + v0 * tt + p0;
	}
	
	vec2 r(float t) {
		if (points.size() >= 2) {
			for (int i = 0; i < points.size() - 2; i++) {
				if (ts[i] <= t && t <= ts[i + 1]) {
					vec2 v0 = ((points[i + 1] - points[i]) * (1.0f / (ts[i + 1] - ts[i])) + (points[i] - points[i + 1]) * (1.0f / (ts[i] - ts[i + 1])))*0.5f; // *(1-tau)
					vec2 v1 = ((points[i + 2] - points[i + 1]) * (1.0f / (ts[i + 2] - ts[i + 1])) + (points[i + 1] - points[i + 2]) * (1.0f / (ts[i + 1] - ts[i + 2])))*0.5f;
					return hermite(points[i], v0, ts[i], points[i + 1], v1, ts[i + 1], t);
				}
			}
		}
	}

	void draw() {
		if (points.size() > 0) {
			std::vector<vec2> rt;

			float max = 0;
			for (int i = 0; i < ts.size(); i++){
				if (ts[i] > max)
					max = ts[i];
			}
			printf("%f", max);

			for (float t = 0.0f; t < max+1; t += 0.1f) {
				vec2 tmp = r(t);
				rt.push_back(tmp);
				printf("t:%f (%f,%f)\n", t, tmp.x, tmp.y);
			}

			///Draw spline
			glBindVertexArray(vao[0]);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
				rt.size() * sizeof(vec2),	// # bytes
				&rt[0],						// address
				GL_DYNAMIC_DRAW);			// we do not change later

			glEnableVertexAttribArray(0);	// AttribArray 0
			glVertexAttribPointer(0,		// vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
				0, NULL);					// stride, offset: tightly packed

			// Set color
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

			float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
									  0, 1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
			//glBindVertexArray(vao);		// Draw call
			glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);


			///Draw control points
			glBindVertexArray(vao[1]);		// make it active
			//glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
				points.size() * sizeof(vec2),	// # bytes
				&points[0],						// address
				GL_DYNAMIC_DRAW);			// we do not change later
			glEnableVertexAttribArray(0);	// AttribArray 0
			glVertexAttribPointer(0,		// vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
				0, NULL);					// stride, offset: tightly packed
			location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1.0f, 0.0f, 0.0f); // 3 floats
			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
				//glBindVertexArray(vao);		// Draw call
			glDrawArrays(GL_POINTS, 0 /*startIdx*/, points.size() /*# Elements*/);
			
		}
	}
};


CatmullRom cr;
int n=0;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2);
	glPointSize(5);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");

	cr.init();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.2, 0.2, 0.2, 1);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	cr.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   
		printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		if (buttonStat == "pressed"){
			cr.addPoint(vec2(cX, cY), n++);
			glutPostRedisplay();
		}
		
		break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  
		printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); 
		if (buttonStat == "pressed") {
			glutPostRedisplay();
		}
		break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
