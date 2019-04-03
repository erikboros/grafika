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
#include <algorithm>    // std::sort

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

bool compareVec2x(vec2 a, vec2 b) {
	return (a.x < b.x);
}

class CatmullRom {

public:
	std::vector<vec2> points;	//control points
	std::vector<float> ts;		//knots
	std::vector<vec2> rt;

private:
	unsigned int vbo; // vertex buffer object
	boolean updated = false;

public:
	void init() {
		glGenVertexArrays(2, &vao[0]);	// get 1 vao id
		//glBindVertexArray(vao[0]);		// make it active
		//glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glGenBuffers(1, &vbo);
	}

	void addPoint(vec2 cpt, float t) {
		points.push_back(cpt);
		ts.push_back(t);

		std::sort (points.begin(), points.end(), compareVec2x); //TODO: WAT
		updated = true;
	}

	int getMaxTs() {
		//using monoton building
		return (int)floor(ts[ts.size()-1]);
	}

	vec2 hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		vec2 a2 = ((p1 - p0) * 3) * (1.0f / ((t1 - t0)*(t1 - t0))) - (v1 + v0 * 2) * (1.0f / (t1 - t0));
		vec2 a3 = ((p0 - p1) * 2) * (1.0f / ((t1 - t0)*(t1 - t0)*(t1 - t0))) + (v1 + v0) * (1.0f / ((t1 - t0)*(t1 - t0)));
		float tt = (t - t0);
		return a3 * (tt*tt*tt) + a2 * (tt*tt) + v0 * tt + p0;
	}

	vec2 r(float t) {
		if (points.size() >= 2) {
			float error = 0.04f;
			if (t>getMaxTs() - error && t<getMaxTs()+error){
				return points[points.size() - 2];
			}
			for (int i = 0; i < points.size() - 2; i++) {
				if (ts[i] <= t && t <= ts[i + 1]) {
					vec2 v0 = ((points[i + 1] - points[i]) * (1.0f / (ts[i + 1] - ts[i])) + (points[i] - points[i + 1]) * (1.0f / (ts[i] - ts[i + 1])))*0.5f; // *(1-tau)
					vec2 v1 = ((points[i + 2] - points[i + 1]) * (1.0f / (ts[i + 2] - ts[i + 1])) + (points[i + 1] - points[i + 2]) * (1.0f / (ts[i + 1] - ts[i + 2])))*0.5f;
					return hermite(points[i], v0, ts[i], points[i + 1], v1, ts[i + 1], t);
				}
			}
		}
		else {
			return vec2(0,0);
		}
	}

	vec2 n(float t) {
		if (points.size() >= 2) {
			if (ts[0] < t && t < ts[ts.size() - 2] - 0.001f) {
				vec2 v = r(t + 0.001f) - r(t);
				float absv = sqrt(v.x*v.x + v.y*v.y);
				vec2 e = vec2(v.x / absv, v.y / absv);
				return vec2(-e.y, e.x);
			}
		}
		return vec2(0,0);
	}

	void draw() {
		if (points.size() > 0) {

			float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
				0, 1, 0, 0,    // row-major!
				0, 0, 1, 0,
				0, 0, 0, 1 };
			int location;

			if (points.size() > 2){

				if (updated) {
					rt.clear();

					float max = getMaxTs();

					for (float t = 0.0f; t < max; t += 0.1f) {
						vec2 tmp = r(t);
						rt.push_back(tmp);
						//printf("t:%f (%f,%f)\n", t, tmp.x, tmp.y);
					}
					rt.push_back(points[points.size()-2]);
					updated = false;
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
				location = glGetUniformLocation(gpuProgram.getId(), "color");
				glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

				location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
				glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
																			//glBindVertexArray(vao);		// Draw call
				glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);

			}
			
			

			///Draw control points
			glBindVertexArray(vao[1]);		// make it active
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
				points.size() * sizeof(vec2),	// # bytes
				&points[0],						// address
				GL_DYNAMIC_DRAW);			// we do not change later
			glEnableVertexAttribArray(0);	// AttribArray 0
			glVertexAttribPointer(0,		// vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
				0, NULL);					// stride, offset: tightly packed
			location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 0.0f, 0.0f, 1.0f); // 3 floats
			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
																		//glBindVertexArray(vao);		// Draw call
			glDrawArrays(GL_POINTS, 0 /*startIdx*/, points.size() /*# Elements*/);
		}
	}
};


class cycle {
private:
	unsigned int tvao;
	unsigned int vbo;
	unsigned int startTime;
	//float v = 0.2f;

	std::vector<vec2> cps;
	float wheelr = 0.05f;

	std::vector<vec2> rim;

	std::vector<vec2> head;
	float headr = 0.02f;

	boolean first = true;

	vec2 circle(float t, float r) {
		return vec2( r*cos(2*M_PI*t), r*sin(2 * M_PI*t));
	}

public:
	cycle() {
		for (float t = 0.05f; t < M_PI; t+=0.05f){
			cps.push_back(circle(t, wheelr));
		}
	}

	void init() {
		glGenVertexArrays(1, &tvao);
		glGenBuffers(1, &vbo);
	}

	void draw(CatmullRom cr, float sec) {
		if (cps.empty())
		{
			printf("cps empty\n");
			return;
		}
		if (first)
		{
			startTime = sec;
			first = false;
		}
		
		vec2 pos;
		sec = sec - startTime;

		int maxt = cr.getMaxTs() - 1;
		int h = sec / maxt;
		if (h % 2 == 0) { //elõre megyünk
			sec = (float)sec - (float)(h*maxt);
		}
		else { //hátra
			sec = maxt - (sec - h * maxt);
		}

		//vec2 n = cr.n(sec);
		pos = cr.r(sec) + cr.n(sec)*wheelr;
		//printf("%f n: (%f,%f)\n", sec, n.x, n.y);

		glBindVertexArray(tvao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			cps.size() * sizeof(vec2),	// # bytes
			&cps[0],						// address
			GL_DYNAMIC_DRAW);			// we do not change later

		glEnableVertexAttribArray(0);	// AttribArray 0
		glVertexAttribPointer(0,		// vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
			0, NULL);					// stride, offset: tightly packed

										// Set color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 0.0f, 1.0f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,		// új x
			0, 1, 0, 0,		// új y
			0, 0, 1, 0,		// új Z
			pos.x, pos.y, 0, 1 };	//eltolás

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
																	//glBindVertexArray(vao);		// Draw call
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, cps.size() /*# Elements*/);
	}
};


CatmullRom cr;
int n=0;

cycle t1;

// Initialization, create an OpenGL context
void onInitialization() {

	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2);
	glPointSize(5);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");

	cr.init();
	t1.init();

}

// Window has become invalid: Redraw
void onDisplay() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;

	
	glClearColor(0.2, 0.2, 0.2, 1);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	//printf("draw call\n");
	cr.draw();
	//printf("rt size: %d\n", cr.rt.size());
	if (cr.points.size() > 2)
	{
		//printf("thing draw call\n");
		t1.draw(cr, sec);
	}
	

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
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
		//printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		if (buttonStat == "pressed"){
			cr.addPoint(vec2(cX, cY), n++);
			glutPostRedisplay();
		}
		
		break;
	case GLUT_MIDDLE_BUTTON: //printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  
		//printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); 

		if (buttonStat == "pressed") {
			glutPostRedisplay();
		}
		break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();
}
