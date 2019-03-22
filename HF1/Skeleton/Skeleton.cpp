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
unsigned int vao;	   // virtual world on the GPU

class Strip {

public: std::vector<vec2> points;	//control points
public:	int nVertices = 0;
private: std::vector<float> ts;		//knots

public: void init() {
			glGenVertexArrays(1, &vao);	// get 1 vao id
			glBindVertexArray(vao);		// make it active

			unsigned int vbo;		// vertex buffer object
			glGenBuffers(1, &vbo);	// Generate 1 buffer
			glBindBuffer(GL_ARRAY_BUFFER, vbo);

			/**
			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				points.size() * sizeof(vec2),  // # bytes
				&points[0],	      	// address
				GL_DYNAMIC_DRAW);	// we do not change later
			

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed
				*/
		}

		void addPoint(vec2 p) {
			points.push_back(p);
			nVertices = points.size();
			ts.push_back(points.size()); //ti
		}
		void addPoints(std::vector<vec2>& p) {
			points = p;
			nVertices = points.size();
		}
		void removeLast() {
			points.pop_back();
			nVertices = points.size();
		}

		void draw() {
			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				points.size() * sizeof(vec2),  // # bytes
				&points[0],	      	// address
				GL_DYNAMIC_DRAW);	// we do not change later

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed

			// Set color to (0, 1, 0) = green
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

			float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
									  0, 1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

			glBindVertexArray(vao);  // Draw call
			glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, nVertices /*# Elements*/);
		}

		float L(int i, float t) {
			float Li = 1.0f;
			for (int j = 0; j < points.size(); j++) {
				if (j != i) {
					Li *= (t - ts[j]) / (ts[i] - ts[j]);
				}
			}
			return Li;
		}
		vec2 r(float t) {
			vec2 rr(0, 0);
			for (int i = 0; i < points.size(); i++) {
				rr.x += points[i].x * L(i, t);
				rr.y += points[i].y * L(i, t);
			}
			return rr;
		}
		void drawLagr() {
			std::vector<vec2> rt;
			for (float t = -10.0f; t < 10.0f; t += 0.1f) {
				vec2 tmp = r(t);
				rt.push_back(tmp);
				printf("t:%f (%f,%f)\n",t , tmp.x, tmp.y);
			}

			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				rt.size() * sizeof(vec2),  // # bytes
				&rt[0],	      	// address
				GL_DYNAMIC_DRAW);	// we do not change later

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed

			// Set color
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

			float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
									  0, 1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

			glBindVertexArray(vao);  // Draw call
			glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);
		}
};

class CatmullRom {

public: 
	std::vector<vec2> points;	//control points
	std::vector<float> ts;		//knots


public: void init() {
	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
}
public:
	void addPoint(vec2 cpt, float t) {
		points.push_back(cpt);
		ts.push_back(t);
	}

	vec2 div(vec2 a, vec2 b) {
		return vec2(a.x / b.x, a.y / b.y);
	}
	
	vec2 hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		vec2 a2 = ((p1 - p0) * 3) * (1.0f / ((t1 - t0)*(t1 - t0))) - (v1 + v0 * 2) * (1.0f / (t1 - t0));
		vec2 a3 = ((p0 - p1) * 2) * (1.0f / ((t1 - t0)*(t1 - t0)*(t1 - t0))) + (v1 + v0) * (1.0f / ((t1 - t0)*(t1 - t0)));
		float tt = (t - t0);
		return a3 * (tt*tt*tt) + a2 * (tt*tt) + v0 * tt + p0;
	}
	
	vec2 r(float t) {
		for (int i = 0; i < points.size()-2; i++){
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec2 v0 = ((points[i + 1] - points[i]) * (1.0f/(ts[i + 1] - ts[i])) +  (points[i] - points[i + 1]) * (1.0f/(ts[i] - ts[i + 1])))*0.5f; // *(1-tau)
				vec2 v1 = ((points[i + 2] - points[i + 1]) * (1.0f/(ts[i + 2] - ts[i + 1])) + (points[i + 1] - points[i + 2]) * (1.0f/(ts[i + 1] - ts[i + 2])))*0.5f;
				return hermite(points[i], v0, ts[i], points[i+1], v1, ts[i+1], t);
			}
		}
	}
	

	void draw() {
		std::vector<vec2> rt;
		for (float t = 0.0f; t < 10.0f; t += 0.1f) {
			vec2 tmp = r(t);
			rt.push_back(tmp);
			printf("t:%f (%f,%f)\n", t, tmp.x, tmp.y);
		}

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			rt.size() * sizeof(vec2),  // # bytes
			&rt[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		// Set color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);
	}
};

class Hermite {
	//public: std::vector<vec2> points;	//control points
	vec2 p0;
	vec2 v0;
	float t0 = 0.0f;
	vec2 p1;
	vec2 v1;
	float t1 = 1.0f;

public: 
	void init() {
		p0 = vec2(-0.5f, -0.5f);
		v0 = vec2(0.0, 4.0);
		p1 = vec2(0.5, 0.5);
		v1 = vec2(1.0, 0.0);
		//printf("(%f,%f)\n", p0.x, p0.y);
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	vec2 div(vec2 a, vec2 b) {
		//printf("div: (%f,%f)/(%f,%f)\n", a.x, a.y, b.x, b.y);
		return vec2(a.x / b.x, a.y / b.y);
	}
	vec2 r(float t) {
		vec2 a2 = ((p1 - p0) * 3) * (1.0f/((t1-t0)*(t1 - t0))) - (v1 + v0 * 2) * (1.0f/(t1 - t0));
		vec2 a3 = ((p0 - p1) * 2) * (1.0f/((t1 - t0)*(t1 - t0)*(t1 - t0))) + (v1 + v0) * (1.0f/((t1 - t0)*(t1 - t0)));
		float tt = (t - t0);
		printf("a2: (%f,%f)\n", a2.x, a2.y);
		printf("a3: (%f,%f)\n", a3.x, a3.y);
		return a3*(tt*tt*tt) + a2*(tt*tt) + v0*tt + p0;
	}

	void draw() {
		std::vector<vec2> rt;


		
		for (float t = 0.0f; t < 1.0f; t += 0.05f) {
			vec2 tmp = r(t);
			rt.push_back(tmp);
			printf("t:%f (%f,%f)\n", t, tmp.x, tmp.y);
		}
		

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			rt.size() * sizeof(vec2),  // # bytes
			&rt[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
		// Set color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats
		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);
	}


};


//Strip s;
CatmullRom cr;
int n=0;
//Hermite hr;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");

	/**
	s.addPoint(vec2(-0.4f, -0.4f));
	s.addPoint(vec2(-0.2f, 0.0f));
	s.addPoint(vec2(0.0f, 0.1f));
	s.addPoint(vec2(0.5f, 0.5f));
	*/

	
	cr.addPoint(vec2(0.0f, -1.0f), n++);
	cr.addPoint(vec2(-0.5f, -0.5f), n++);
	cr.addPoint(vec2(-0.2f, 0.0f), n++);
	cr.addPoint(vec2(0.2f, 0.5f), n++);
	cr.addPoint(vec2(0.5f, 0.5f), n++);

	cr.init();
	


	//hr.init();
	//s.init();
	cr.init();


}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	//s.draw();
	//s.drawLagr();

	cr.draw();

	//printf("draw: nvertices:%d", s.nVertices);

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
			//s.addPoint(vec2(cX, cY));
			cr.addPoint(vec2(cX, cY), n++);
			glutPostRedisplay();
		}
		
		break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  
		printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); 
		if (buttonStat == "pressed") {
			//s.removeLast();
			glutPostRedisplay();
		}
		break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
