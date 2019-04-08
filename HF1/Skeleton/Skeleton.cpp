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
//
//Források: https://www.khronos.org/registry/OpenGL-Refpages/
//	http://cg.iit.bme.hu/portal/sites/default/files/oktatott%20t%C3%A1rgyak/sz%C3%A1m%C3%ADt%C3%B3g%C3%A9pes%20grafika/grafikus%20alap%20hw/sw/smoothtriangle.cpp?fbclid=IwAR2f3RM_IhACJD-fYc5pJpaOXuxcd65n_Ks4gmI4BW-Z8A_Bfg4OsNDZU8M
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

//Forrás: http://cg.iit.bme.hu/portal/sites/default/files/oktatott%20t%C3%A1rgyak/sz%C3%A1m%C3%ADt%C3%B3g%C3%A9pes%20grafika/grafikus%20alap%20hw/sw/smoothtriangle.cpp?fbclid=IwAR2f3RM_IhACJD-fYc5pJpaOXuxcd65n_Ks4gmI4BW-Z8A_Bfg4OsNDZU8M
class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(2, 2) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
	void Target(vec2 t) { wCenter = t; }
};

Camera2D camera;
bool follow = false;

float abs(vec2 v){
	return sqrt(v.x*v.x + v.y*v.y);
}

/**
* Catmull-Rom spline pontjainak(rt) számolása contorllpontok(points) közé
* Haszn: deklarálás, init() (onInit, gpuProgram.create után, draw())
*/
class CatmullRom {

public:
	std::vector<vec2> points;	//control points
	std::vector<float> ts;		//knots
	std::vector<vec2> rt;
	vec3 lineColor = vec3(0.3f, 0.3f, 0.3f);
	std::vector<vec2> solid;
	vec3 fillColor = vec3(0.3f, 0.3f, 0.3f);
	bool stationary = false;
	bool fill = false;
	float tension = -0.1;
	float bias = 0;
	float continuity = 1;

private:
	unsigned int vbo; // vertex buffer object
	bool updated = false;
	unsigned int vao;	   // virtual world on the GPU
	bool drawPoints = false;

public:
	void init() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glGenBuffers(1, &vbo);
	}

	void addPoint(vec2 cpt) {
		if (points.empty() || points[points.size()-1].x < cpt.x) {
			points.push_back(cpt);
			
		}
		else {
			int i = 0;
			while (points[i].x < cpt.x && i < points.size()-1) {
				i++;
			}
			points.insert(points.begin() + i, cpt);
			
		}

		ts.clear();
		ts.push_back(0.0f);
		for (int i = 1; i < points.size(); i++) {
			ts.push_back(ts[i - 1] + (abs(points[i] - points[i - 1]))); //*10?
		}
		updated = true;
	}

	float getMaxTs() {
		return ts[ts.size() - 1];
	}

	float getMaxDrawTs() {
		if (ts.size() >= 4) {
			return ts[ts.size() - 2];
		}
		else {
			return 0.0f;
		}
	}

	float getMinDrawTs() {
		if (ts.size() >= 2) {
			return ts[1];
		}
		else {
			return 0.0f;
		}
	}
	void setParam(float t, float b, float c) {
		tension = t;
		bias = b;
		continuity = c;
	}

	bool setDrawPoints(bool set) {
		drawPoints = set;
		return drawPoints;
	}

	void setLineColor(vec3 c) {
		lineColor = c;
	}

	void setFillColor(vec3 v) {
		fillColor = v;
	}

	void setFill(bool b) {
		fill = b;
	}

	vec2 hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		vec2 a2 = ((p1 - p0) * 3) * (1.0f / ((t1 - t0)*(t1 - t0))) - (v1 + v0 * 2) * (1.0f / (t1 - t0));
		vec2 a3 = ((p0 - p1) * 2) * (1.0f / ((t1 - t0)*(t1 - t0)*(t1 - t0))) + (v1 + v0) * (1.0f / ((t1 - t0)*(t1 - t0)));
		float tt = (t - t0);
		vec2 ret = a3 * (tt*tt*tt) + a2 * (tt*tt) + v0 * tt + p0;
		return ret;
	}

	//https://en.wikipedia.org/wiki/Kochanek%E2%80%93Bartels_spline
	vec2 r(float time) {
		if (points.size()>=4) {
			for (int i = 1; i < points.size() - 2; i++) {
				if (ts[i] <= time && time <= ts[i + 1]) {
					float t = tension;
					float b = bias;
					float c = continuity;
					vec2 v0 = (points[i] - points[i - 1])*((1 - t)*(1 + b)*(1 + c))*0.5f + (points[i+1] - points[i]) * ((1-t)*(1-b)*(1-c))*0.5f;
					vec2 v1 = (points[i + 1] - points[i])*((1 - t)*(1 + b)*(1 - c))*0.5f + (points[i+2]-points[i+1]) * ((1-t)*(1-b)*(1+c))*0.5f;
					return hermite(points[i], v0, ts[i], points[i + 1], v1, ts[i + 1], time);
				}
			}
		}
		else {
			return vec2(0, 0);
		}
	}

	vec2 v(float t) {
		if (points.size() >= 4) {
			if (ts[1] < t && t < ts[ts.size() - 2] - 0.001f) {
				vec2 v = r(t + 0.001f) - r(t);
				float absv = sqrt(v.x*v.x + v.y*v.y);
				return vec2(v.x / absv, v.y / absv);
			}
		}
		return vec2(0, 0);
	}

	vec2 n(float t) {
		if (points.size() >= 4) {
			if (ts[1] < t && t < ts[ts.size() - 2] - 0.001f) {
				vec2 v = r(t + 0.001f) - r(t);
				float absv = sqrt(v.x*v.x + v.y*v.y);
				vec2 e = vec2(v.x / absv, v.y / absv);
				return vec2(-e.y, e.x);
			}
			else {
				return vec2(0, 0);
			}
		}
		return vec2(0,0);
	}

	mat4 M(vec2 position, float rotate) {

		mat4 P = {	1,0,0,0,
					0,1,0,0,
					0,0,1,0,
					position.x,position.y,0,1 };

		mat4 R = {	cos(rotate),-sin(rotate),0,0,
					sin(rotate),cos(rotate),0,0,
					0,0,1,0,
					0,0,0,1 };

		if (stationary){
			return R * P;
		}
		else {
			return R * P * camera.V() * camera.P();
		}
	}

	void draw() {
		glLineWidth(2);
		glBindVertexArray(vao);		// make it active
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		int location;

		if (points.size() > 0) {

			if (points.size() >= 4){
				if (updated) {
					rt.clear();
					float max = getMaxDrawTs();
					for (float t = ts[1]; t < max; t += 0.05f) {
						vec2 tmp = vec2(0, 0);
						tmp = r(t);
						rt.push_back(tmp);
						if (fill) {
							solid.push_back(tmp);
							solid.push_back(vec2(tmp.x, -2));
						}
					}
					rt.push_back(points[points.size()-2]);
					updated = false;
				}

				///Draw spline
				glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
					rt.size() * sizeof(vec2),	// # bytes
					&rt[0],						// address
					GL_DYNAMIC_DRAW);			// we do not change later
				glEnableVertexAttribArray(0);	// AttribArray 0
				glVertexAttribPointer(0,		// vbo -> AttribArray 0
					2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
					0, NULL);					// stride, offset: tightly packed
				location = glGetUniformLocation(gpuProgram.getId(), "color");
				glUniform3f(location, lineColor.x, lineColor.y, lineColor.z); // 3 floats
				location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
				glUniformMatrix4fv(location, 1, GL_TRUE, &M(vec2(0,0),0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
				glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, rt.size() /*# Elements*/);

				///fill
				if (fill) {
					solid.push_back(vec2(points[points.size() - 1].x, -2));
					solid.push_back(vec2(points[0].x, -2));
					glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
						solid.size() * sizeof(vec2),	// # bytes
						&solid[0],						// address
						GL_DYNAMIC_DRAW);			// we do not change later
					glEnableVertexAttribArray(0);	// AttribArray 0
					glVertexAttribPointer(0,		// vbo -> AttribArray 0
						2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
						0, NULL);					// stride, offset: tightly packed
					location = glGetUniformLocation(gpuProgram.getId(), "color");
					glUniform3f(location, fillColor.x, fillColor.y, fillColor.z); // 3 floats
					location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
					glUniformMatrix4fv(location, 1, GL_TRUE, &M(vec2(0, 0), 0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
					glDrawArrays(GL_TRIANGLE_STRIP, 0 /*startIdx*/, solid.size() /*# Elements*/);
				}
				

			}
			
			if (drawPoints) {	///Draw control points
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
				glUniformMatrix4fv(location, 1, GL_TRUE, &M(vec2(0, 0), 0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
				glDrawArrays(GL_POINTS, 0 /*startIdx*/, points.size() /*# Elements*/);
			}
		}
	}
};

/**
* "Monocikli"
* Egy CatmullRom pointjait képes követni
* a spline adott idõbeli normálvektával a kerék sugarával eltolva
*/
class cycle {
private:
	unsigned int vao;
	unsigned int vbo;
	bool start = true;

	bool forward = true;
	float prevpos = 0;
	float prevtime = 0;
	float prevphi = 0;
	float v = 0.1f;

	vec3 lineColor = vec3(0, 0, 0);

	std::vector<vec2> cps;
	float wheelr = 0.05f;
	std::vector<vec2> rim;
	std::vector<vec2> body;
	std::vector<vec2> head;
	float headr = 0.025f;
	std::vector<vec2> legs;

	vec2 circle(float t, float r) {
		return vec2( r*cos(2*M_PI*t), r*sin(2 * M_PI*t));
	}

public:
	vec2 pos = vec2(0, 0);
	cycle() { //add points

		for (float t = 0.05f; t < M_PI; t+=0.05f){
			cps.push_back(circle(t, wheelr));
		}

		rim.push_back(vec2(0,wheelr));
		rim.push_back(vec2(0,-wheelr));
		rim.push_back(vec2(wheelr, 0));
		rim.push_back(vec2(-wheelr, 0));

		body.push_back(vec2(0, wheelr*1.4f));
		body.push_back(vec2(0, wheelr*3.5f));

		for (float t = 0.05f; t < M_PI; t += 0.05f) {
			vec2 p = circle(t, headr);
			head.push_back(vec2(p.x ,p.y + wheelr * 3.5 + headr));
		}
	}

	void init() {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
	}

	mat4 M(vec2 position, float rotate) {
		mat4 P = {	1,0,0,0,
					0,1,0,0,
					0,0,1,0,
					position.x,position.y,0,1 };

		mat4 R = {	cos(rotate),-sin(rotate),0,0,
					sin(rotate),cos(rotate),0,0,
					0,0,1,0,
					0,0,0,1 };
		
		return  R * P * camera.V() * camera.P();
	}

	void draw(CatmullRom cr, float sec) {
		if (cps.empty()){
			printf("cps empty\n");
			return;
		}
		if (start) {
			prevtime = sec;
			prevpos = cr.getMinDrawTs();
			start = false;
		}

		vec2 vv = vec2(0, 0);
		float maxt = cr.getMaxDrawTs();
		float mint = cr.getMinDrawTs();

		float dt = sec - prevtime;
		prevtime = sec;
		
		if (v < 0.15) v = 0.15;	//min speed
		if (v > 1.5) v = 1.5;	//max speed

		float ds = dt * v;

		if (forward && (prevpos + ds) > maxt) {
			forward = false;
		}
		if (!forward && (prevpos - ds) < mint) {
			forward = true;
		}

		if (forward){
			pos = cr.r(prevpos + ds) +(cr.n(prevpos + ds)*wheelr);
			prevpos = prevpos + ds;
			vv = cr.v(prevpos + ds);
			if (vv.x != 0) {
				v -= sin(atan(vv.y / vv.x))*0.001f;
			}
		}
		else {
			pos = cr.r(prevpos - ds) + (cr.n(prevpos + ds)*wheelr);
			prevpos = prevpos - ds;
			vv = cr.v(prevpos);
			if (vv.x != 0) {
				v += sin(atan(vv.y / vv.x))*0.001f;
			}
		}
		//v += 0.05f;	//saját erõ

		float dphi = v * dt / wheelr;
		float phi = 0;
		if (forward)
		{
			phi = prevphi + dphi;
			prevphi += dphi;
		}
		else {
			phi = prevphi - dphi;
			prevphi -= dphi;
		}

		legs.clear();
		legs.push_back(vec2(cos(-phi)*wheelr/2, sin(-phi)*wheelr / 2));
		legs.push_back(vec2(0, wheelr*1.4f));
		legs.push_back(vec2(cos(-phi)*-wheelr / 2 , sin(-phi)*-wheelr / 2));

		if (follow) {
			camera.Target(pos);
		}
		else {
			camera.Target(vec2(0, 0));
		}
		
		glLineWidth(1);
		///draw wheel
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			cps.size() * sizeof(vec2),	// # bytes
			&cps[0],					// address
			GL_DYNAMIC_DRAW);			// we do not change later
		glEnableVertexAttribArray(0);	// AttribArray 0
		glVertexAttribPointer(0,		// vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
			0, NULL);					// stride, offset: tightly packed			
		int location = glGetUniformLocation(gpuProgram.getId(), "color");	// Set color
		glUniform3f(location, lineColor.x, lineColor.y, lineColor.z);		// 3 floats
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");			// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &M(pos, 0).m[0][0]);		// Load a 4x4 row-major float matrix to the specified location
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, cps.size() /*# Elements*/);

		///draw rim
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			rim.size() * sizeof(vec2),	// # bytes
			&rim[0],					// address
			GL_DYNAMIC_DRAW);			// we do not change later
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");		// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &M(pos, phi).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glDrawArrays(GL_LINES, 0 /*startIdx*/, rim.size() /*# Elements*/);

		///draw body
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			body.size() * sizeof(vec2),	// # bytes
			&body[0],					// address
			GL_DYNAMIC_DRAW);			// we do not change later
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");		// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &M(pos, 0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, body.size() /*# Elements*/);

		///draw head
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			head.size() * sizeof(vec2),	// # bytes
			&head[0],					// address
			GL_DYNAMIC_DRAW);			// we do not change later
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");		// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &M(pos, 0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, head.size() /*# Elements*/);

		///draw legs
		glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
			legs.size() * sizeof(vec2),	// # bytes
			&legs[0],					// address
			GL_DYNAMIC_DRAW);			// we do not change later
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");		// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &M(pos, 0).m[0][0]);	// Load a 4x4 row-major float matrix to the specified location
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, legs.size() /*# Elements*/);
	}
};


CatmullRom crbg; //************************************************************
Texture tbg;

CatmullRom cr;
cycle t1;
bool go = true;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(5); //Forrás: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glPointSize.xml
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");	// create program for the GPU

	
	crbg.init();
	crbg.setParam(0.5, 0, 0.9);
	crbg.stationary = true;
	crbg.setLineColor(vec3(1, 1, 1));
	crbg.setFillColor(vec3(0.3, 0.6, 0.1));
	crbg.setFill(true);
	for (float i = -2; i < 2.1; i+=0.5f){
		crbg.addPoint(vec2(i, (float)(rand()%10)/10));
	}

	cr.init();
	cr.setDrawPoints(true);

	t1.init();
}

// Window has become invalid: Redraw
void onDisplay() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;

	glClearColor(0.4, 0.6, 1.0, 1);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	crbg.draw();
	cr.draw();
	
	if (cr.points.size() >= 4 && go) {
		t1.draw(cr, sec);
	}

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		follow = !follow;
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
			cr.addPoint(vec2(cX, cY));
			glutPostRedisplay();
		}
		
		break;
	case GLUT_MIDDLE_BUTTON: //printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  
		//printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); 
		
		break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();
}
