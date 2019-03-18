//=============================================================================================
// Rotation with quaternion
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSourceSphere = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform float R;
	uniform vec4 quaternion;			
	layout(location = 0) in vec2 ll;	// Varying input: latitude-longitude

	out vec2 texCoord;

	vec4 qmul(vec4 q1, vec4 q2) {
		return vec4(q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz), q1.w * q2.w - dot(q1.xyz, q2.xyz));
	} 

	void main() {
		texCoord = vec2(ll.x / 2, ll.y) / 3.1415;
	    vec3 u = vec3(cos(ll.x) * sin(ll.y), sin(ll.x) * sin(ll.y), cos(ll.y)) * R;
		vec3 v = qmul(qmul(quaternion, vec4(u, 0)), vec4(-quaternion.xyz, quaternion.w)).xyz;
		gl_Position = vec4(v, 1);		
	}
)";

// fragment shader in GLSL
const char * const fragmentSourceSphere = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 texCoord;

	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = texture(textureUnit, texCoord);
	}
)";

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSourceLine = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 p;	

	void main() {
		gl_Position = vec4(p, 0, 1);		
	}
)";

// fragment shader in GLSL
const char * const fragmentSourceLine = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(1, 1, 1, 1); // extend RGB to RGBA
	}
)";

const float R = 0.6;

vec4 quaternion(float angle, vec3 axis) { 
	axis = normalize(axis) * sinf(angle / 2);
	return vec4(axis.x, axis.y, axis.z, cosf(angle / 2));
}

std::vector<vec4> ReadBMP(char * pathname, int& width, int& height) {	// read image as BMP files 
	FILE * file = fopen(pathname, "r");
	if (!file) {
		printf("%s does not exist\n", pathname);
		exit(-1);
	}
	unsigned short bitmapFileHeader[27];					// bitmap header
	fread(&bitmapFileHeader, 27, 2, file);
	if (bitmapFileHeader[0] != 0x4D42) {   // magic number
		printf("Not bmp file\n");
		exit(-1);
	}
	if (bitmapFileHeader[14] != 24) {
		printf("Only true color bmp files are supported\n");
		exit(-1);
	}
	width = bitmapFileHeader[9];
	height = bitmapFileHeader[11];
	unsigned int size = (unsigned long)bitmapFileHeader[17] + (unsigned long)bitmapFileHeader[18] * 65536;
	fseek(file, 54, SEEK_SET);

	std::vector<byte> byteImage(size);
	fread(&byteImage[0], 1, size, file); 	// read the pixels
	fclose(file);

	std::vector<vec4> image(width * height);

	// Swap R and B since in BMP, the order is BGR
	//????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
	int i = 0;
	for (int imageIdx = 0; imageIdx < size; imageIdx += 3) {
		if (byteImage[imageIdx + 2] == 255 && byteImage[imageIdx + 1] == 255 && byteImage[imageIdx] == 255)
			image[i++] = vec4(0, 0, 0, 0);
		else
			image[i++] = vec4(byteImage[imageIdx + 2] / 256.0f, byteImage[imageIdx + 1] / 256.0f, byteImage[imageIdx] / 256.0f, 1);
	}
	
	return image;



}

GPUProgram gpuProgram; // vertex and fragment shaders
vec4 q(0, 0, 0, 1);	// This represent the rotation
vec3 axis(0, 0, 1);

class Sphere {
	unsigned int vao;
	int nVertices;
	Texture * pTexture;
public:
	Sphere() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

		std::vector<vec2> vertices;
		const int N = 40, M = 40;
		nVertices = N * M * 6;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				vertices.push_back(vec2(i * M_PI * 2.0 / N, j * M_PI / M));
				vertices.push_back(vec2((i + 1) * M_PI * 2.0 / N, j * M_PI / M));
				vertices.push_back(vec2((i + 1) * M_PI * 2.0 / N, (j + 1) * M_PI / M));
				vertices.push_back(vec2(i * M_PI * 2.0 / N, j * M_PI / M));
				vertices.push_back(vec2((i + 1) * M_PI * 2.0 / N, (j + 1) * M_PI / M));
				vertices.push_back(vec2(i * M_PI * 2.0 / N, (j + 1) * M_PI / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * vertices.size(),  // # bytes
			&vertices[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		// Create texture
		int width, height;
		std::vector<vec4> image = ReadBMP("C:/Users/boros/Documents/GitHub/grafika/E/2_Geom_alg/Debug/earth1.bmp", width, height);
		pTexture = new Texture(width, height, image);
	}

	void Draw() {
		glEnable(GL_DEPTH_TEST);
		gpuProgram.Use();
		glBindVertexArray(vao);  // Draw call

		int location = glGetUniformLocation(gpuProgram.getId(), "R");
		if (location >= 0) glUniform1f(location, R);
		else printf("uniform R cannot be set\n");

		q.SetUniform(gpuProgram.getId(), "quaternion");
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");

		glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, nVertices /*# Elements*/);
	}
};

class Line {
	GPUProgram gpuProgram; // vertex and fragment shaders
	unsigned int vao;
public:
	Line() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
		
		Set(vec2(1, 0), vec2(1, 0));

		// create program for the GPU
		gpuProgram.Create(vertexSourceLine, fragmentSourceLine, "outColor");
	}
	void Set(vec2 start, vec2 end) {
		std::vector<vec2> vertices;
		vertices.push_back(start);
		vertices.push_back(end);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * 2,  // # bytes
			&vertices[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later
	}
	void Draw() {
		glDisable(GL_DEPTH_TEST);
		gpuProgram.Use();
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINES, 0 /*startIdx*/, 2 /*# Elements*/);
	}
};

Sphere * pSphere;
Line * pLine;

// Initialization
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(10);
	glEnable(GL_CULL_FACE);

	pSphere = new Sphere();
	pLine = new Line();

	// create program for the GPU
	gpuProgram.Create(vertexSourceSphere, fragmentSourceSphere, "outColor");

	printf("\nUsage: \n");
	printf("Mouse Left Button: Define axis of rotation\n");
	printf("Mouse Move: Modify angle of rotation\n");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	pSphere->Draw();
	pLine->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	float angle = cX * cX + cY * cY - axis.x * axis.x - axis.y * axis.y;
	q = quaternion(angle, axis);
	printf("Quaternion: %f, %f, %f, %f\n", q.w, q.x, q.y, q.z );
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (state == GLUT_DOWN) {
		vec2 p(cX, cY);
		float r2 = dot(p, p);
		if (r2 < R * R) {
			float cZ = -sqrtf(R * R - r2);
			axis = vec3(cX, cY, cZ);
			pLine->Set(p, p * (1 / R));
		}
		else {
			axis = vec3(cX, cY, 0);
			p = normalize(p);
			pLine->Set(p * R, p);
		}
		q = quaternion(0, axis);
	}
	else {
		q = vec4(0, 0, 0, 1);
	}
	printf("Quaternion: %f, %f, %f, %f\n", q.w, q.x, q.y, q.z);
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}


