// nvcc -o testing -lglut -lGL -lm testing.cu

#include <stdio.h>
#include <math.h>
#ifndef __GNUC__
	#include <GL/freeglut.h>
	typedef __int32 int32_t;
	typedef unsigned __int32 uint32_t;
	typedef unsigned __int16 uint16_t;
#else
	#include <stdint.h>
	#include <GL/glut.h>
#endif

// image sizes
#define IMG_WIDTH	400
#define IMG_HEIGHT	300
#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

// Hough transform parameters
#define HS_STEP		1
#define HS_ANGLES	(90/HS_STEP + 1)
#define HS_1_WIDTH	(IMG_HEIGHT + IMG_WIDTH + IMG_HEIGHT)
#define HS_1_SIZE	(HS_1_WIDTH * HS_ANGLES)
#define HS_2_WIDTH	(IMG_WIDTH + IMG_HEIGHT + IMG_WIDTH)
#define HS_2_SIZE	(HS_2_WIDTH * HS_ANGLES)

typedef struct {
	uint32_t filesz;
	uint16_t creator1;
	uint16_t creator2;
	uint32_t bmp_offset;
} bmp_file_header;

typedef struct {
	uint32_t header_sz;
	int32_t width;
	int32_t height;
	uint16_t nplanes;
	uint16_t bitspp;
	uint32_t compress_type;
	uint32_t bmp_bytesz;
	int32_t hres;
	int32_t vres;
	uint32_t ncolors;
	uint32_t nimpcolors;
} bmp_info_header;

unsigned char *input;

void allocate_memory() {
	int n_pixels = IMG_WIDTH * IMG_HEIGHT;
	int image_size = sizeof(char) * n_pixels * 3;
	input = (unsigned char*) malloc(image_size);
}

void tidy_and_exit() {
	free(input);
	exit(0);
}

#define YSHIFT 95
static void display() {
	glClear(GL_COLOR_BUFFER_BIT);

	glRasterPos2i(0, YSHIFT);
	//glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);

	glRasterPos2i(IMG_WIDTH, YSHIFT);
	//glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);

	glRasterPos2i(2*IMG_WIDTH, YSHIFT);
	//glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);

	glRasterPos2i(3*IMG_WIDTH, YSHIFT);
	glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);
	float tanth, b, x1, y1, x2, y2;
	x1 = 0.0; x2 = 1.0;
	tanth = tan((39-45)*M_PI/180);
	b = (float)464/(IMG_WIDTH+IMG_HEIGHT+2*HS_ANGLES);
	y1 = tanth*x1 + b; y2 = tanth*x2 + b;
	x1 = x1*IMG_WIDTH + 3*IMG_WIDTH; x2 = x2*IMG_WIDTH + 3*IMG_WIDTH;
	y1 = y1*IMG_HEIGHT + YSHIFT; y2 = y2*IMG_HEIGHT + YSHIFT;
	glColor3ub(255, 0, 0);
	glBegin(GL_LINES);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	glEnd();
	x1 = 0.0; x2 = 1.0;
	tanth = tan((62-45)*M_PI/180);
	b = (float)IMG_WIDTH/(2*IMG_HEIGHT) - (float)441/(IMG_WIDTH+IMG_HEIGHT+2*HS_ANGLES);
//printf("b=%5.3f\n", b);
	y1 = tanth*x1 + b; y2 = tanth*x2 + b;
	x1 = x1*IMG_WIDTH + 3*IMG_WIDTH; x2 = x2*IMG_WIDTH + 3*IMG_WIDTH;
	y1 = y1*IMG_HEIGHT + YSHIFT; y2 = y2*IMG_HEIGHT + YSHIFT;
	glBegin(GL_LINES);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	glEnd();

	glFlush();
}

void load_image_data(char* filename) {
	FILE *f = fopen(filename, "rb");
	if (!f) {
		printf("failed to open file\n");
		exit(0);
	}

	char signature[2];
	fread(signature, 1, 2, f);
	bmp_file_header bfh;
	fread(&bfh, 1, sizeof(bmp_file_header), f);
	bmp_info_header bih;
	fread(&bih, 1, sizeof(bmp_info_header), f);

	if (bih.width != IMG_WIDTH || bih.height != IMG_HEIGHT)
	{
		printf("Error: unexpected image size (%d x %d)\n", bih.width, bih.height);
		printf("       expected (%d x %d)\n", IMG_WIDTH, IMG_HEIGHT);
		fclose(f);
		exit(0);
	}

	allocate_memory();

	fread(input, 1, IMG_WIDTH*IMG_HEIGHT*3, f);

	fclose(f);
}

static void key_pressed(unsigned char key, int x, int y) {
	switch (key) {
	case 27: // escape
		tidy_and_exit();
		break;
	default:
		printf("\nPress escape to exit\n");
		break;
	}
}

int main(int argc, char **argv) {
	load_image_data("2pencils.bmp");

	glutInit(&argc, argv);
	glutInitWindowSize(IMG_WIDTH*4, IMG_HEIGHT+YSHIFT);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutCreateWindow("Hough lines");
	glutDisplayFunc(display);
	glutKeyboardFunc(key_pressed);
	glOrtho(0, IMG_WIDTH*4, 0, IMG_HEIGHT, 0, 1);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glutMainLoop();

	tidy_and_exit();

	return 0;
}
