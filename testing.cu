// nvcc -o testing -lglut -lGL -lm testing.cu

#include <stdio.h>
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

#define THREADS_X 	32
#define THREADS_Y	4
#define BLOCKS_X 	(IMG_WIDTH  / THREADS_X)
#define BLOCKS_Y 	(IMG_HEIGHT / THREADS_Y)

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

unsigned char *input, *output1, *output2;
unsigned char *dev_input, *dev_output, *dev_grey, *dev_edges;
unsigned int *houghspace1, *houghspace2;
int n_max1, b_max1, n_max2, b_max2;

static void handle_error(cudaError_t err) {
	if (err != 0) {
		fprintf(stderr, "%sparam\n", cudaGetErrorString(err));
		exit(err);
	}
}

void allocate_memory() {
	int n_pixels = IMG_WIDTH * IMG_HEIGHT;
	int image_size = sizeof(char) * n_pixels * 3;
	input = (unsigned char*) malloc(image_size);
	output1 = (unsigned char*) malloc(image_size);
	output2 = (unsigned char*) malloc(image_size);
	houghspace1 = (unsigned int*) malloc(HS_1_SIZE*sizeof(unsigned int));
	houghspace2 = (unsigned int*) malloc(HS_2_SIZE*sizeof(unsigned int));
	handle_error(cudaMalloc(&dev_input, image_size));
	handle_error(cudaMalloc(&dev_output, image_size));
	handle_error(cudaMalloc(&dev_grey, n_pixels));
	handle_error(cudaMalloc(&dev_edges, n_pixels));
}

__global__ void colour_threshold(unsigned char *bgr, unsigned char *greyscale, int n_pixels, unsigned char b, unsigned char g, unsigned char r, int threshold) {

	int thread_id = (blockIdx.x * blockDim.x) + (threadIdx.x);

	if (thread_id < n_pixels) {
		unsigned char *pixel = &bgr[thread_id * 3];
		int db = *pixel++ -b;
		int dg = *pixel++ -g;
		int dr = *pixel   -r;
		int distance = (db*db) + (dg*dg) + (dr*dr);

		if (distance <= threshold) {
			greyscale[thread_id] = 255;
		} else {
			greyscale[thread_id] = 0;
		}
	}
}

__global__ void greyscale_to_bgr(unsigned char *greyscale, unsigned char *bgr, int n_pixels) {

	int thread_id = (blockIdx.x * blockDim.x) + (threadIdx.x);

	if (thread_id < n_pixels) {
		bgr = &bgr[thread_id * 3];
		bgr[0] = greyscale[thread_id];
		bgr[1] = greyscale[thread_id];
		bgr[2] = greyscale[thread_id];
	}
}

__global__ void detect_edges(unsigned char *in, unsigned char *out, int n_pixels, int width, int height) {
	int thread_id = (blockIdx.x * blockDim.x) + (threadIdx.x);

	if (thread_id < n_pixels) {
		int b, d, f, h, r, x, y;

		y = thread_id / width;
		x = thread_id - (width * y);

		if (x==0 || y==0 || x==width-1 || y==height-1) {
			out[thread_id] = 0;
		} else {
			b = thread_id + width;
			d = thread_id - 1;
			f = thread_id + 1;
			h = thread_id - width;

			r = 0;

			if (in[thread_id]) {
				r+=4;
			}

			if (in[b]) {
				r=r-1;
			}
			if (in[d]) {
				r=r-1;
			}
			if (in[f]) {
				r=r-1;
			}
			if (in[h]) {
				r=r-1;
			}

			if (r>0) {
				out[thread_id]=255;
			} else {
				out[thread_id]=0;
			}
		}
	}
}

__global__ void kHough4(unsigned char const * const image, unsigned int* const houghspace1, unsigned int* const houghspace2) {
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//cache all possible values of M
	__shared__ float sh_m_array[THREADS_X*THREADS_Y];
	int const n = threadIdx.y*THREADS_X + threadIdx.x;
	sh_m_array[n] = (n-((HS_ANGLES-1)/2.0f)) / (float)((HS_ANGLES-1)/2.0f);
	__syncthreads();
	
	//read one image pixel from global memory
	unsigned char pixel = image[y*IMG_WIDTH + x];
	
	//vote for each non zero pixel
	if (pixel > 0)
	{
		for (int n = 0; n < HS_ANGLES; n++)
		{
			float const m = sh_m_array[n];
			int const b1 = x - (int)(y*m) + IMG_HEIGHT;
			int const b2 = y - (int)(x*m) + IMG_WIDTH;

			houghspace1[n*HS_1_WIDTH+b1]++;
			houghspace2[n*HS_2_WIDTH+b2]++;
		}
	}
}

void calcHough(unsigned int* houghspace1, unsigned int* houghspace2) {
	unsigned int *ghoughspace1, *ghoughspace2;

	//set cache configuration
	cudaFuncSetCacheConfig(kHough4, cudaFuncCachePreferShared);
	
	//allocate variables on device (GPU)
	handle_error(cudaMalloc((void**)&ghoughspace1, HS_1_SIZE*sizeof(unsigned int)));
	handle_error(cudaMalloc((void**)&ghoughspace2, HS_2_SIZE*sizeof(unsigned int)));

	//reset Hough space
	handle_error(cudaMemset(ghoughspace1, 0, HS_1_SIZE*sizeof(unsigned int)));
	handle_error(cudaMemset(ghoughspace2, 0, HS_2_SIZE*sizeof(unsigned int)));

	//run kernel
	dim3 dimBlock1(THREADS_X, THREADS_Y);
	dim3 dimGrid1(BLOCKS_X, BLOCKS_Y);
	kHough4<<<dimGrid1, dimBlock1>>>(dev_edges, ghoughspace1, ghoughspace2);

	//copy the GPU results back to CPU
	handle_error(cudaMemcpy(houghspace1, ghoughspace1, HS_1_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	handle_error(cudaMemcpy(houghspace2, ghoughspace2, HS_2_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	//free GPU memory
	handle_error(cudaFree(ghoughspace1));
	handle_error(cudaFree(ghoughspace2));
}

void calculate_output() {
	int n_pixels = IMG_WIDTH * IMG_HEIGHT;
	int image_size = sizeof(char) * n_pixels * 3;

	handle_error(cudaMemcpy(dev_input, input, image_size, cudaMemcpyHostToDevice));

	colour_threshold<<<IMG_WIDTH, IMG_HEIGHT>>>(dev_input, dev_grey, n_pixels, 0, 255, 255, 20000);
	greyscale_to_bgr<<<IMG_WIDTH, IMG_HEIGHT>>>(dev_grey, dev_output, n_pixels);
	handle_error(cudaMemcpy(output1, dev_output, image_size, cudaMemcpyDeviceToHost));

	detect_edges<<<IMG_WIDTH, IMG_HEIGHT>>>(dev_grey, dev_edges, n_pixels, IMG_WIDTH, IMG_HEIGHT);
	greyscale_to_bgr<<<IMG_WIDTH, IMG_HEIGHT>>>(dev_edges, dev_output, n_pixels);
	handle_error(cudaMemcpy(output2, dev_output, image_size, cudaMemcpyDeviceToHost));

	calcHough(houghspace1, houghspace2);
	unsigned int value, max_hs1=0, max_hs2=0;
	int n, b;
	for (n = 0; n < HS_ANGLES; n++) //found the maximum value in Hough space
	{
		for (b = 0; b < HS_1_WIDTH; b++)
		{
			value = houghspace1[n*HS_1_WIDTH+b];
			if (value > max_hs1)
			{
				max_hs1 = value;
				n_max1 = n; b_max1 = b;
			}
		}
		for (b = 0; b < HS_2_WIDTH; b++)
		{
			value = houghspace2[n*HS_2_WIDTH+b];
			if (value > max_hs2)
			{
				max_hs2 = value;
				n_max2 = n; b_max2 = b;
			}
		}
	}
}

void tidy_and_exit() {
	free(input);
	free(output1);
	free(output2);
	free(houghspace1);
	free(houghspace2);
	handle_error(cudaFree(dev_input));
	handle_error(cudaFree(dev_output));
	handle_error(cudaFree(dev_grey));
	handle_error(cudaFree(dev_edges));
	exit(0);
}

#define YSHIFT 95
static void display() {
	unsigned char *temp;

	glClear(GL_COLOR_BUFFER_BIT);

	glRasterPos2i(0, YSHIFT);
	glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);

	glRasterPos2i(IMG_WIDTH, YSHIFT);
	glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, output2);

	glRasterPos2i(2*IMG_WIDTH, YSHIFT);
	glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, output1);

	glRasterPos2i(3*IMG_WIDTH, YSHIFT);
	glDrawPixels(IMG_WIDTH, IMG_HEIGHT, GL_BGR_EXT, GL_UNSIGNED_BYTE, input);
	float tanth, b, x1, y1, x2, y2;
	x1 = 0.0; x2 = 1.0;
	tanth = tan((n_max1-45)*M_PI/180);
	b = (float)b_max1/(IMG_WIDTH+IMG_HEIGHT+2*HS_ANGLES);
	y1 = tanth*x1 + b; y2 = tanth*x2 + b;
	//scaling and shifted to raster coordinates
	x1 = x1*IMG_WIDTH + 3*IMG_WIDTH; x2 = x2*IMG_WIDTH + 3*IMG_WIDTH;
	y1 = y1*IMG_HEIGHT + YSHIFT; y2 = y2*IMG_HEIGHT + YSHIFT;
	glColor3ub(255, 0, 0); //red color
	glBegin(GL_LINES); //draw line
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	glEnd();
	x1 = 0.0; x2 = 1.0;
	tanth = tan((n_max2-45)*M_PI/180);
	b = (float)IMG_WIDTH/(2*IMG_HEIGHT) - (float)b_max2/(IMG_WIDTH+IMG_HEIGHT+2*HS_ANGLES);
	y1 = tanth*x1 + b; y2 = tanth*x2 + b;
	x1 = x1*IMG_WIDTH + 3*IMG_WIDTH; x2 = x2*IMG_WIDTH + 3*IMG_WIDTH;
	y1 = y1*IMG_HEIGHT + YSHIFT; y2 = y2*IMG_HEIGHT + YSHIFT;
	glBegin(GL_LINES);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	glEnd();

	temp = (unsigned char*) malloc(HS_1_SIZE*sizeof(unsigned char));
	for (int i = 0; i < HS_1_SIZE; i++) temp[i] = 8*houghspace1[i];
	glRasterPos2i(0, 0);
	glDrawPixels(HS_1_WIDTH, HS_ANGLES, GL_LUMINANCE, GL_UNSIGNED_BYTE, temp);
	free(temp);

	temp = (unsigned char*) malloc(HS_2_SIZE*sizeof(unsigned char));
	for (int i = 0; i < HS_2_SIZE; i++) temp[i] = 8*houghspace2[i];
	glRasterPos2i(HS_1_WIDTH, 0);
	glDrawPixels(HS_2_WIDTH, HS_ANGLES, GL_LUMINANCE, GL_UNSIGNED_BYTE, temp);
	free(temp);

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
	calculate_output();

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
