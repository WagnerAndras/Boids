#include "MyApp.h"
// #include "SDL_log.h"
#include "includes/GLUtils.hpp"
#include "includes/SDL_GLDebugMessageCallback.h"
#include "includes/ProgramBuilder.h"


#include "device_launch_parameters.h"
//#include <__clang_cuda_builtin_vars.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "includes/helper_cuda.h"
#include "src/includes/Camera.h"

#include <cmath>
#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/matrix.hpp>
#include <imgui.h>

#include <glm/trigonometric.hpp>
#include <vector>
#include <random>

CMyApp::CMyApp()
{
  // Explicitly set device 0
  cudaSetDevice(0);

  // Get number of threads per block
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  m_thread_num = deviceProp.maxThreadsPerBlock;
}

CMyApp::~CMyApp()
{
}

void CMyApp::SetupDebugCallback()
{
	// Enable and set the debug callback function if we are in debug context
	GLint context_flags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
	if (context_flags & GL_CONTEXT_FLAG_DEBUG_BIT)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
		glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DONT_CARE, 0, nullptr, GL_FALSE);
		glDebugMessageCallback(SDL_GLDebugMessageCallback, nullptr);
	}
}

void CMyApp::InitShaders()
{
	if (m_programBoidID != 0) return; // don't reinitialize

	m_programBoidID = glCreateProgram();
	ProgramBuilder{ m_programBoidID }
		.ShaderStage(GL_VERTEX_SHADER, "Boid.vert")
		.ShaderStage(GL_FRAGMENT_SHADER, "Boid.frag")
		.Link();
}

void CMyApp::CleanShaders()
{
	glDeleteProgram(m_programBoidID);
}


void CMyApp::InitGeometry()
{
	if (m_BoidGPU.count > 0) return; // don't reinitialize

	MeshObject<glm::vec3> m_BoidMeshCPU;

	// Simple triangle
	m_BoidMeshCPU.vertexArray = {
		glm::vec3(  1,  0,   0   ),
		glm::vec3( -1,  0.5, 0.5 ),
		glm::vec3( -1, -0.5, 0.5 ),
		glm::vec3( -1, -0.5,-0.5 ),
		glm::vec3( -1,  0.5,-0.5 ),
	};

	m_BoidMeshCPU.indexArray =
	{
		// sides
		0, 1, 2,
		0, 2, 3,
		0, 3, 4,
		0, 4, 1,
		// back
		1, 3, 2,
		1, 4, 3,
	};

	m_BoidGPU = CreateGLObjectFromMesh( m_BoidMeshCPU, { { 0, offsetof( glm::vec3,x), 3, GL_FLOAT}});
}

void CMyApp::CleanGeometry()
{
	CleanOGLObject( m_BoidGPU );
}

void CMyApp::InitPositions()
{
	// Initializing the Boid positions and rotations 
	std::random_device r; // seed source
	std::seed_seq seeds{r(), r(), r(), r(), r(), r(), r(), r()};
	std::mt19937 mt(seeds); // random engine with seeds
	std::uniform_real_distribution<float> randOffset(-1.0f, 1.0f);
	std::uniform_real_distribution<float> randAngle(-glm::pi<float>(), glm::pi<float>());
	
	// initialize each boid with a posiotion and an angle
	Boid* boids = (Boid*)malloc(m_inst_num * sizeof(Boid));
	for (int i = 0; i < m_inst_num; ++i)
	{
		float angle = randAngle(mt);
		boids[i] = Boid {
				glm::vec2(randOffset(mt), randOffset(mt)),
				glm::vec2(glm::cos(angle), glm::sin(angle)),
			};
	}

	// Allocate vectors in device memory
  checkCudaErrors( cudaMalloc(&d_boids, m_inst_num * sizeof(Boid)) );
  checkCudaErrors( cudaMalloc(&d_sdirs, m_inst_num * sizeof(glm::vec2)) );
  
  // Put initial positions on GPU
  checkCudaErrors( cudaMemcpy(d_boids, boids, m_inst_num * sizeof(Boid), cudaMemcpyHostToDevice) );
	free(boids);

	// world matrix buffer for interop
  // Create buffer object and register it with CUDA
  glGenBuffers(1, &world_matricesBO);
  glBindBuffer(GL_ARRAY_BUFFER, world_matricesBO);
  glBufferData(GL_ARRAY_BUFFER, m_inst_num * sizeof(glm::mat4), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  checkCudaErrors( cudaGraphicsGLRegisterBuffer(
  			&world_matricesBO_CUDA,
  			world_matricesBO,
  			cudaGraphicsMapFlagsWriteDiscard) );
}

bool CMyApp::Init()
{
	SetupDebugCallback();

	// Set a bluish clear color
	// glClear() will use this for clearing the color buffer.
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

	InitShaders();
	InitGeometry();
	InitPositions();

	// Other

	glEnable(GL_CULL_FACE);	 // Enable discarding the back-facing faces.
	glCullFace(GL_BACK);     // GL_BACK: facets facing away from camera, GL_FRONT: facets facing towards the camera

	// Camera
	m_camera.SetView(
		glm::vec3(0, 2, 2),// From where we look at the scene - eye
		glm::vec3(0, 0, 0),	// Which point of the scene we are looking at - at
		glm::vec3(0, 1, 0)	// Upwards direction - up
	);
	m_cameraManipulator.SetCamera(&m_camera);

	return true;
}

void CMyApp::Clean()
{
	glDeleteBuffers(1, &world_matricesBO);
	
	CleanShaders();
	CleanGeometry();

	cudaFree(d_boids);
	cudaFree(d_sdirs);
	cudaGraphicsUnregisterResource(world_matricesBO_CUDA);
}

void CMyApp::Restart()
{
	glDeleteBuffers(1, &world_matricesBO);
	
	cudaFree(d_boids);
	cudaFree(d_sdirs);
	cudaGraphicsUnregisterResource(world_matricesBO_CUDA);

	Init();
}

__global__ void SteerBoids(Boid* g_boids, glm::vec2* sdirs, int INST_NUM, SteeringParams sp)
{
	extern __shared__ Boid boids[];

	int bd = blockDim.x;
	int i = threadIdx.x;
	int g = blockIdx.x * bd + i; // Global index
	
	for (int j = 0; j < gridDim.x; j++)
	{
		int ind = j * bd + i;
		if (ind < INST_NUM)
		{
			boids[ind] = g_boids[ind]; // copy into block shared memory
		}
	}
	
	__syncthreads(); // wait for all copies to happen

	if (g >= INST_NUM) return;

	glm::vec2 separation = glm::vec2(0.0f);
	glm::vec2 alignment = glm::vec2(0.0f);
	glm::vec2 cohesion = glm::vec2(0.0f);
	for (int j = 0; j < INST_NUM; j++)
	{

		// see if it's the same
		if (j == g) continue;

		glm::vec2 to_other = boids[j].pos - boids[g].pos;
		float dst = glm::length(to_other);

		// see if it's inside the perception radius
		if (dst > sp.perception_distance) continue;

		glm::vec2 to_other_normalized = to_other / dst;

		// see if it's in the field of view
		if (glm::dot(boids[g].dir, to_other_normalized) <= sp.half_fov_cos) continue;



		// Separation
		separation += -to_other_normalized * glm::sqrt(sp.perception_distance / dst - 1.0f);

		// Alignment
		alignment += boids[j].dir;

		// Cohesion
		cohesion += to_other_normalized;
	}

	// Set steering direction
	sdirs[g] = glm::normalize(boids[g].dir +
														separation * sp.separation_weight +
														alignment * sp.alignment_weight +
														cohesion * sp.cohesion_weight);
}

__global__ void MoveBoids(Boid* boids, glm::vec2* sdirs, int INST_NUM, glm::mat4* world_matrices, MovementParams mp, float DeltaTimeInSec, glm::mat4 view_proj)
{
	int b = blockIdx.x;
	int i = threadIdx.x;
	int g = b * blockDim.x + i; // Global index
	if (g >= INST_NUM) return;
	
	glm::vec3 dir = glm::vec3(boids[g].dir, 0.0f);
	glm::vec3 sdir = glm::vec3(sdirs[g], 0.0f);

	// turn towards the steering direction
	float angle = glm::acos(glm::dot(dir, sdir)) * glm::min(DeltaTimeInSec * mp.angular_velocity, 1.0f);
	glm::vec3 axis = glm::cross(dir, sdir);
	if (abs(axis.z) > 0.01f) {
		glm::vec2 ndir = glm::rotate(angle, axis) * glm::vec4(dir, 1.0f);
		boids[g].dir = glm::normalize(ndir);
	}


	// move in the new direction
	boids[g].pos += boids[g].dir * mp.velocity * DeltaTimeInSec;
	boids[g].pos.x = std::fmodf(boids[g].pos.x + 3.0f, 2.0f) - 1.0f;
	boids[g].pos.y = std::fmodf(boids[g].pos.y + 3.0f, 2.0f) - 1.0f;

	world_matrices[g] =
		view_proj
		*
		glm::translate(glm::vec3(boids[g].pos, 0))
		*
		glm::rotate(atan2(boids[g].dir.y, boids[g].dir.x), glm::vec3(0, 0, 1))
		*
		glm::scale(glm::vec3(0.01));
}

void CMyApp::Update( const SUpdateInfo& updateInfo )
{
	m_ElapsedTimeInSec = updateInfo.ElapsedTimeInSec;
	m_DeltaTimeInSec = updateInfo.DeltaTimeInSec;

	m_cameraManipulator.Update(m_DeltaTimeInSec);
	
	// Set steering direction for all boids in kernel
	// TODO It should work if there's less than m_inst_num * sizeof(Boid) shared memory
  int block_num = (m_inst_num - 1) / m_thread_num + 1;
	SteerBoids<<<block_num, m_thread_num, m_inst_num * sizeof(Boid)>>>(d_boids, d_sdirs, m_inst_num, m_steering_params);
  checkCudaErrors( cudaGetLastError()  );

  // Map buffer object
	glm::mat4* world_matrices;
	checkCudaErrors( cudaGraphicsMapResources(1, &world_matricesBO_CUDA, 0) );
  
  size_t num_bytes;
  checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void**)&world_matrices,
                                       &num_bytes,
                                       world_matricesBO_CUDA));

  // Execute kernel
  // Set new positions based on the steering directions
	MoveBoids<<<block_num, m_thread_num>>>(d_boids, d_sdirs, m_inst_num, world_matrices, m_movement_params, m_DeltaTimeInSec, m_camera.GetViewProj());
  checkCudaErrors( cudaGetLastError()  );

  // Unmap buffer object
  checkCudaErrors( cudaGraphicsUnmapResources(1, &world_matricesBO_CUDA, 0) );	
}

void CMyApp::Render()
{
	// töröljük a frampuffert (GL_COLOR_BUFFER_BIT)...
	// ... és a mélységi Z puffert (GL_DEPTH_BUFFER_BIT)
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(m_programBoidID);
	glBindVertexArray(m_BoidGPU.vaoID);
	glBindBuffer(GL_ARRAY_BUFFER, world_matricesBO);

	for (int i = 0; i < 4; ++i) {
    	glEnableVertexAttribArray(1 + i);
    	glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
    	glVertexAttribDivisor(1 + i, 1); // Advance once per instance
	}

	glDrawElementsInstanced(GL_TRIANGLES, m_BoidGPU.count, GL_UNSIGNED_INT, 0, m_inst_num);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);
}

int inst_num = 1024;
void CMyApp::RenderGUI()
{
	// ImGui::ShowDemoWindow();
	if (ImGui::Begin("FPS"))
	{
		const float refresh_time = 0.5f;
		static float timer = 0;
		static int   frameCount = 0;
		static float fps = 0;
		static float avgFrameTime = 0.0f;

		timer += m_DeltaTimeInSec;
		++frameCount;
		if (timer > refresh_time) {
			avgFrameTime = timer / frameCount;
			fps = frameCount / timer;
			timer = 0;
			frameCount = 0;
		}
		ImGui::Text("FPS: %d", static_cast<int>(fps));
		ImGui::Text("ms %f", avgFrameTime);
	}
	ImGui::End();
	
	if (ImGui::Begin("Settings"))
	{
		ImGui::SliderInt("Boid number", &inst_num, 1, 2048);
		if (ImGui::Button("Restart"))
		{
			m_inst_num = inst_num;
			CMyApp::Restart();
		}
// the number of boids (5%)
// the FOV of boids (10%)
		static float fov = 180.0f; // in degrees
		if (ImGui::SliderFloat("FOV", &fov, 0.0f, 360.0f))
		{
			m_steering_params.half_fov_cos = std::cos((fov * 0.5f * M_PI / 180.0f));
		}
		ImGui::SliderFloat("Perception distance", &m_steering_params.perception_distance, 0.0f, 1.0f);
		ImGui::SliderFloat("Separation weight", &m_steering_params.separation_weight, 0.0f, 10.0f);
		ImGui::SliderFloat("Alignment weight", &m_steering_params.alignment_weight, 0.0f, 10.0f);
		ImGui::SliderFloat("Cohesion weight", &m_steering_params.cohesion_weight, 0.0f, 10.0f);
		ImGui::SliderFloat("Angular velocity", &m_movement_params.angular_velocity, 0.0f, 10.0f);
		ImGui::SliderFloat("Velocity", &m_movement_params.velocity, 0.0f, 1.0f);

// the weights of boid rules: (5%)
// separation
// alignment
// cohesion
// the type of initial distribution of boids (e.g., uniform randomization) (10%).
	}
	ImGui::End();
}

// https://wiki.libsdl.org/SDL2/SDL_KeyboardEvent
// https://wiki.libsdl.org/SDL2/SDL_Keysym
// https://wiki.libsdl.org/SDL2/SDL_Keycode
// https://wiki.libsdl.org/SDL2/SDL_Keymod

void CMyApp::KeyboardDown(const SDL_KeyboardEvent& key)
{
	if (key.repeat == 0) // Triggers only once when held
	{
		if (key.keysym.sym == SDLK_F5 && key.keysym.mod & KMOD_CTRL) // CTRL + F5
		{
			CleanShaders();
			InitShaders();
		}
		if (key.keysym.sym == SDLK_F1) // F1
		{
			GLint polygonModeFrontAndBack[2] = {};
			// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glGet.xhtml
			glGetIntegerv(GL_POLYGON_MODE, polygonModeFrontAndBack); // Query the current polygon mode. It gives the front and back modes separately.
			GLenum polygonMode = (polygonModeFrontAndBack[0] != GL_FILL ? GL_FILL : GL_LINE); // Switch between FILL and LINE
			// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glPolygonMode.xhtml
			glPolygonMode(GL_FRONT_AND_BACK, polygonMode); // Set the new polygon mode
		}
	}
	m_cameraManipulator.KeyboardDown(key);
}

void CMyApp::KeyboardUp(const SDL_KeyboardEvent& key)
{
	m_cameraManipulator.KeyboardUp(key);
}

// https://wiki.libsdl.org/SDL2/SDL_MouseMotionEvent

void CMyApp::MouseMove(const SDL_MouseMotionEvent& mouse)
{
	m_cameraManipulator.MouseMove(mouse);
}

// https://wiki.libsdl.org/SDL2/SDL_MouseButtonEvent

void CMyApp::MouseDown(const SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseUp(const SDL_MouseButtonEvent& mouse)
{
}

// https://wiki.libsdl.org/SDL2/SDL_MouseWheelEvent

void CMyApp::MouseWheel(const SDL_MouseWheelEvent& wheel)
{
	m_cameraManipulator.MouseWheel(wheel);
}

// New window size
void CMyApp::Resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h);
	m_camera.SetAspect( static_cast<float>(_w) / _h );
}

// Other SDL events
// https://wiki.libsdl.org/SDL2/SDL_Event

void CMyApp::OtherEvent( const SDL_Event& ev )
{
}
