#include "MyApp.h"
#include "SDL_log.h"
#include "includes/GLUtils.hpp"
#include "includes/SDL_GLDebugMessageCallback.h"
#include "includes/ProgramBuilder.h"


#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "includes/helper_cuda.h"

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
#include <iostream>
#include <vector>
#include <random>

CMyApp::CMyApp()
{
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

	MeshObject<glm::vec2> m_BoidMeshCPU;

	// Simple triangle
	m_BoidMeshCPU.vertexArray = {
		glm::vec2( -1, 1 ),
		glm::vec2( -1,-1 ),
		glm::vec2(  1, 0 ),
	};

	m_BoidMeshCPU.indexArray =
	{
		0, 1, 2
	};

	m_BoidGPU = CreateGLObjectFromMesh( m_BoidMeshCPU, { { 0, offsetof( glm::vec2,x), 2, GL_FLOAT}});
}

void CMyApp::CleanGeometry()
{
	CleanOGLObject( m_BoidGPU );
}

void CMyApp::InitPositions()
{
  // Explicitly set device 0
  cudaSetDevice(0);

	// Initializing the Boid positions and rotations 
	std::random_device r; // seed source
	std::seed_seq seeds{r(), r(), r(), r(), r(), r(), r(), r()};
	std::mt19937 mt(seeds); // random engine with seeds
	std::uniform_real_distribution<float> randOffset(-1.0f, 1.0f);
	std::uniform_real_distribution<float> randAngle(-glm::pi<float>(), glm::pi<float>());
	
	// initialize each boid with a posiotion and an angle
	Boid m_boids[m_inst_num];
	for (int i = 0; i < m_inst_num; ++i)
	{
		float angle = randAngle(mt);
		m_boids[i] = Boid {
				glm::vec2(randOffset(mt), randOffset(mt)),
				glm::vec2(glm::cos(angle), glm::sin(angle)),
			};
	}

	// Allocate vectors in device memory
  checkCudaErrors( cudaMalloc(&d_boids, m_inst_num * sizeof(Boid)) );
  checkCudaErrors( cudaMalloc(&d_sdirs, m_inst_num * sizeof(glm::vec2)) );
  
  // Put initial positions on GPU
  checkCudaErrors( cudaMemcpy(d_boids, m_boids, m_inst_num * sizeof(Boid), cudaMemcpyHostToDevice) );

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

void CMyApp::Update( const SUpdateInfo& updateInfo )
{
	m_ElapsedTimeInSec = updateInfo.ElapsedTimeInSec;
	m_DeltaTimeInSec = updateInfo.DeltaTimeInSec;
}

__global__ void SteerBoids(Boid* boids, glm::vec2* sdirs, int INST_NUM, SteeringParams sp)
{
	int i = threadIdx.x; // Thread number

	boids[i].dir = glm::normalize(boids[i].dir);

	glm::vec2 separation = glm::vec2(0.0f);
	glm::vec2 alignment = glm::vec2(0.0f);
	glm::vec2 cohesion = glm::vec2(0.0f);
	for (int j = 0; j < INST_NUM; j++)
	{

		// see if it's the same
		if (j == i) continue;

		glm::vec2 to_other = boids[j].pos - boids[i].pos;
		float dst = glm::length(to_other);

		// see if it's inside the perception radius
		if (dst > sp.perception_distance) continue;

		glm::vec2 to_other_normalized = to_other / dst;

		// see if it's in the field of view
		if (glm::dot(boids[i].dir, to_other_normalized) <= sp.half_fov_cos) continue;



		// Separation
		separation += -to_other_normalized * glm::sqrt(sp.perception_distance / dst - 1.0f);

		// Alignment
		alignment += boids[j].dir;

		// Cohesion
		cohesion += to_other_normalized;
	}

	// Set steering direction
	sdirs[i] = glm::normalize(boids[i].dir +
														separation * sp.separation_weight +
														alignment * sp.alignment_weight +
														cohesion * sp.cohesion_weight);
}


__global__ void MoveBoids(Boid* boids, glm::vec2* sdirs, glm::mat4* world_matrices, MovementParams mp, float DeltaTimeInSec)
{
	int i = threadIdx.x;
	glm::vec3 dir = glm::vec3(boids[i].dir, 0.0f);
	glm::vec3 sdir = glm::vec3(sdirs[i], 0.0f);

	// turn towards the steering direction
	float angle = glm::acos(glm::dot(dir, sdir)) * glm::min(DeltaTimeInSec * mp.angular_velocity, 1.0f);
	glm::vec3 axis = glm::cross(dir, sdir);
	if (abs(axis.z) > 0.01f) {
		glm::vec2 ndir = glm::rotate(angle, axis) * glm::vec4(dir, 1.0f);
		boids[i].dir = ndir;
	}


	// move in the new direction
	boids[i].pos += boids[i].dir * mp.velocity * DeltaTimeInSec;
	boids[i].pos.x = std::fmodf(boids[i].pos.x + 3.0f, 2.0f) - 1.0f;
	boids[i].pos.y = std::fmodf(boids[i].pos.y + 3.0f, 2.0f) - 1.0f;

	world_matrices[i] =
		glm::translate(glm::vec3(boids[i].pos, 0))
		*
		glm::rotate(atan2(boids[i].dir.y, boids[i].dir.x), glm::vec3(0, 0, 1))
		*
		glm::scale(glm::vec3(0.01));
}

void CMyApp::Render()
{
	// töröljük a frampuffert (GL_COLOR_BUFFER_BIT)...
	// ... és a mélységi Z puffert (GL_DEPTH_BUFFER_BIT)
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT);

	// Set steering direction for all boids in kernel
	// TODO block calls
	SteerBoids<<<1, m_inst_num>>>(d_boids, d_sdirs, m_inst_num, m_steering_params);
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
	MoveBoids<<<1, m_inst_num>>>(d_boids, d_sdirs, world_matrices, m_movement_params, m_DeltaTimeInSec);
  checkCudaErrors( cudaGetLastError()  );

  // Unmap buffer object
  checkCudaErrors( cudaGraphicsUnmapResources(1, &world_matricesBO_CUDA, 0) );
	
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
		if (ImGui::Button("Restart"))
		{
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
