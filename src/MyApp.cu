#include "MyApp.h"
#include "includes/GLUtils.hpp"
#include "includes/SDL_GLDebugMessageCallback.h"
#include "includes/ProgramBuilder.h"


#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "includes/helper_cuda.h"

//#include <__clang_cuda_builtin_vars.h>
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
	m_programNoInstanceID = glCreateProgram();
	ProgramBuilder{ m_programNoInstanceID }
		.ShaderStage(GL_VERTEX_SHADER, "Boid.vert")
		.ShaderStage(GL_FRAGMENT_SHADER, "Boid.frag")
		.Link();
}

void CMyApp::CleanShaders()
{
	glDeleteProgram(m_programNoInstanceID);
}


void CMyApp::InitGeometry()
{
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
	// Initializing the Boid positions and rotations 
	std::random_device r; // seed source
	std::seed_seq seeds{r(), r(), r(), r(), r(), r(), r(), r()};
	std::mt19937 mt(seeds); // random engine with seeds
	std::uniform_real_distribution<float> randOffset(-1.0f, 1.0f);
	std::uniform_real_distribution<float> randAngle(-glm::pi<float>(), glm::pi<float>());
	
	// initialize each boid with a posiotion and an angle
	for (int i = 0; i < INST_NUM; ++i)
	{
		float angle = randAngle(mt);
		m_boids[i] = Boid {
				glm::vec2(randOffset(mt), randOffset(mt)),
				glm::vec2(glm::cos(angle), glm::sin(angle)),
			};
	}

	// Allocate vectors in device memory
  checkCudaErrors( cudaMalloc(&d_boids, INST_NUM * sizeof(Boid)));
  checkCudaErrors( cudaMalloc(&d_sdirs, INST_NUM * sizeof(glm::vec2)));
	checkCudaErrors( cudaMalloc(&d_world_matrices, INST_NUM * sizeof(glm::mat4)));
  
  checkCudaErrors( cudaMemcpy(d_boids, m_boids, INST_NUM * sizeof(Boid), cudaMemcpyHostToDevice));
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
	//InitAttributeMode();

	// Other

	glEnable(GL_CULL_FACE);	 // Enable discarding the back-facing faces.
	glCullFace(GL_BACK);     // GL_BACK: facets facing away from camera, GL_FRONT: facets facing towards the camera

	return true;
}

void CMyApp::Clean()
{
	glDeleteBuffers(1, &m_uboID);
	glDeleteBuffers(1, &m_matrixBufferID);
	
	CleanShaders();
	CleanGeometry();

	cudaFree(d_boids);
	cudaFree(d_sdirs);
}

void CMyApp::Update( const SUpdateInfo& updateInfo )
{
	m_ElapsedTimeInSec = updateInfo.ElapsedTimeInSec;
	m_DeltaTimeInSec = updateInfo.DeltaTimeInSec;
}

__global__ void SteerBoids(Boid* boids, glm::vec2* sdirs)
{
	int i = threadIdx.x;
	const float FOV_COS = std::cos((FOV * M_PI / 180.0f) / 2.0f);
	sdirs[i] = boids[i].dir;

	for (int j = 0; j < INST_NUM; j++)
	{

		// see if it's the same
		if (j == i) continue;

		glm::vec2 to_other = boids[j].pos - boids[i].pos;
		float dst = glm::length(to_other);

		// see if it's inside the perception radius
		if (dst > PERCEPTION_DISTANCE) continue;

		glm::vec2 to_other_normalized = to_other / dst;

		// see if it's in the field of view
		if (glm::dot(boids[i].dir, to_other_normalized) < FOV_COS) continue;


		// TODO weight functions
		sdirs[i] +=

		// Separation
		-to_other_normalized * (glm::sqrt(PERCEPTION_DISTANCE / dst - 1.0f) * 2.5f) +

		// Alignment
		boids[j].dir +

		// Cohesion
		to_other_normalized;
	}

	sdirs[i] = glm::normalize(sdirs[i]);
}


__global__ void MoveBoids(Boid* boids, glm::vec2* sdirs, glm::mat4* world_matrices, float DeltaTimeInSec)
{
	int i = threadIdx.x;
	glm::vec3 dir = glm::vec3(boids[i].dir, 0.0f);
	glm::vec3 sdir = glm::vec3(sdirs[i], 0.0f);

	// turn towards the steering direction
	float angle = glm::acos(glm::dot(dir, sdir)) * glm::min(DeltaTimeInSec * ANGULAR_VELOCITY, 1.0f);
	glm::vec3 axis = glm::cross(dir, sdir);
	if (abs(axis.z) > 0.01f) {
		glm::vec2 ndir = glm::rotate(angle, axis) * glm::vec4(dir, 1.0f);
		boids[i].dir = ndir;
	}


	// move in the new direction
	boids[i].pos += boids[i].dir * VELOCITY * DeltaTimeInSec;
	boids[i].pos.x = std::fmodf(boids[i].pos.x + 3.0f, 2.0f) - 1.0f;
	boids[i].pos.y = std::fmodf(boids[i].pos.y + 3.0f, 2.0f) - 1.0f;

	world_matrices[i] =
		glm::translate(glm::vec3(boids[i].pos, 0))
		*
		glm::rotate(atan2(boids[i].dir.y, boids[i].dir.x), glm::vec3(0, 0, 1))
		*
		glm::scale(glm::vec3(0.01));
}

void CMyApp::DrawNoInstance()
{
	glUseProgram(m_programNoInstanceID);

	glBindVertexArray(m_BoidGPU.vaoID);

	// Set steering direction for all boids in kernel
	SteerBoids<<<1, INST_NUM>>>(d_boids, d_sdirs);
  checkCudaErrors( cudaGetLastError() );
  // Set new positions based on the steering directions
	MoveBoids<<<1, INST_NUM>>>(d_boids, d_sdirs, d_world_matrices, m_DeltaTimeInSec);
  checkCudaErrors( cudaGetLastError() );
  checkCudaErrors( cudaMemcpy(m_world_matrices, d_world_matrices, INST_NUM * sizeof(glm::mat4), cudaMemcpyDeviceToHost));

	for (int i = 0; i < INST_NUM; ++i)
	{

	// TODO mat3:
		glUniformMatrix4fv( ul("world"), 1, GL_FALSE, glm::value_ptr(m_world_matrices[i]));
		glDrawElements(GL_TRIANGLES, m_BoidGPU.count, GL_UNSIGNED_INT, 0);
	}

	glBindVertexArray(0);
	glUseProgram(0);
	
	// exit(0);
}


void CMyApp::Render()
{
	// töröljük a frampuffert (GL_COLOR_BUFFER_BIT)...
	// ... és a mélységi Z puffert (GL_DEPTH_BUFFER_BIT)
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT);

	DrawNoInstance();
	//DrawUboInstance();
	//DrawArrayAttrInstanced();
}

void CMyApp::RenderGUI()
{
	// ImGui::ShowDemoWindow();
	if (ImGui::Begin("Instancing"))
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
