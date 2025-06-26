#pragma once

#include "includes/GLUtils.hpp"
#include "includes/Camera.h"
#include "includes/CameraManipulator.h"

// GLM
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/trigonometric.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

struct Boid {
	glm::vec3 pos;
	glm::vec3 dir;
};

struct SUpdateInfo
{
	float ElapsedTimeInSec = 0.0f; // Program indulása óta eltelt idő
	float DeltaTimeInSec   = 0.0f; // Előző Update óta eltelt idő
};

struct SteeringParams {
	// vision
	float half_fov_cos = 0.0f;
	float perception_distance = 0.1; // in uv

	// steering function weights
	float separation_weight = 2.0f;
	float alignment_weight = 1.0f;
	float cohesion_weight = 1.0f;
};

struct MovementParams {
	float angular_velocity = glm::half_pi<float>(); // in radians/second
	float velocity = 0.1; // in uv/second
};

class CMyApp
{
public:
	CMyApp();
	~CMyApp();

	bool Init();
	void Clean();
	void Restart();

	void Update( const SUpdateInfo& );
	void Render();
	void RenderGUI();

	void KeyboardDown(const SDL_KeyboardEvent&);
	void KeyboardUp(const SDL_KeyboardEvent&);
	void MouseMove(const SDL_MouseMotionEvent&);
	void MouseDown(const SDL_MouseButtonEvent&);
	void MouseUp(const SDL_MouseButtonEvent&);
	void MouseWheel(const SDL_MouseWheelEvent&);
	void Resize(int, int);

	void OtherEvent( const SDL_Event& );

protected:
	void SetupDebugCallback();
	
	// Boids
  Boid* d_boids;
	glm::vec3* d_sdirs;
	GLuint world_matricesBO = 0;
	struct cudaGraphicsResource* world_matricesBO_CUDA;

	// Variables
	int m_distribution_idx = 0;

	SteeringParams m_steering_params = {};
	MovementParams m_movement_params = {};
	
	int m_inst_num = 1024;		// how many heads we draw
	
	float m_ElapsedTimeInSec = 0.0f;
	float m_DeltaTimeInSec = 0.0f;

	// Camera
	Camera m_camera;
	CameraManipulator m_cameraManipulator;
	
	// CUDA
	int m_thread_num = 1024;

	// OpenGL

	// Shader variables
	GLuint m_programBoidID = 0;
	GLuint m_programCubeID = 0;

	// Shader initialization and termination
	void InitShaders();
	void CleanShaders();
	
	// Geometry variables
	OGLObject m_BoidGPU = {};
	OGLObject m_CubeGPU = {};
	
	// Geometry initialization and termination
	void InitPositions();
	void InitGeometry();
	void CleanGeometry();

};

