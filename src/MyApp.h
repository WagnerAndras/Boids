#pragma once

#include "includes/GLUtils.hpp"
#include "includes/Camera.h"

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
	glm::vec2 pos;
	glm::vec2 dir;
};

struct SUpdateInfo
{
	float ElapsedTimeInSec = 0.0f; // Program indulása óta eltelt idő
	float DeltaTimeInSec   = 0.0f; // Előző Update óta eltelt idő
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

	void Resize(int, int);

	void OtherEvent( const SDL_Event& );

protected:
	void SetupDebugCallback();
	
	// Boids
  Boid* d_boids;
	glm::vec2* d_sdirs;
	GLuint world_matricesBO = 0;
	struct cudaGraphicsResource* world_matricesBO_CUDA;

	// Variables
	
	// TODO weight functions
	int m_inst_num = 1000;		// how many heads we draw
	float m_half_fov_cos = 0.0f;
	float m_perception_distance = 0.1; // in uv
	float m_angular_velocity = glm::half_pi<float>(); // in radians/second
	float m_velocity = 0.1; // in uv/second
	
	float m_ElapsedTimeInSec = 0.0f;
	float m_DeltaTimeInSec = 0.0f;

	// Camera
	Camera m_camera;
	
	// OpenGL

	// Shader variables
	GLuint m_programBoidID = 0;


	// Shader initialization and termination
	void InitShaders();
	void CleanShaders();
	
	// Geometry variables
	OGLObject m_BoidGPU = {};
	
	// Geometry initialization and termination
	void InitPositions();
	void InitGeometry();
	void CleanGeometry();

};

