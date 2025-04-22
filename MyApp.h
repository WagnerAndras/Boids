#pragma once

#include "includes/GLUtils.hpp"
#include "includes/Camera.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

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

	void Update( const SUpdateInfo& );
	void Render();

	void Resize(int, int);

	void OtherEvent( const SDL_Event& );

protected:
	void SetupDebugCallback();
	void DrawNoInstance();
	void DrawUboInstance();
	void DrawArrayAttrInstanced();

	// Variables

	float m_ElapsedTimeInSec = 0.0f;
	
	// Camera
	Camera m_camera;

	
	// OpenGL

	// Shader variables
	static constexpr int uboSize = 100;	// How many objects we draw with one draw call
	static constexpr int uboSizeBytes = uboSize * 2 * sizeof(glm::mat4);
	GLuint m_programID = 0;		  // shaderek programja
	GLuint m_programNoInstanceID = 0;
	GLuint m_programUboInstanceID = 0;
	GLuint m_programArrayAttrInstanceID = 0;


	// Shader initialization and termination
	void InitShaders();
	void CleanShaders();
	
	// Geometry variables
	OGLObject m_BoidGPU = {};
	GLuint m_uboID = 0;
	GLuint m_matrixBufferID = 0;
	static constexpr GLuint uniformBlockBinding = 0;
	
	static constexpr int INST_NUM = 1000;		// How many heads we draw
	std::vector<glm::mat4> m_world_matricies;	// World matrices

	// Geometry initialization and termination
	void InitPositions();
	void InitAttributeMode();
	void InitGeometry();
	void CleanGeometry();

};

