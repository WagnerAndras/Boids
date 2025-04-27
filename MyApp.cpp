#include "MyApp.h"
#include "includes/GLUtils.hpp"
#include "includes/SDL_GLDebugMessageCallback.h"
#include "includes/ProgramBuilder.h"

#include <cmath>
#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
//#include <imgui.h>

#include <glm/trigonometric.hpp>
#include <iostream>
#include <string>
#include <array>
#include <algorithm>
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
		//.ShaderStage(GL_FRAGMENT_SHADER, "Boid.frag")
		.Link();

/*
	m_programUboInstanceID = glCreateProgram();
	ProgramBuilder{ m_programUboInstanceID }
		.ShaderStage(GL_VERTEX_SHADER, "Vert_InstancedUBO.vert")
		.ShaderStage(GL_FRAGMENT_SHADER, "Boid.frag")
		.Link();

	// Get the index of a named uniform block
	GLuint blockIndex = glGetUniformBlockIndex(	m_programUboInstanceID, // Program ID
												"m_ubo_buffer");		// Uniform block name
	// We assign a binding point to an active uniform block
	glUniformBlockBinding(	m_programUboInstanceID,	// Program ID
							blockIndex,				// The index of the active uniform block within program whose binding to assign.
							uniformBlockBinding);	// Specifies the binding point to which to bind the uniform block with index uniformBlockIndex within program.

	m_programArrayAttrInstanceID = glCreateProgram();
	ProgramBuilder{ m_programArrayAttrInstanceID }
		.ShaderStage(GL_VERTEX_SHADER, "Vert_InstancedAttr.vert")
		.ShaderStage(GL_FRAGMENT_SHADER, "Boid.frag")
		.Link();
*/
}

void CMyApp::CleanShaders()
{
	glDeleteProgram(m_programArrayAttrInstanceID);
	//glDeleteProgram(m_programNoInstanceID);
	//glDeleteProgram(m_programUboInstanceID);
}


void CMyApp::InitGeometry()
{
	const std::initializer_list<VertexAttributeDescriptor> vertexAttribList =
	{
		{ 0, 0, 2, GL_FLOAT }, // position only, no offset
	};

	MeshObject<glm::vec2> m_BoidMeshCPU; // position only

	// Simple triangle
	m_BoidMeshCPU.vertexArray = {
		glm::vec2( -1, 1 ),
		glm::vec2( -1,-1 ),
		glm::vec2(  1, 0 ),
	};

	// TODO: unneccessary
	m_BoidMeshCPU.indexArray =
	{
		0, 1, 2
	};

	m_BoidGPU = CreateGLObjectFromMesh( m_BoidMeshCPU, vertexAttribList );
}

void CMyApp::CleanGeometry()
{
	CleanOGLObject( m_BoidGPU );
}

void CMyApp::InitPositions()
{
	// Initializing the Boid positions and rotations 
	std::random_device rd;
	std::mt19937 mt(rd());
	mt.seed(42);
	std::uniform_real_distribution<float> randOffset(-1.0f, 1.0f);
	std::uniform_real_distribution<float> randAngle(0.0f, 2.0f * glm::pi<float>());
	
	// initialize each boid with a posiotion and an angle
	m_boids.reserve(INST_NUM);
	for (int i = 0; i < INST_NUM; ++i)
	{
		m_boids.push_back(Boid {
				glm::vec2(randOffset(rd), randOffset(rd)),
				randAngle(rd),
				0
			});
	}

	m_world_matricies = std::vector<glm::mat4>(INST_NUM);
	//m_world_matricies.assign(INST_NUM, glm::mat4(0));

	/*
	// We create one buffer id
	glCreateBuffers(1, &m_uboID);
  glNamedBufferData( m_uboID, uboSizeBytes, nullptr, GL_DYNAMIC_DRAW );
	// Bind range within a buffer object to an indexed buffer target
	glBindBufferRange(	GL_UNIFORM_BUFFER,	// Target
						uniformBlockBinding,		// Index
						m_uboID,			// Buffer ID
						0,					// Offset
						uboSizeBytes);		// Size in bytes
	*/
}

/*
void CMyApp::InitAttributeMode()
{
	// TODO: vec3, mat3
	static constexpr int vec4Size = sizeof(glm::vec4);
	static constexpr int mat4Size = sizeof(glm::mat4);

	// To help setup the new vao attributes
	const auto addAttrib = [&](int binding, int attr)
	{
		// We can't put our matrix into one attribute, because only max 4 components are allowed per attribute,
		// so we need four attribute per 4x4matrix
		for ( int col_i = 0; col_i < 4; col_i++ )
		{

			glEnableVertexArrayAttrib( m_BoidGPU.vaoID, attr + col_i );
			glVertexArrayAttribBinding( m_BoidGPU.vaoID, attr + col_i, binding ); // melyik VBO-ból olvassa az adatokat
			glVertexArrayAttribFormat( m_BoidGPU.vaoID, attr + col_i, 4, GL_FLOAT, GL_FALSE, col_i * vec4Size );

		}
	};

	// We add another buffer to our VAO

	glCreateBuffers(1, &m_matrixBufferID);
  glNamedBufferData( m_matrixBufferID, INST_NUM * mat4Size, nullptr, GL_STATIC_DRAW ); // This allocates the memory on GPU
	glNamedBufferSubData(m_matrixBufferID, 0, INST_NUM * mat4Size, m_world_matricies.data());

	glVertexArrayVertexBuffer( m_BoidGPU.vaoID, 1, m_matrixBufferID, 0, mat4Size );

	// If divisor is zero, the attributes in binding indexed VBO advances once per vertex. If divisor is non-zero,
	// the attribute advances once per divisor instances of the set(s) of vertices being rendered
	glVertexArrayBindingDivisor(m_BoidGPU.vaoID, // VAO
								 1,	// Index
								 1 );// Divisor
	//glVertexArrayBindingDivisor(m_BoidGPU.vaoID, // VAO
	//							 2,	// Index
	//							 1 );// Divisor
	

	addAttrib(1,3);
	//addAttrib(2,7);
}
*/


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
}

void CMyApp::Update( const SUpdateInfo& updateInfo )
{
	m_ElapsedTimeInSec = updateInfo.ElapsedTimeInSec;
	m_DeltaTimeInSec = updateInfo.DeltaTimeInSec;

}

void CMyApp::SteerBoids()
{
	for (int i = 0; i < INST_NUM; ++i)
	{
		glm::vec2 steer_vec = glm::vec2(0);
		glm::vec2 dir = glm::vec2(glm::cos(m_boids[i].angle), glm::sin(m_boids[i].angle));

		for (int j = 0; j < INST_NUM; j++)
		{

			// see if it's the same
			if (j == i) continue;

			glm::vec2 to_other = m_boids[j].pos - m_boids[i].pos;
			float dst = glm::length(to_other);

			// see if it's inside the perception radius
			if (dst > PERCEPTION_DISTANCE) continue;

			glm::vec2 to_other_n = to_other / dst;

			// see if it's in the field of view
			if (glm::dot(dir, to_other_n) < FOV_COS) continue;


			// TODO weight functions
			steer_vec +=

			// Separation
			-to_other_n * (glm::sqrt(PERCEPTION_DISTANCE / dst) - 1) +

			// Alignment
			glm::vec2(glm::cos(m_boids[j].angle), glm::sin(m_boids[j].angle)) +

			// Cohesion
			to_other_n;
		}

		m_boids[i].steering = glm::atan(steer_vec.y, steer_vec.x);
	}
}

void CMyApp::DrawNoInstance()
{
	// Nothing unusal here

	glUseProgram(m_programNoInstanceID);

	glBindVertexArray(m_BoidGPU.vaoID);

	SteerBoids(); // set steering direction

	for (int i = 0; i < INST_NUM; ++i)
	{
		// turn towards the steering direction
		m_boids[i].angle += glm::min(m_DeltaTimeInSec * ANGULAR_VELOCITY, 1.0f) * (m_boids[i].steering - m_boids[i].angle);

		// move in the new direction
		float whole;
		m_boids[i].pos += glm::vec2(glm::cos(m_boids[i].angle), glm::sin(m_boids[i].angle)) * VELOCITY * m_DeltaTimeInSec;
		//TODO: bring back on other side

		glm::mat4 word =
			glm::translate(glm::vec3(m_boids[i].pos, 0))
			*
			glm::rotate(m_boids[i].angle, glm::vec3(0, 0, 1))
			*
			glm::scale(glm::vec3(0.01));

		m_world_matricies[i] = word;

	// TODO mat3:
		glUniformMatrix4fv( ul("world"), 1, GL_FALSE, glm::value_ptr(m_world_matricies[i]));
		glDrawElements(GL_TRIANGLES, m_BoidGPU.count, GL_UNSIGNED_INT, 0);
	}

	glBindVertexArray(0);
	glUseProgram(0);
}

/*
void CMyApp::DrawUboInstance()
{
	glUseProgram(m_programUboInstanceID);
	glBindVertexArray(m_BoidGPU.vaoID);

	glUniformMatrix4fv( ul("viewProj"), 1, GL_FALSE, glm::value_ptr(m_camera.GetViewProj()));
	glUniform1i( ul("textureImage"), 0);

	glBindBuffer(GL_UNIFORM_BUFFER, m_uboID);
	

	// Fill the UBO with data then draw, and repeat until every object is drawn 
	for (int rendered_total = 0; rendered_total < INST_NUM;)
	{
		int to_render = std::min(uboSize, INST_NUM - rendered_total);


		// TODO: mat3
		std::vector<glm::mat4> data(2 * uboSize);
		// Similar to memcpy
		std::copy(m_world_matricies.begin() + rendered_total,	m_world_matricies.begin() + rendered_total + to_render,		data.begin());

		// Set buffer data, we use subdata because we don't want to reallocate the buffer, we just want to set the data
		glBufferSubData(GL_UNIFORM_BUFFER,//m_uboID 				// Target
						0,							// Offset
						uboSizeBytes,				// Size in bytes
						(const void*)data.data());	// Pointer to the data

		
		// We draw multiple instances of Suzzanne with one call
		glDrawElementsInstanced(GL_TRIANGLES,		// Primitive type
								m_BoidGPU.count,	// Count
								GL_UNSIGNED_INT,	// Index buffer data type
								0,					// Offset in the index buffer
								to_render);			// How many instance do we draw? (Only new compared to glDrawElements)

		rendered_total += to_render;
	}
	// We can unbind them
	glBindVertexArray(0);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	glUseProgram(0);
}
*/

/*
void CMyApp::DrawArrayAttrInstanced()
{
	glBindVertexArray( m_BoidGPU.vaoID );

	// We don't have to set out World and WorldIT matrices, 
	// because they are shipped by the VAO

	glUseProgram(m_programArrayAttrInstanceID);

	// We draw multiple instances of Suzzanne with one call
	glDrawElementsInstanced(GL_TRIANGLES,	// Primitive type
		m_BoidGPU.count,	// Count
		GL_UNSIGNED_INT,	// Index buffer data type
		0,					// Offset in the index buffer
		INST_NUM);			// How many instance do we draw? (Only new compared to glDrawElements)

	glBindVertexArray(0);
}
*/

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
