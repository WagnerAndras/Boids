#version 430

// VBO-ból érkező változók
layout( location = 0 ) in vec2 vs_in_position;

// a pipeline-ban tovább adandó értékek
out vec3 vs_out_color;

// shader külső paraméterei

// transzformációs mátrixok
uniform mat4 world;
//uniform mat4 viewProj; // Egyben adjuk át, előre össze szorozva a view és projection mátrixokat.

void main()
{
	//gl_Position  = viewProj * world * vec4( vs_in_position, 1.0 );
	gl_Position  = world * vec4( vs_in_position, 0.0, 1.0 );
	vs_out_color = vec3(0.7);
}
