#project('Boids', 'cpp')
project('Boids', ['cpp', 'cuda'])

gl_dep = dependency('GL')
glew_dep = dependency('GLEW')
glm_dep = dependency('glm')
imgui_dep = dependency('imgui')
sdl2_dep = dependency('SDL2')
sdl2img_dep = dependency('SDL2_image')

fs = import('fs')

fs.copyfile('src/Boid.frag', install : true, install_dir : '.')
fs.copyfile('src/Boid.vert', install : true, install_dir : '.')

fs.copyfile('src/Frag_Lighting.frag', install : true, install_dir : '.')
fs.copyfile('src/Vert_PosNormTex.vert', install : true, install_dir : '.')
fs.copyfile('src/Vert_InstancedAttr.vert', install : true, install_dir : '.')
fs.copyfile('src/Vert_InstancedUBO.vert', install : true, install_dir : '.')


executable('Boids',
           'src/main.cpp',
           'src/MyApp.cu',
           #'src/MyApp.cpp',
           'src/includes/Camera.cpp',
           'src/includes/GLUtils.cpp',
           'src/includes/ProgramBuilder.cpp',
           'src/includes/SDL_GLDebugMessageCallback.cpp',
           dependencies : [gl_dep, glew_dep, glm_dep, imgui_dep, sdl2_dep, sdl2img_dep],
           install : true)
