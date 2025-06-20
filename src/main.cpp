// GLEW
#include <GL/glew.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

// ImGui
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

// Standard
#include <iostream>
#include <sstream>

#include "MyApp.h"

int main(int argc, char* args[])
{
	// 1: Initialize SDL

	// Setup error logging function
	SDL_LogSetPriority(SDL_LOG_CATEGORY_ERROR, SDL_LOG_PRIORITY_ERROR);
	// We initialize the graphics subsystem, if there is an error we log it, and terminate
	if (SDL_Init(SDL_INIT_VIDEO) == -1)
	{
		// Log the error and terminate the program
		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[SDL initialization] Error during the SDL initialization: %s", SDL_GetError());
		return 1;
	}

	// After SDL_Init is called, upon exit the subsytems should be turned off
	// This way SDL_Quit is called even if std::exit is called
	std::atexit(SDL_Quit);

	// 2: Set our OpenGL requirements, create the window, start OpenGL

	// We can set which OpenGL context we want to create,
	// If it's not done it defaults to the highest available version

	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef _DEBUG // If it's a debug build
	// OpenGL context will be in debug mode, this way debugcallback will work
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif 

	// These will affect our default framebuffer
	// Set how many bits per pixel we use to store our red, green, blue, opacity information
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,	32);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE,	8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,	8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,	8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,	8);
	// Double buffering
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	// Depth buffer size in bits
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	// Antialiasing - if needed
	//SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS,  1);
	//SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES,  2);

	// Create our window
	SDL_Window* win = nullptr;
	win = SDL_CreateWindow("Hello SDL&OpenGL!",	// Window header
		100,									// Initial X coordinate of the top-left corner
		100,									// Initial Y coordinate of the top-left corner
		800,									// Window width
		600,									// Window height
		SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);	// Additional flags


	// If the window creation failed, log the error and terminate
	if (win == nullptr)
	{
		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[Window creation] Error during the SDL initialization: %s", SDL_GetError());
		return 1;
	}

	// 3: Create the OpenGL context - we will draw with it

	SDL_GLContext	context = SDL_GL_CreateContext(win);
	if (context == nullptr)
	{
		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[OGL context creation] Error during the creation of the OGL context: %s", SDL_GetError());
		return 1;
	}

	// Use vsync
	SDL_GL_SetSwapInterval(0); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync

	// Start GLEW
	GLenum error = glewInit();
	if (error != GLEW_OK)
	{
		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[GLEW] Error during the initialization of glew.");
		return 1;
	}

	// Query the OpenGL version
	int glVersion[2] = { -1, -1 };
	glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
	glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);

	SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Running OpenGL %d.%d", glVersion[0], glVersion[1]);

	if (glVersion[0] == -1 && glVersion[1] == -1)
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(win);

		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[OGL context creation] Error during the inialization of the OGL context! Maybe one of the SDL_GL_SetAttribute(...) calls is erroneous.");

		return 1;
	}

	std::stringstream window_title;
	window_title << "OpenGL " << glVersion[0] << "." << glVersion[1];
	SDL_SetWindowTitle(win, window_title.str().c_str());

	// Init ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui::StyleColorsDark();

	ImGui_ImplSDL2_InitForOpenGL(win, context);
	ImGui_ImplOpenGL3_Init();

	// 4: Start main event loop
	{
		bool quit = false;	// Should we terminate?
		SDL_Event ev;		// Event to be processed

		// App instance
		CMyApp app;
		if (!app.Init())
		{
			SDL_GL_DeleteContext(context);
			SDL_DestroyWindow(win);
			SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[app.Init] Error during the initialization of the application!");
			return 1;
		}

		while (!quit)
		{
			// As long as there are events to be processed, we process them all:
			while (SDL_PollEvent(&ev))
			{
				ImGui_ImplSDL2_ProcessEvent(&ev);
				bool is_mouse_captured = ImGui::GetIO().WantCaptureMouse;		// Does imgui want the mouse?
				bool is_keyboard_captured = ImGui::GetIO().WantCaptureKeyboard;	// Does imgui want the keyboard?

				switch (ev.type)
				{
				case SDL_QUIT:
					quit = true;
					break;
				case SDL_KEYDOWN:
					if (ev.key.keysym.sym == SDLK_ESCAPE)
						quit = true;

					// ALT + ENTER switches to full screen and back
					if ((ev.key.keysym.sym == SDLK_RETURN)  // Enter is pushed down ...
						&& (ev.key.keysym.mod & KMOD_ALT)   // with ALT ...
						&& !(ev.key.keysym.mod & (KMOD_SHIFT | KMOD_CTRL | KMOD_GUI)) // but without using any other modifier key
						&& ev.key.repeat == 0) // Prevents flickering when held down
					{
						Uint32 FullScreenSwitchFlag = (SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP) ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP;
						SDL_SetWindowFullscreen(win, FullScreenSwitchFlag);
					}
					/*
					if (!is_keyboard_captured)
						app.KeyboardDown(ev.key);
					break;
				case SDL_KEYUP:
					if (!is_keyboard_captured)
						app.KeyboardUp(ev.key);
					break;
				case SDL_MOUSEBUTTONDOWN:
					if (!is_mouse_captured)
						app.MouseDown(ev.button);
					break;
				case SDL_MOUSEBUTTONUP:
					if (!is_mouse_captured)
						app.MouseUp(ev.button);
					break;
				case SDL_MOUSEWHEEL:
					if (!is_mouse_captured)
						app.MouseWheel(ev.wheel);
					break;
				case SDL_MOUSEMOTION:
					if (!is_mouse_captured)
						app.MouseMove(ev.motion);
					break;
					*/
				case SDL_WINDOWEVENT:
					// On some platforms (e.g. Windows) SIZE_CHANGED is not called when first shown.
					// We think this is a bug in the SDL library.
					// Therefore, we treat this case separately,
					// since MyApp may contain settings dependent on window size, 
					// e.g.aspect ratio of the camera when calling perspective().
					if ((ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) || (ev.window.event == SDL_WINDOWEVENT_SHOWN))
					{
						int w, h;
						SDL_GetWindowSize(win, &w, &h);
						app.Resize(w, h);
					}
					break;
				default:
					app.OtherEvent(ev);
				}
			}

			// Let's calculate the amount of time required for the update!
			static Uint32 LastTick = SDL_GetTicks();// We statically store what the previous "tick" was
			Uint32 CurrentTick = SDL_GetTicks();	// Current "tick"
			SUpdateInfo updateInfo					// Convert it to seconds
			{
				static_cast<float>(CurrentTick) / 1000.0f,
				static_cast<float>(CurrentTick - LastTick) / 1000.0f
			};
			LastTick = CurrentTick; // Save the current tick as the last

			app.Update(updateInfo);
			app.Render();

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplSDL2_NewFrame(); // After this ImGui commands can be called until ImGui::Render()

			ImGui::NewFrame();
			app.RenderGUI();
			ImGui::Render();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			SDL_GL_SwapWindow(win);
		}
		app.Clean(); // Let our object clean up after itself
	} // This way the destructor of the app can run while our context is still alive => the destructors of the classes that include GPU resources also run here

	// 5: Shutdown

	// Shutdown ImGui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	// Delete context
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(win);

	return 0;
}
