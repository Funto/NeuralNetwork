#include "GUI.h"
#include "NeuralNetwork.h"
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_glfw.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define WIN_SIZEX	1280
#define WIN_SIZEY	720
#define WIN_TITLE	"Neural network"

static void _glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

bool GUI::init()
{
	// --- Initialize GLFW ---
	printf("Initializing GUI...\n");
	glfwSetErrorCallback(_glfwErrorCallback);
	if(glfwInit() != GLFW_TRUE)
	{
		fprintf(stderr, "Failed to init GLFW\n");
		return EXIT_FAILURE;
	}

	// --- Create GLFW window ---
	// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
	const char* glslVersion = "#version 100";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glslVersion = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glslVersion = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// 3.0+ only
#endif
	
	// Create window with graphics context
	m_pMainWindow = glfwCreateWindow(WIN_SIZEX, WIN_SIZEY, WIN_TITLE, NULL, NULL);
	if (!m_pMainWindow)
	{
		glfwTerminate();
		fprintf(stderr, "Failed to create GLFW window\n");
		return false;
	}
	glfwMakeContextCurrent(m_pMainWindow);

	// --- Init Dear ImGui ---
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	// io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;	// Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;	// Enable Docking
#if !defined(__EMSCRIPTEN__)							// No multi-windows in Web version
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
#endif

	ImGui_ImplGlfw_InitForOpenGL(m_pMainWindow, true);
	ImGui_ImplOpenGL3_Init(glslVersion);
	
	return true;
}

void GUI::shut()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();

	ImGui::DestroyContext();
	glfwTerminate();
}

void GUI::updateAndDrawImGui()
{
	//ImGui::ShowDemoWindow();

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if(ImGui::MenuItem("Exit"))
				gData.bExitApp = true;
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoBringToFrontOnFocus;

	static bool s_bDebugFloatingMainWindow = false;
	if(!s_bDebugFloatingMainWindow)
	{
		const float menuSizeY = ImGui::GetItemRectSize().y;
		ImGui::SetNextWindowPos(ImGui::GetMainViewport()->Pos + ImVec2(0, menuSizeY));
		ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size - ImVec2(0, menuSizeY));
		windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
	}

	if(ImGui::Begin("Main window", nullptr, windowFlags))
	{
		static int s_idxTestImage = 0;
		const LabeledImage* pImg = &gData.testImages[s_idxTestImage];

		static bool s_bFirstTime = true;
		if(s_bFirstTime)
		{
			s_bFirstTime = false;
			gData.pNN->feedForward(*pImg, false);
		}

		if(ImGui::Button(formatTempStr("Test image %d###btnTestImage", s_idxTestImage)))
		{
			s_idxTestImage++;
			if(s_idxTestImage >= (int)gData.testImages.size())
				s_idxTestImage = 0;
			pImg = &gData.testImages[s_idxTestImage];
			gData.pNN->feedForward(*pImg, false);
		}

		ImGui::BeginDisabled();
		if(ImGui::Button("Begin training"))
		{
			// TODO
		}
		ImGui::SetItemTooltip("TODO: not implemented");
		ImGui::EndDisabled();

		ImGui::Checkbox("Free-form drawing", &m_bFreeFormDrawing);
		
		ImDrawList* pDrawList = ImGui::GetWindowDrawList();

		int winPosX=0, winPosY=0;
		glfwGetWindowPos(m_pMainWindow, &winPosX, &winPosY);
		const ImVec2 winPos = ImVec2((float)winPosX, (float)winPosY);
		
		int winSizeX=0, winSizeY=0;
		glfwGetWindowSize(m_pMainWindow, &winSizeX, &winSizeY);
		const ImVec2 winSize = ImVec2((float)winSizeX, (float)winSizeY);

		const int nbLayers = _countof(gData.pNN->layers);
		const float leftMargin = 0.2f * winSize.x;
		const float rightMargin = 0.1f * winSize.x;
		const float topMargin = 0.05f * winSize.y;
		const float bottomMargin = 0.05f * winSize.y;
		const float betweenLayers = (winSize.x-leftMargin-rightMargin) / (float)(nbLayers-1);

		// Draw & update input image
		{
			const ImVec2 increment = ImVec2(leftMargin / IMG_SX, leftMargin / IMG_SY);
			const float imageBorder = 0.01f * winSize.x;
			const ImVec2 posStart = ImVec2(imageBorder, topMargin + winSize.y * 0.1f);
			const float imageSize = leftMargin - 2.f*imageBorder;
			const ImVec2 posEnd = posStart + ImVec2(imageSize,imageSize);
			const ImVec2 pixelSize = ImVec2(imageSize / IMG_SX, imageSize / IMG_SY);

			// Free-form drawing implementation
			if(m_bFreeFormDrawing)
			{
				const ImVec2 mousePos = ImGui::GetMousePos();
				const ImVec2 relMousePos = mousePos - winPos;
				memset(&m_freeFormDrawingImg.data[0], 0, IMG_SX*IMG_SY);

				ImVec2 pos = posStart;
				for(int y=0 ; y < IMG_SY ; y++, pos.y += pixelSize.y)
				{
					const float dy = pos.y - relMousePos.y;
					pos.x = posStart.x;
					for(int x=0 ; x < IMG_SX ; x++, pos.x += pixelSize.x)
					{
						const float dx = pos.x - relMousePos.x;

						static float s_scale = 10000.f;
						static float s_minVal = 0.01f;
						const unsigned char colValue = (unsigned char)std::min(255.f, std::max(0.f, s_scale / std::max(s_minVal, dx*dx + dy*dy)));
						m_freeFormDrawingImg.data[x + y*IMG_SX] = colValue;
					}
				}
				//printf("%f %f\n", mousePos.x, mousePos.y);
				m_freeFormDrawingImg.updateFloatDataFromData();
				gData.pNN->feedForward(m_freeFormDrawingImg, false);
			}

			// Draw input image
			{
				const LabeledImage& img = m_bFreeFormDrawing ? m_freeFormDrawingImg : gData.testImages[s_idxTestImage];
				ImVec2 pos = posStart;
				for(int y=0 ; y < IMG_SY ; y++, pos.y += pixelSize.y)
				{
					pos.x = posStart.x;
					for(int x=0 ; x < IMG_SX ; x++, pos.x += pixelSize.x)
					{
						const unsigned char colValue = img.data[x + y*IMG_SX];
						pDrawList->AddRectFilled(winPos + pos, winPos + pos + pixelSize, IM_COL32(colValue, colValue, colValue, 0xff));
					}
				}
			}
		}

		const float			charSize		= ImGui::CalcTextSize("A").x;
		static const char*	s_labels[]		= {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

		// Draw connections
		{
			ImVec2 curEndNeuronPos = ImVec2(leftMargin + betweenLayers, topMargin);
			for(int idxEndLayer=1 ; idxEndLayer < nbLayers ; idxEndLayer++, curEndNeuronPos.x += betweenLayers, curEndNeuronPos.y = topMargin)
			{
				const Layer& startLayer = gData.pNN->layers[idxEndLayer-1];
				const int nbStartNeurons = startLayer.nbOutputs;
				const float verticalSpaceBetweenStartNeurons = (winSize.y - topMargin - bottomMargin) / (float)(nbStartNeurons-1);

				const Layer& endLayer = gData.pNN->layers[idxEndLayer];
				const int nbEndNeurons = endLayer.nbOutputs;
				const float verticalSpaceBetweenEndNeurons = (winSize.y - topMargin - bottomMargin) / (float)(nbEndNeurons-1);
			
				for(int idxEndNeuron=0 ; idxEndNeuron < nbEndNeurons ; idxEndNeuron++, curEndNeuronPos.y += verticalSpaceBetweenEndNeurons)
				{
					//float f = layer.neuronValues[idxNeuron];
					//pDrawList->AddCircleFilled(winPos + curNeuronPos, ((float)fabs(f) + 0.2f)*5.f, f >= 0.f ? neuronColorPos : neuronColorNeg);

					ImVec2 curStartNeuronPos = ImVec2(curEndNeuronPos.x - betweenLayers, topMargin);
					for(int idxStartNeuron=0 ; idxStartNeuron < nbStartNeurons ; idxStartNeuron++, curStartNeuronPos.y += verticalSpaceBetweenStartNeurons)
					{
						static float s_lineScale = 1.f;
						const float weight = endLayer.weightsAndBias[idxStartNeuron];
						const float inputNeuronValue = startLayer.neuronValues[idxStartNeuron];
						static const ImColor s_weightColorPos = ImColor(0.5f,1.0f,0.5f,1.0f);;
						static const ImColor s_weightColorNeg = ImColor(1.0f,0.5f,0.5f,1.0f);
						pDrawList->AddLine(winPos + curStartNeuronPos, winPos + curEndNeuronPos,
							weight > 0.f ? s_weightColorPos : s_weightColorNeg,
							weight * inputNeuronValue * s_lineScale);
							//weight * s_lineScale);
					}
				}
			}
		}

		// Draw neurons
		{
			const ImColor		neuronColorPos	= ImColor(0.5f,1.0f,0.5f,1.0f);
			const ImColor		neuronColorNeg	= ImColor(1.0f,0.5f,0.5f,1.0f);
		
			ImVec2 curNeuronPos = ImVec2(leftMargin, topMargin);
			for(int idxLayer=0 ; idxLayer < nbLayers ; idxLayer++, curNeuronPos.x += betweenLayers, curNeuronPos.y = topMargin)
			{
				const Layer& layer = gData.pNN->layers[idxLayer];
				const int nbNeurons = layer.nbOutputs;
				const float verticalSpaceBetweenNeurons = (winSize.y - topMargin - bottomMargin) / (float)(nbNeurons-1);
			
				for(int idxNeuron=0 ; idxNeuron < nbNeurons ; idxNeuron++, curNeuronPos.y += verticalSpaceBetweenNeurons)
				{
					float f = layer.neuronValues[idxNeuron];
					pDrawList->AddCircleFilled(winPos + curNeuronPos, ((float)fabs(f) + 0.2f)*5.f, f >= 0.f ? neuronColorPos : neuronColorNeg);

					if(idxLayer == nbLayers-1)
					{
						pDrawList->AddText(winPos + curNeuronPos + ImVec2(charSize*3.f, -charSize), IM_COL32_WHITE, s_labels[idxNeuron]);
					}
				}
			}
		}
	}
	ImGui::End();
}

void GUI::render()
{
	const ImVec4 clearColor = ImVec4(115.f / 255.f, 131.f / 255.f, 140.f / 255.f, 1.f);

	// Rendering
	int display_w, display_h;
	glfwGetFramebufferSize(m_pMainWindow, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w,
				 clearColor.z * clearColor.w, clearColor.w);
	glClear(GL_COLOR_BUFFER_BIT);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Update and Render additional Platform Windows
	// (Platform functions may change the current OpenGL context, so we save/restore it to make it
	// easier to paste this code elsewhere.
	//  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		GLFWwindow* pBackupCurrentContext = glfwGetCurrentContext();

		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();

		glfwMakeContextCurrent(pBackupCurrentContext);
	}
}

void GUI::mainLoop()
{
	static std::function<void()> loop;

	loop = [&] {
		if(!gData.bDebugAlwaysRedrawContent)
			glfwWaitEvents();
		else
			glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		updateAndDrawImGui();

		ImGui::Render();
		render();

		glfwSwapBuffers(m_pMainWindow);

		gData.curFrame++;
	};

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop([](){loop();}, 0, true);
	glfwSwapInterval(1);	// Enable vsync
#else
	glfwSwapInterval(1);	// Enable vsync
	while (!glfwWindowShouldClose(m_pMainWindow) && !gData.bExitApp)
	{
		loop();
	}
#endif
}
