#pragma once

struct LabeledImage;
struct GLFWwindow;

class GUI
{
public:
	bool	init();
	void	shut();
	void	updateAndDrawImGui();
	void	render();
	void	mainLoop();

private:
	GLFWwindow*		m_pMainWindow = nullptr;
	bool			m_bFreeFormDrawing = false;
	LabeledImage	m_freeFormDrawingImg;
};
