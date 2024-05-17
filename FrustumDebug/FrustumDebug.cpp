// Include necessary headers
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <random>
#include "Rendering/Frustum.h"
#include "Camera.h"

// Constants and global variables
const int WIDTH = 2560;
const int HEIGHT = 1080;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

Camera camera(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f, 6.0f, 0.1f, 45.0f);
Frustum frustum;
glm::mat4 projectionMatrix;
glm::mat4 viewMatrix;

struct Cube {
    glm::vec3 position;
    float size;
    glm::vec3 color; // Add a color field
};

std::vector<Cube> cubes;
unsigned int VAO, VBO;
unsigned int shaderProgram;

void initializeOpenGL(GLFWwindow* window);
void render(GLFWwindow* window);
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void initializeCubes();
void initializeShaders();
void drawFrustum(const Frustum& frustum);

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 32); // Request a 32-bit depth buffer
    glfwWindowHint(GLFW_SAMPLES, 4); // Enable 4x multisampling
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Frustum Debug", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewInit();

    initializeOpenGL(window);

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwTerminate();
            return 0;
        }

        render(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// Function to generate a random float between 0.0 and 1.0
float randomFloat() {
    static std::default_random_engine engine(static_cast<unsigned int>(time(0)));
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(engine);
}

unsigned int frustumVAO, frustumVBO;
void initializeFrustumBuffers() {
    glGenVertexArrays(1, &frustumVAO);
    glGenBuffers(1, &frustumVBO);

    glBindVertexArray(frustumVAO);
    glBindBuffer(GL_ARRAY_BUFFER, frustumVBO);

    // Assuming we have 24 vertices (12 lines with 2 points each) for the frustum
    glBufferData(GL_ARRAY_BUFFER, 24 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void initializeOpenGL(GLFWwindow* window) {
    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);

    initializeShaders();
    initializeCubes();
    initializeFrustumBuffers();
}

void render(GLFWwindow* window) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
    viewMatrix = camera.getViewMatrix();

    frustum.update(viewMatrix, projectionMatrix);
    frustum.printFrustumDetails(viewMatrix * projectionMatrix);

    drawFrustum(frustum);

    glUseProgram(shaderProgram);
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    unsigned int colorLoc = glGetUniformLocation(shaderProgram, "cubeColor");

    int visibleObjectCount = 0;

    for (const Cube& cube : cubes) {
        if (frustum.isSphereInFrustum(cube.position, cube.size)) {
            visibleObjectCount++;

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cube.position);
            model = glm::scale(model, glm::vec3(cube.size));

            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
            glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
            glUniform3fv(colorLoc, 1, glm::value_ptr(cube.color));

            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
    }

    std::string windowTitle = "Frustum Debug - Object Count: " + std::to_string(visibleObjectCount);
    glfwSetWindowTitle(window, windowTitle.c_str());
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_D, deltaTime);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void initializeCubes() {
    int gridSize = 1;
    float spacing = 5.0f;

    for (int i = -gridSize; i <= gridSize; i++) {
        for (int j = -gridSize; j <= gridSize; j++) {
            Cube cube;
            cube.position = glm::vec3(i * spacing, 0.0f, j * spacing);
            cube.size = 1.0f;
            cube.color = glm::vec3(randomFloat(), randomFloat(), randomFloat()); // Assign random color
            cubes.push_back(cube);
        }
    }

    float vertices[] = {
        // Positions
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

unsigned int frustumShaderProgram;

void initializeShaders() {
    // Cube shader
    const char* cubeVertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    )";

    const char* cubeFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 cubeColor;
    void main() {
        FragColor = vec4(cubeColor, 1.0);
    }
    )";

    // Compile and link cube shader
    unsigned int cubeVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(cubeVertexShader, 1, &cubeVertexShaderSource, NULL);
    glCompileShader(cubeVertexShader);

    unsigned int cubeFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(cubeFragmentShader, 1, &cubeFragmentShaderSource, NULL);
    glCompileShader(cubeFragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, cubeVertexShader);
    glAttachShader(shaderProgram, cubeFragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(cubeVertexShader);
    glDeleteShader(cubeFragmentShader);

    // Frustum shader
    const char* frustumVertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
    )";

    const char* frustumFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 1.0, 0.0, 1.0); // Yellow color
    }
    )";

    // Compile and link frustum shader
    unsigned int frustumVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(frustumVertexShader, 1, &frustumVertexShaderSource, NULL);
    glCompileShader(frustumVertexShader);

    unsigned int frustumFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frustumFragmentShader, 1, &frustumFragmentShaderSource, NULL);
    glCompileShader(frustumFragmentShader);

    frustumShaderProgram = glCreateProgram();
    glAttachShader(frustumShaderProgram, frustumVertexShader);
    glAttachShader(frustumShaderProgram, frustumFragmentShader);
    glLinkProgram(frustumShaderProgram);

    glDeleteShader(frustumVertexShader);
    glDeleteShader(frustumFragmentShader);
}

void drawFrustum(const Frustum& frustum) {
    std::vector<float> frustumVertices = {
        // Near plane
        frustum.corners[0].x, frustum.corners[0].y, frustum.corners[0].z, frustum.corners[1].x, frustum.corners[1].y, frustum.corners[1].z,
        frustum.corners[1].x, frustum.corners[1].y, frustum.corners[1].z, frustum.corners[3].x, frustum.corners[3].y, frustum.corners[3].z,
        frustum.corners[3].x, frustum.corners[3].y, frustum.corners[3].z, frustum.corners[2].x, frustum.corners[2].y, frustum.corners[2].z,
        frustum.corners[2].x, frustum.corners[2].y, frustum.corners[2].z, frustum.corners[0].x, frustum.corners[0].y, frustum.corners[0].z,

        // Far plane
        frustum.corners[4].x, frustum.corners[4].y, frustum.corners[4].z, frustum.corners[5].x, frustum.corners[5].y, frustum.corners[5].z,
        frustum.corners[5].x, frustum.corners[5].y, frustum.corners[5].z, frustum.corners[7].x, frustum.corners[7].y, frustum.corners[7].z,
        frustum.corners[7].x, frustum.corners[7].y, frustum.corners[7].z, frustum.corners[6].x, frustum.corners[6].y, frustum.corners[6].z,
        frustum.corners[6].x, frustum.corners[6].y, frustum.corners[6].z, frustum.corners[4].x, frustum.corners[4].y, frustum.corners[4].z,

        // Connect near and far planes
        frustum.corners[0].x, frustum.corners[0].y, frustum.corners[0].z, frustum.corners[4].x, frustum.corners[4].y, frustum.corners[4].z,
        frustum.corners[1].x, frustum.corners[1].y, frustum.corners[1].z, frustum.corners[5].x, frustum.corners[5].y, frustum.corners[5].z,
        frustum.corners[2].x, frustum.corners[2].y, frustum.corners[2].z, frustum.corners[6].x, frustum.corners[6].y, frustum.corners[6].z,
        frustum.corners[3].x, frustum.corners[3].y, frustum.corners[3].z, frustum.corners[7].x, frustum.corners[7].y, frustum.corners[7].z
    };

    glBindBuffer(GL_ARRAY_BUFFER, frustumVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, frustumVertices.size() * sizeof(float), frustumVertices.data());

    glUseProgram(frustumShaderProgram);

    unsigned int viewLoc = glGetUniformLocation(frustumShaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(frustumShaderProgram, "projection");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glBindVertexArray(frustumVAO);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
}







