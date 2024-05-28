#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
#include "FileSystemUtils.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <random>
#include <chrono>
#include <map>

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

struct Vertex {
    glm::vec3 Position;
    glm::vec2 TexCoord;
    glm::vec3 Normal;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
    glm::ivec4 BoneIDs;
    glm::vec4 Weights;
};

const aiScene* scene = nullptr;
Assimp::Importer importer;

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct Mesh {
    unsigned int VAO, VBO, EBO;
    std::vector<unsigned int> indices;
};

unsigned int shaderProgram;
unsigned int brightPassShaderProgram;
unsigned int blurShaderProgram;
unsigned int finalCombineShaderProgram;

std::vector<Vertex> aggregatedVertices; // Global vector to hold all vertices of the model
std::vector<Mesh> loadedMeshes;
AABB loadedModelAABB;

// Skybox shit
unsigned int skyboxVAO, skyboxVBO;
unsigned int skyboxShaderProgram;
unsigned int skyCubeMapTexture;

unsigned int hdrFBO;
unsigned int colorBuffers[2];
unsigned int rboDepth;
unsigned int pingpongFBO[2];
unsigned int pingpongColorbuffers[2];

void initializeOpenGL(GLFWwindow* window);
void render(GLFWwindow* window);
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void initializeShaders();
void loadModel(const std::string& path);
void processNode(aiNode* node, const aiScene* scene);
void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation);
void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
AABB computeAABB(const std::vector<Vertex>& vertices);
AABB transformAABB(const AABB& aabb, const glm::mat4& transform);
unsigned int loadTexture(const char* path);
unsigned int compileShader(unsigned int type, const char* source);
unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader);
void initializeSkybox();
void renderQuad();

unsigned int loadCubemap(std::vector<std::string> faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else {
            std::cerr << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 32); // Request a 32-bit depth buffer
    glfwWindowHint(GLFW_SAMPLES, 4); // Enable 4x multisampling
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Frustum Culling", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewInit();

    std::vector<std::string> waterFaces{
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_right.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_left.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_up.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_down.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_front.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_back.tga")
     };

    skyCubeMapTexture = loadCubemap(waterFaces);

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

void initializeOpenGL(GLFWwindow* window) {
    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);

    initializeShaders();

    initializeSkybox();

    // HDR framebuffer setup
    glGenFramebuffers(1, &hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);

    // Create two floating point color buffers (use GL_RGBA16F or GL_RGBA32F)
    glGenTextures(2, colorBuffers);
    for (unsigned int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Ensure clamping to edge
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Ensure clamping to edge
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0);
    }

    // Create and attach depth buffer (renderbuffer)
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, WIDTH, HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

    unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Ping-pong framebuffers for blurring
    glGenFramebuffers(2, pingpongFBO);
    glGenTextures(2, pingpongColorbuffers);
    for (unsigned int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Ensure clamping to edge
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Ensure clamping to edge
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongColorbuffers[i], 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cerr << "Ping-pong framebuffer not complete!" << std::endl;
    }
}

void renderScene(GLFWwindow* window) {
    projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
    viewMatrix = camera.getViewMatrix();

    frustum.update(viewMatrix, projectionMatrix);

    // Render skybox
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(skyboxShaderProgram);
    glm::mat4 viewMatrixSkybox = glm::mat4(glm::mat3(camera.getViewMatrix()));
    glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrixSkybox));
    glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glBindVertexArray(skyboxVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyCubeMapTexture);
    glUniform1i(glGetUniformLocation(skyboxShaderProgram, "skybox"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
}

// Define a variable for blur spread
float blurSpread = 3.5f; // Adjust this value as needed

// Inside the render function, before the blur pass
void render(GLFWwindow* window) {

    // 1. Render scene into floating point framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderScene(window);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 2. Extract bright areas
    glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[0]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(brightPassShaderProgram);
    glUniform1f(glGetUniformLocation(brightPassShaderProgram, "brightnessThreshold"), 0.6f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorBuffers[0]);
    glUniform1i(glGetUniformLocation(brightPassShaderProgram, "hdrBuffer"), 0);
    renderQuad();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 3. Blur bright areas (Gaussian blur)
    bool horizontal = true, first_iteration = true;
    unsigned int amount = 10;
    glUseProgram(blurShaderProgram);
    float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    glUniform1fv(glGetUniformLocation(blurShaderProgram, "weight"), 5, weights);
    glUniform1f(glGetUniformLocation(blurShaderProgram, "blurSpread"), blurSpread); // Set blur spread

    for (unsigned int i = 0; i < amount; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[horizontal]);
        glUniform1i(glGetUniformLocation(blurShaderProgram, "horizontal"), horizontal);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, first_iteration ? pingpongColorbuffers[0] : pingpongColorbuffers[!horizontal]);
        renderQuad();
        horizontal = !horizontal;
        if (first_iteration)
            first_iteration = false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 4. Render final quad with scene and bloom
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(finalCombineShaderProgram);
    glUniform1f(glGetUniformLocation(finalCombineShaderProgram, "bloomIntensity"), 1.2f);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorBuffers[0]);
    glUniform1i(glGetUniformLocation(finalCombineShaderProgram, "scene"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[!horizontal]);
    glUniform1i(glGetUniformLocation(finalCombineShaderProgram, "bloomBlur"), 1);

    renderQuad();
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

void initializeShaders() {
    // Cube shader
    const char* cubeVertexShaderSource = R"(
    #version 430 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aOffset;
    layout(location = 2) in vec2 aGridPosition;

    out vec2 GridPosition;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float time;

    void main() {
        GridPosition = aGridPosition;
        float frequency = 2.0;
        float phaseShift = (aGridPosition.x + aGridPosition.y) * 0.5;
        float verticalDisplacement = sin(time * frequency + phaseShift) * 0.5;
        vec3 newPos = aPos + vec3(0.0, verticalDisplacement, 0.0);
        gl_Position = projection * view * model * vec4(newPos + aOffset, 1.0);
    }
    )";

    const char* cubeFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 GridPosition;

    uniform float time;

    void main() {
        float frequency = 2.0;
        float phaseShift = (GridPosition.x + GridPosition.y) * 0.5;
        float t = time * frequency + phaseShift;
    
        vec3 color;
        color.r = sin(t) * 0.5 + 0.5;
        color.g = sin(t + 2.0 * 3.14159 / 3.0) * 0.5 + 0.5;
        color.b = sin(t + 4.0 * 3.14159 / 3.0) * 0.5 + 0.5;

        FragColor = vec4(color, 1.0);
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

    // Bright pass shader
    const char* brightPassVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

    const char* brightPassFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    layout(binding = 0) uniform sampler2D hdrBuffer;
    uniform float brightnessThreshold;

    void main() {
        vec3 hdrColor = texture(hdrBuffer, TexCoords).rgb;
        float brightness = dot(hdrColor, vec3(0.2126, 0.7152, 0.0722)); // Luminance calculation
        if (brightness > brightnessThreshold)
            FragColor = vec4(hdrColor, 1.0);
        else
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    )";

    unsigned int brightPassVertexShader = compileShader(GL_VERTEX_SHADER, brightPassVertexShaderSource);
    unsigned int brightPassFragmentShader = compileShader(GL_FRAGMENT_SHADER, brightPassFragmentShaderSource);
    brightPassShaderProgram = createShaderProgram(brightPassVertexShader, brightPassFragmentShader);
    glDeleteShader(brightPassVertexShader);
    glDeleteShader(brightPassFragmentShader);

    // Blur shader
    const char* blurVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

    const char* blurFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    layout(binding = 0) uniform sampler2D image;
    uniform bool horizontal;
    uniform float weight[5];
    uniform float blurSpread; // New uniform to control blur spread

    void main() {
        vec2 tex_offset = blurSpread / textureSize(image, 0); // Adjust based on blurSpread
        vec3 result = texture(image, TexCoords).rgb * weight[0]; // current fragment's contribution
        for (int i = 1; i < 5; ++i) {
            if (horizontal) {
                result += texture(image, TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
                result += texture(image, TexCoords - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            } else {
                result += texture(image, TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
                result += texture(image, TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            }
        }
        FragColor = vec4(result, 1.0);
    }
    )";

    unsigned int blurVertexShader = compileShader(GL_VERTEX_SHADER, blurVertexShaderSource);
    unsigned int blurFragmentShader = compileShader(GL_FRAGMENT_SHADER, blurFragmentShaderSource);
    blurShaderProgram = createShaderProgram(blurVertexShader, blurFragmentShader);
    glDeleteShader(blurVertexShader);
    glDeleteShader(blurFragmentShader);

    // Final combine shader
    const char* finalCombineVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

    const char* finalCombineFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 TexCoords;

    layout(binding = 0) uniform sampler2D scene;
    layout(binding = 1) uniform sampler2D bloomBlur;
    uniform float bloomIntensity; // Add bloom intensity uniform

    void main() {
        vec3 hdrColor = texture(scene, TexCoords).rgb;
        vec3 bloomColor = texture(bloomBlur, TexCoords).rgb;
        FragColor = vec4(hdrColor + bloomIntensity * bloomColor, 1.0); // Additive blending
    }
    )";

    unsigned int finalCombineVertexShader = compileShader(GL_VERTEX_SHADER, finalCombineVertexShaderSource);
    unsigned int finalCombineFragmentShader = compileShader(GL_FRAGMENT_SHADER, finalCombineFragmentShaderSource);
    finalCombineShaderProgram = createShaderProgram(finalCombineVertexShader, finalCombineFragmentShader);
    glDeleteShader(finalCombineVertexShader);
    glDeleteShader(finalCombineFragmentShader);

    // Skybox shader
    const char* skyboxVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    out vec3 TexCoords;

    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        TexCoords = aPos;
        mat4 rotView = mat4(mat3(view)); // remove translation part of the view matrix
        gl_Position = projection * rotView * vec4(aPos, 1.0);
    }
    )";

    const char* skyboxFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec3 TexCoords;

    uniform samplerCube skybox;

    void main()
    {
        FragColor = texture(skybox, TexCoords);
    }
    )";

    unsigned int skyboxVertexShader = compileShader(GL_VERTEX_SHADER, skyboxVertexShaderSource);
    unsigned int skyboxFragmentShader = compileShader(GL_FRAGMENT_SHADER, skyboxFragmentShaderSource);

    skyboxShaderProgram = createShaderProgram(skyboxVertexShader, skyboxFragmentShader);

    glDeleteShader(skyboxVertexShader);
    glDeleteShader(skyboxFragmentShader);
}

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::string shaderType = (type == GL_VERTEX_SHADER) ? "VERTEX" : (type == GL_FRAGMENT_SHADER) ? "FRAGMENT" : "UNKNOWN";
        std::cerr << "ERROR::SHADER::" << shaderType << "::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}

unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader) {
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Validate the program
    glValidateProgram(program);
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::VALIDATION_FAILED\n" << infoLog << std::endl;
    }

    return program;
}

unsigned int loadTexture(const char* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else {
        std::cerr << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

void loadModel(const std::string& path) {
    scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return;
    }

    std::cout << "Model loaded successfully." << std::endl;

    if (scene->mNumAnimations > 0) {
        std::cout << "Number of animations: " << scene->mNumAnimations << std::endl;
        for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
            std::cout << "Animation " << i << " duration: " << scene->mAnimations[i]->mDuration << std::endl;
        }
    }
    else {
        std::cout << "No animations found in the model." << std::endl;
    }

    aggregatedVertices.clear();
    processNode(scene->mRootNode, scene);
    loadedModelAABB = computeAABB(aggregatedVertices);
}

void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices) {
    Mesh mesh;

    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
    glEnableVertexAttribArray(0);

    // Vertex Texture Coords
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoord));
    glEnableVertexAttribArray(1);

    // Vertex Normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(2);

    // Vertex Tangents
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
    glEnableVertexAttribArray(3);

    // Vertex Bitangents
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
    glEnableVertexAttribArray(4);

    // Bone IDs
    glVertexAttribIPointer(5, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, BoneIDs));
    glEnableVertexAttribArray(5);

    // Bone Weights
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Weights));
    glEnableVertexAttribArray(6);

    glBindVertexArray(0);

    mesh.indices = indices;
    loadedMeshes.push_back(mesh);
}

void initializeSkybox()
{
    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };

    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
}

void renderQuad() {
    static unsigned int quadVAO = 0;
    static unsigned int quadVBO;
    if (quadVAO == 0) {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,

            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,  1.0f, 1.0f
        };
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

// Function to compute AABB
AABB computeAABB(const std::vector<Vertex>& vertices) {
    glm::vec3 min = vertices[0].Position;
    glm::vec3 max = vertices[0].Position;

    for (const auto& vertex : vertices) {
        min = glm::min(min, vertex.Position);
        max = glm::max(max, vertex.Position);
    }

    return { min, max };
}

// Function to transform AABB
AABB transformAABB(const AABB& aabb, const glm::mat4& transform) {
    glm::vec3 corners[8] = {
        aabb.min,
        glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z),
        glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z),
        glm::vec3(aabb.min.x, aabb.max.y, aabb.max.z),
        glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z),
        glm::vec3(aabb.max.x, aabb.min.y, aabb.max.z),
        glm::vec3(aabb.max.x, aabb.max.y, aabb.min.z),
        aabb.max
    };

    glm::vec3 newMin = transform * glm::vec4(corners[0], 1.0f);
    glm::vec3 newMax = newMin;

    for (int i = 1; i < 8; ++i) {
        glm::vec3 transformedCorner = transform * glm::vec4(corners[i], 1.0f);
        newMin = glm::min(newMin, transformedCorner);
        newMax = glm::max(newMax, transformedCorner);
    }

    return { newMin, newMax };
}

void processNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene, node->mTransformation);
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Process vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        aiVector3D transformedPosition = nodeTransformation * mesh->mVertices[i];
        vertex.Position = glm::vec3(transformedPosition.x, transformedPosition.y, transformedPosition.z);
        vertex.Normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        vertex.TexCoord = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
        vertex.Tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
        vertex.Bitangent = glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);
        vertices.push_back(vertex);
    }

    // Process indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // Aggregate vertices for AABB computation
    aggregatedVertices.insert(aggregatedVertices.end(), vertices.begin(), vertices.end());

    // Store the processed mesh
    storeMesh(vertices, indices);
}
