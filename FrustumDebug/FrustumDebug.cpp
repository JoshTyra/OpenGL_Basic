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

struct Cube {
    glm::vec3 position;
    float size;
    glm::vec3 color;
};

std::vector<glm::vec3> cubePositions;
std::vector<glm::vec2> cubeGridPositions;

struct Vertex {
    glm::vec3 Position;
    glm::vec2 TexCoord;
    glm::vec3 Normal;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
    glm::ivec4 BoneIDs;
    glm::vec4 Weights;
};

struct BoneInfo {
    glm::mat4 BoneOffset;
    glm::mat4 FinalTransformation;

    BoneInfo() : BoneOffset(1.0f), FinalTransformation(1.0f) {}
};

std::map<std::string, int> boneMapping; // maps a bone name to its index
std::vector<BoneInfo> boneInfo;
int numBones = 0;
std::vector<glm::mat4> boneTransforms;
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

std::vector<Cube> cubes;
unsigned int VAO, VBO;
unsigned int shaderProgram;
unsigned int characterShaderProgram;
unsigned int characterTexture;
unsigned int characterNormalMap;
unsigned int cubemapTexture;
unsigned int characterMaskTexture;
unsigned int brightPassShaderProgram;
unsigned int blurShaderProgram;
unsigned int finalCombineShaderProgram;

unsigned int instanceVBO;
unsigned int gridPositionVBO;

unsigned int characterInstanceVBO;

const glm::vec3 staticNodeRotationAxis(1.0f, 0.0f, 0.0f);
const float staticNodeRotationAngle = glm::radians(-90.0f);
float characterRotationSpeed = 0.5f; // Adjust the rotation speed as desired
std::vector<glm::vec3> characterPositions;

std::vector<Vertex> aggregatedVertices; // Global vector to hold all vertices of the model
std::vector<Mesh> loadedMeshes;
AABB loadedModelAABB;

// Water shit
unsigned int waterVAO, waterVBO, waterEBO;
unsigned int waterShaderProgram;
unsigned int waterNormalMap;
unsigned int waterCubeMapTexture;
unsigned int fresnelTexture;
std::vector<Vertex> waterVertices;
std::vector<unsigned int> waterIndices;

// Skybox shit
unsigned int skyboxVAO, skyboxVBO;
unsigned int skyboxShaderProgram;

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
void initializeCubes();
void initializeShaders();
void loadModel(const std::string& path);
void processNode(aiNode* node, const aiScene* scene);
void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation);
void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
AABB computeAABB(const std::vector<Vertex>& vertices);
AABB transformAABB(const AABB& aabb, const glm::mat4& transform);
unsigned int loadTexture(const char* path);
float randomFloat();
unsigned int compileShader(unsigned int type, const char* source);
unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader);
void renderQuad();
void createWaterMesh(float scale);
void renderWater();
void initializeSkybox();
void calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);
void calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim);
void calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);
const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName);
void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform);
void updateBoneTransforms(float timeInSeconds, const aiScene* scene);
unsigned int findPosition(float animationTime, const aiNodeAnim* nodeAnim);
unsigned int findRotation(float animationTime, const aiNodeAnim* nodeAnim);
unsigned int findScaling(float animationTime, const aiNodeAnim* nodeAnim);

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

glm::vec3 hexToRGB(const std::string& hex) {
    int r = std::stoi(hex.substr(1, 2), nullptr, 16);
    int g = std::stoi(hex.substr(3, 2), nullptr, 16);
    int b = std::stoi(hex.substr(5, 2), nullptr, 16);
    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

std::vector<std::string> colorCodes = {
    "#C13E3E", // Multiplayer Red
    "#3639C9", // Multiplayer Blue
    "#C9BA36", // Multiplayer Gold/Yellow
    "#208A20", // Multiplayer Green
    "#B53C8A", // Multiplayer Purple
    "#DF9735", // Multiplayer Orange
    "#744821", // Multiplayer Brown
    "#EB7EC5", // Multiplayer Pink
    "#D2D2D2", // Multiplayer White
    "#758550", // Campaign Color Lighter
    "#55613A"  // Campaign Color Darker
    "#000000", // Halo ce multiplayer black
    "#707E71", // Halo ce multiplayer gray
    "#01FFFF", // Halo ce multiplayer cyan
    "#6493ED", // Halo ce multiplayer cobalt
    "#C69C6C", // Halo ce multiplayer tan
};

glm::vec3 getRandomColor() {
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::uniform_int_distribution<int> distribution(0, colorCodes.size() - 1);
    return hexToRGB(colorCodes[distribution(engine)]);
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

    std::vector<std::string> faces{
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_right.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_left.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_top.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_down.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_front.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_back.tga")
    };

    cubemapTexture = loadCubemap(faces);


    std::vector<std::string> waterFaces{
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_right.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_left.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_up.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_down.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_front.tga"),
    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_back.tga")
    };

    waterCubeMapTexture = loadCubemap(waterFaces);

    std::string staticModelPath = FileSystemUtils::getAssetFilePath("models/combat_sword_idle.fbx");
    loadModel(staticModelPath);

    characterTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_D.tga").c_str());

    characterNormalMap = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_bump.tga").c_str());

    characterMaskTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_cc.tga").c_str());

    waterNormalMap = loadTexture(FileSystemUtils::getAssetFilePath("textures/water_bump4.tga").c_str());

    fresnelTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/fresnel.tga").c_str());

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
    initializeCubes();
    createWaterMesh(25.0f);
    initializeSkybox();

    glm::vec3 randomColor = getRandomColor();
    glUseProgram(characterShaderProgram);
    glUniform3fv(glGetUniformLocation(characterShaderProgram, "changeColor"), 1, glm::value_ptr(randomColor));

    // Create and bind the character instance buffer
    glGenBuffers(1, &characterInstanceVBO);

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
    glBindTexture(GL_TEXTURE_CUBE_MAP, waterCubeMapTexture);
    glUniform1i(glGetUniformLocation(skyboxShaderProgram, "skybox"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    // Perform frustum culling on cubes
    std::vector<glm::vec3> visibleCubePositions;
    std::vector<glm::vec2> visibleCubeGridPositions;

    float cubeSize = 1.0f;
    for (size_t i = 0; i < cubePositions.size(); ++i) {
        const glm::vec3& position = cubePositions[i];
        if (frustum.isSphereInFrustum(position, cubeSize)) {
            visibleCubePositions.push_back(position);
            visibleCubeGridPositions.push_back(cubeGridPositions[i]);
        }
    }

    if (!visibleCubePositions.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, visibleCubePositions.size() * sizeof(glm::vec3), &visibleCubePositions[0], GL_STATIC_DRAW);
    }

    if (!visibleCubeGridPositions.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, gridPositionVBO);
        glBufferData(GL_ARRAY_BUFFER, visibleCubeGridPositions.size() * sizeof(glm::vec2), &visibleCubeGridPositions[0], GL_STATIC_DRAW);
    }

    glUseProgram(shaderProgram);
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    unsigned int timeLoc = glGetUniformLocation(shaderProgram, "time");

    float currentTime = static_cast<float>(glfwGetTime());

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(1.0f));

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniform1f(timeLoc, currentTime);

    glBindVertexArray(VAO);

    int numVisibleCubes = visibleCubePositions.size();
    if (numVisibleCubes > 0) {
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, numVisibleCubes);
    }

    // Perform frustum culling on character instances
    std::vector<glm::vec3> visibleCharacterPositions;
    for (const auto& position : characterPositions) {
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, position);
        modelMatrix = glm::rotate(modelMatrix, staticNodeRotationAngle, staticNodeRotationAxis);
        modelMatrix = glm::rotate(modelMatrix, (float)glfwGetTime() * characterRotationSpeed, glm::vec3(0.0f, 0.0f, 1.0f));
        modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f));

        AABB transformedAABB = transformAABB(loadedModelAABB, modelMatrix);
        if (frustum.isAABBInFrustum(transformedAABB.min, transformedAABB.max)) {
            visibleCharacterPositions.push_back(position);
        }
    }

    if (!visibleCharacterPositions.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);
        glBufferData(GL_ARRAY_BUFFER, visibleCharacterPositions.size() * sizeof(glm::vec3), &visibleCharacterPositions[0], GL_STATIC_DRAW);
    }

    glUseProgram(characterShaderProgram);

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
    glm::vec3 viewPos = camera.getPosition();
    glm::vec3 ambientColor = glm::vec3(0.25f, 0.25f, 0.25f);
    glm::vec3 diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 specularColor = glm::vec3(0.5f, 0.5f, 0.5f);
    float shininess = 32.0f;
    float lightIntensity = 1.0f;

    glUniform3fv(glGetUniformLocation(characterShaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
    glUniform3fv(glGetUniformLocation(characterShaderProgram, "viewPos"), 1, glm::value_ptr(viewPos));
    glUniform3fv(glGetUniformLocation(characterShaderProgram, "ambientColor"), 1, glm::value_ptr(ambientColor));
    glUniform3fv(glGetUniformLocation(characterShaderProgram, "diffuseColor"), 1, glm::value_ptr(diffuseColor));
    glUniform3fv(glGetUniformLocation(characterShaderProgram, "specularColor"), 1, glm::value_ptr(specularColor));
    glUniform1f(glGetUniformLocation(characterShaderProgram, "shininess"), shininess);
    glUniform1f(glGetUniformLocation(characterShaderProgram, "lightIntensity"), lightIntensity);

    unsigned int modelLocChar = glGetUniformLocation(characterShaderProgram, "model");
    unsigned int viewLocChar = glGetUniformLocation(characterShaderProgram, "view");
    unsigned int projectionLocChar = glGetUniformLocation(characterShaderProgram, "projection");
    unsigned int textureLoc = glGetUniformLocation(characterShaderProgram, "texture_diffuse");
    unsigned int normalMapLoc = glGetUniformLocation(characterShaderProgram, "texture_normal");
    unsigned int cubemapLoc = glGetUniformLocation(characterShaderProgram, "cubemap");

    glUniformMatrix4fv(viewLocChar, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionLocChar, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, characterTexture);
    glUniform1i(textureLoc, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, characterNormalMap);
    glUniform1i(normalMapLoc, 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, characterMaskTexture);
    glUniform1i(glGetUniformLocation(characterShaderProgram, "texture_mask"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glUniform1i(cubemapLoc, 3);

    glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glVertexAttribDivisor(5, 1);

    if (scene && scene->mNumAnimations > 0) {
        float currentTime = static_cast<float>(glfwGetTime());
        updateBoneTransforms(currentTime, scene);

        for (unsigned int i = 0; i < boneTransforms.size(); i++) {
            std::string uniformName = "boneTransforms[" + std::to_string(i) + "]";
            glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, uniformName.c_str()), 1, GL_FALSE, glm::value_ptr(boneTransforms[i]));
        }
    }
    else {
        std::cerr << "ERROR::SCENE:: No valid animations found in the scene." << std::endl;
    }

    for (const auto& position : visibleCharacterPositions) {
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, position);
        modelMatrix = glm::rotate(modelMatrix, staticNodeRotationAngle, staticNodeRotationAxis);
        modelMatrix = glm::rotate(modelMatrix, (float)glfwGetTime() * characterRotationSpeed, glm::vec3(0.0f, 0.0f, 1.0f));
        modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f));
        glUniformMatrix4fv(modelLocChar, 1, GL_FALSE, glm::value_ptr(modelMatrix));

        for (const auto& mesh : loadedMeshes) {
            glBindVertexArray(mesh.VAO);
            glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, 1);
        }
    }

    // Render water
    renderWater();

    std::string windowTitle = "Frustum Culling - Visible Objects: " + std::to_string(visibleCharacterPositions.size());
    glfwSetWindowTitle(window, windowTitle.c_str());
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
    glUniform1f(glGetUniformLocation(brightPassShaderProgram, "brightnessThreshold"), 0.3f);
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
    glUniform1f(glGetUniformLocation(finalCombineShaderProgram, "bloomIntensity"), 0.9f);

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

void initializeCubes() {
    int gridSize = 0;
    float spacing = 3.5f;

    for (int i = -gridSize; i <= gridSize; i++) {
        for (int j = -gridSize; j <= gridSize; j++) {
            cubePositions.push_back(glm::vec3(i * spacing, 0.0f, j * spacing));
            cubeGridPositions.push_back(glm::vec2(i + gridSize, j + gridSize));
        }
    }

    int characterGridSize = 10; // Adjust this value as desired
    float characterSpacing = 5.0f; // Adjust the spacing between characters

    for (int i = -characterGridSize; i <= characterGridSize; i++) {
        for (int j = -characterGridSize; j <= characterGridSize; j++) {
            characterPositions.push_back(glm::vec3(i * characterSpacing, 0.0f, j * characterSpacing));
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

    // Create and bind the instance buffer
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, cubePositions.size() * sizeof(glm::vec3), &cubePositions[0], GL_STATIC_DRAW);

    // Set up the instance buffer attributes
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glVertexAttribDivisor(1, 1);

    // Create and bind the grid position buffer
    glGenBuffers(1, &gridPositionVBO);
    glBindBuffer(GL_ARRAY_BUFFER, gridPositionVBO);
    glBufferData(GL_ARRAY_BUFFER, cubeGridPositions.size() * sizeof(glm::vec2), &cubeGridPositions[0], GL_STATIC_DRAW);

    // Set up the grid position buffer attributes
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);
}

void initializeShaders() {
    // Cube shader
    const char* cubeVertexShaderSource = R"(
    #version 330 core
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
    #version 330 core
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

    const char* characterVertexShaderSource = R"(
    #version 430 core

    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    layout(location = 2) in vec3 aNormal;
    layout(location = 3) in vec3 aTangent;
    layout(location = 4) in vec3 aBitangent;
    layout(location = 5) in ivec4 aBoneIDs; // Bone IDs
    layout(location = 6) in vec4 aWeights;  // Bone weights
    layout(location = 7) in vec3 aInstancePosition; // Instance position

    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 TangentLightDir;
    out vec3 TangentViewPos;
    out vec3 TangentFragPos;
    out vec3 ReflectDir;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 boneTransforms[100]; // Assume a maximum of 100 bones

    uniform vec3 lightDir;
    uniform vec3 viewPos;

    void main() {
        // Calculate the bone transformation
        mat4 boneTransform = boneTransforms[aBoneIDs[0]] * aWeights[0];
        boneTransform += boneTransforms[aBoneIDs[1]] * aWeights[1];
        boneTransform += boneTransforms[aBoneIDs[2]] * aWeights[2];
        boneTransform += boneTransforms[aBoneIDs[3]] * aWeights[3];

        // Apply the bone transformation and the instance position
        vec3 transformedPos = vec3(boneTransform * vec4(aPos, 1.0)) + aInstancePosition;

        // Apply model, view, and projection transformations
        gl_Position = projection * view * model * vec4(transformedPos, 1.0);

        // Calculate the fragment position in world space
        FragPos = vec3(model * vec4(transformedPos, 1.0));

        // Pass the texture coordinates to the fragment shader
        TexCoord = aTexCoord;

        // Calculate the TBN matrix
        mat3 normalMatrix = transpose(inverse(mat3(model)));
        vec3 T = normalize(normalMatrix * aTangent);
        vec3 N = normalize(normalMatrix * aNormal);
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T);

        mat3 TBN = transpose(mat3(T, B, N));
        TangentLightDir = TBN * lightDir;
        TangentViewPos = TBN * viewPos;
        TangentFragPos = TBN * FragPos;

        // Calculate the reflection vector
        vec3 I = normalize(viewPos - FragPos);
        ReflectDir = reflect(I, N);
    }
)";

        const char* characterFragmentShaderSource = R"(
        #version 430 core

        out vec4 FragColor;

        in vec2 TexCoord;
        in vec3 TangentLightDir;
        in vec3 TangentViewPos;
        in vec3 TangentFragPos;
        in vec3 ReflectDir;

        uniform vec3 ambientColor;
        uniform vec3 diffuseColor;
        uniform vec3 specularColor;
        uniform float shininess;

        uniform sampler2D texture_diffuse;
        uniform sampler2D texture_normal;
        uniform sampler2D texture_mask;
        uniform samplerCube cubemap;
        uniform float lightIntensity;
        uniform vec3 changeColor;

        void main() {
            vec3 normal = texture(texture_normal, TexCoord).rgb;
            normal = normal * 2.0f - 1.0f;
            normal.y = -normal.y;
            normal = normalize(normal);  // Apply bump strength

            vec4 diffuseTexture = texture(texture_diffuse, TexCoord);
            vec3 diffuseTexColor = diffuseTexture.rgb;
            float alphaValue = diffuseTexture.a;
            float blendFactor = 0.1f;

            vec3 maskValue = texture(texture_mask, TexCoord).rgb;
            vec3 blendedColor = mix(diffuseTexColor, diffuseTexColor * changeColor, maskValue);

            vec3 alphaBlendedColor = mix(blendedColor, blendedColor * alphaValue, blendFactor);

            float specularMask = diffuseTexture.a;

            vec3 ambient = ambientColor * alphaBlendedColor;

            vec3 lightDir = normalize(TangentLightDir);
            float diff = max(dot(normal, lightDir), 0.0f) * lightIntensity;
            vec3 diffuse = diffuseColor * diff * alphaBlendedColor;

            vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
            vec3 halfwayDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess) * lightIntensity;
            vec3 specular = specularColor * spec * specularMask;

            float fresnelBias = 0.1f;
            float fresnelScale = 0.5f;
            float fresnelPower = 0.5f;
            vec3 I = normalize(TangentFragPos - TangentViewPos);
            float fresnel = fresnelBias + fresnelScale * pow(1.0f - dot(I, normal), fresnelPower);
            specular *= fresnel;

            vec3 color = ambient + diffuse + specular;

            vec3 reflectedColor = texture(cubemap, ReflectDir).rgb;
            reflectedColor *= specularMask;
            color = mix(color, reflectedColor, 0.2f);

            FragColor = vec4(color, 1.0f);
        }
    )";

    // Compile and link character shader
    unsigned int characterVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(characterVertexShader, 1, &characterVertexShaderSource, NULL);
    glCompileShader(characterVertexShader);

    unsigned int characterFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(characterFragmentShader, 1, &characterFragmentShaderSource, NULL);
    glCompileShader(characterFragmentShader);

    characterShaderProgram = glCreateProgram();
    glAttachShader(characterShaderProgram, characterVertexShader);
    glAttachShader(characterShaderProgram, characterFragmentShader);
    glLinkProgram(characterShaderProgram);

    glDeleteShader(characterVertexShader);
    glDeleteShader(characterFragmentShader);

    // Bright pass shader
    const char* brightPassVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

    const char* brightPassFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D hdrBuffer;
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
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

    const char* blurFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D image;
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
    #version 330 core
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

    uniform sampler2D scene;
    uniform sampler2D bloomBlur;
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

    // Water shader
    const char* waterVertexShaderSource = R"(
    #version 430 core

    layout(location = 0) in vec3 inPosition;
    layout(location = 1) in vec3 inNormal;
    layout(location = 2) in vec2 inTexCoord;

    out vec3 fragPosition;
    out vec3 fragNormal;
    out vec2 fragTexCoord;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        fragPosition = vec3(model * vec4(inPosition, 1.0));
        fragNormal = mat3(transpose(inverse(model))) * inNormal;
        fragTexCoord = inTexCoord;
        gl_Position = projection * view * vec4(fragPosition, 1.0);
    }
    )";

    const char* waterFragmentShaderSource = R"(
    #version 430 core
    in vec3 fragPosition;
    in vec3 fragNormal;
    in vec2 fragTexCoord;
    out vec4 fragColor;
    uniform samplerCube skybox;
    uniform sampler2D normalMap;
    uniform sampler2D fresnelMap;
    uniform vec3 cameraPosition;
    uniform float waveStrength;
    uniform float refractionRatio;
    uniform float time;

    void main() 
    {
        vec2 animatedTexCoord1 = fragTexCoord + vec2(time * 0.01, time * 0.02);
        vec2 animatedTexCoord2 = fragTexCoord + vec2(-time * 0.015, time * 0.025);
        vec3 normal1 = texture(normalMap, animatedTexCoord1).rgb;
        vec3 normal2 = texture(normalMap, animatedTexCoord2).rgb;
        normal1 = normalize(normal1 * 2.0 - 1.0);
        normal2 = normalize(normal2 * 2.0 - 1.0);

        vec3 perturbedNormal = normalize(fragNormal + (normal1 + normal2) * waveStrength);

        vec3 I = normalize(fragPosition - cameraPosition);
        float fresnelTerm = dot(I, perturbedNormal);
        fresnelTerm = clamp(fresnelTerm, 0.0, 1.0);

        vec3 R = reflect(I, perturbedNormal);
        vec3 Rf = refract(I, perturbedNormal, refractionRatio);

        vec4 reflectionColor = texture(skybox, R);
        vec4 refractionColor = texture(skybox, Rf);

        vec2 fresnelTexCoord = vec2(fresnelTerm, 0.5); // Calculate the texture coordinates for the fresnel map
        vec3 fresnelBlend = texture(fresnelMap, fresnelTexCoord).rgb;

        vec3 finalColor = mix(refractionColor.rgb, reflectionColor.rgb, fresnelBlend);

        fragColor = vec4(finalColor, 1.0);
    }
    )";

    unsigned int waterVertexShader = compileShader(GL_VERTEX_SHADER, waterVertexShaderSource);
    unsigned int waterFragmentShader = compileShader(GL_FRAGMENT_SHADER, waterFragmentShaderSource);
    waterShaderProgram = createShaderProgram(waterVertexShader, waterFragmentShader);
    glDeleteShader(waterVertexShader);
    glDeleteShader(waterFragmentShader);

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
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
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
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
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

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_LINEAR);
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
        vertex.BoneIDs = glm::ivec4(0);
        vertex.Weights = glm::vec4(0.0f);
        vertices.push_back(vertex);
    }

    // Process indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // Process bones
    for (unsigned int i = 0; i < mesh->mNumBones; i++) {
        aiBone* bone = mesh->mBones[i];
        int boneIndex = 0;

        if (boneMapping.find(bone->mName.C_Str()) == boneMapping.end()) {
            boneIndex = numBones;
            numBones++;
            BoneInfo bi;
            boneInfo.push_back(bi);
            boneInfo[boneIndex].BoneOffset = glm::transpose(glm::make_mat4(&bone->mOffsetMatrix.a1));
            boneMapping[bone->mName.C_Str()] = boneIndex;
        }
        else {
            boneIndex = boneMapping[bone->mName.C_Str()];
        }

        for (unsigned int j = 0; j < bone->mNumWeights; j++) {
            int vertexID = bone->mWeights[j].mVertexId;
            float weight = bone->mWeights[j].mWeight;

            for (int k = 0; k < 4; ++k) {
                if (vertices[vertexID].Weights[k] == 0.0f) {
                    vertices[vertexID].BoneIDs[k] = boneIndex;
                    vertices[vertexID].Weights[k] = weight;
                    break;
                }
            }
        }
    }

    // Aggregate vertices for AABB computation
    aggregatedVertices.insert(aggregatedVertices.end(), vertices.begin(), vertices.end());

    // Store the processed mesh
    storeMesh(vertices, indices);
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

// Function to generate a random float between 0.0 and 1.0
float randomFloat() {
    static std::default_random_engine engine(static_cast<unsigned int>(time(0)));
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(engine);
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

void createWaterMesh(float scale)
{
    // Create water mesh with two polygons (a single quad)
    const int waterGridSize = 1;
    const float waterGridSpacing = 1.0f * scale;

    waterVertices.clear();
    waterIndices.clear();

    // Create vertices
    for (int z = 0; z <= waterGridSize; ++z) {
        for (int x = 0; x <= waterGridSize; ++x) {
            Vertex vertex;
            vertex.Position = glm::vec3(x * waterGridSpacing, 0.0f, z * waterGridSpacing);
            vertex.Normal = glm::vec3(0.0f, 1.0f, 0.0f); // Ensure normals are pointing up
            vertex.TexCoord = glm::vec2(static_cast<float>(x) / waterGridSize, static_cast<float>(z) / waterGridSize);
            waterVertices.push_back(vertex);
        }
    }

    // Create indices for the two triangles
    unsigned int topLeft = 0;
    unsigned int topRight = 1;
    unsigned int bottomLeft = 2;
    unsigned int bottomRight = 3;

    waterIndices.push_back(topLeft);
    waterIndices.push_back(bottomLeft);
    waterIndices.push_back(topRight);

    waterIndices.push_back(topRight);
    waterIndices.push_back(bottomLeft);
    waterIndices.push_back(bottomRight);

    // Generate buffers and arrays
    glGenVertexArrays(1, &waterVAO);
    glGenBuffers(1, &waterVBO);
    glGenBuffers(1, &waterEBO);

    glBindVertexArray(waterVAO);

    // Bind and set vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferData(GL_ARRAY_BUFFER, waterVertices.size() * sizeof(Vertex), waterVertices.data(), GL_STATIC_DRAW);

    // Bind and set index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, waterIndices.size() * sizeof(unsigned int), waterIndices.data(), GL_STATIC_DRAW);

    // Define vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoord));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

void renderWater() {
    glUseProgram(waterShaderProgram);

    // Water model matrix
    glm::mat4 waterModelMatrix = glm::mat4(1.0f);

    // Set up uniform variables
    glUniformMatrix4fv(glGetUniformLocation(waterShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(waterModelMatrix));
    glUniformMatrix4fv(glGetUniformLocation(waterShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(glGetUniformLocation(waterShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniform3fv(glGetUniformLocation(waterShaderProgram, "cameraPosition"), 1, glm::value_ptr(camera.getPosition()));
    glUniform1f(glGetUniformLocation(waterShaderProgram, "waveStrength"), 0.25f);
    glUniform1f(glGetUniformLocation(waterShaderProgram, "refractionRatio"), 1.33f);

    float time = static_cast<float>(glfwGetTime());
    glUniform1f(glGetUniformLocation(waterShaderProgram, "time"), time);

    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, waterCubeMapTexture);
    glUniform1i(glGetUniformLocation(waterShaderProgram, "skybox"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, waterNormalMap);
    glUniform1i(glGetUniformLocation(waterShaderProgram, "normalMap"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, fresnelTexture);
    glUniform1i(glGetUniformLocation(waterShaderProgram, "fresnelMap"), 2);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Bind VAO and draw water geometry
    glBindVertexArray(waterVAO);
    glDrawElements(GL_TRIANGLES, waterIndices.size(), GL_UNSIGNED_INT, 0);
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

unsigned int findScaling(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumScalingKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mScalingKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

unsigned int findRotation(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumRotationKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mRotationKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

unsigned int findPosition(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumPositionKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mPositionKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

void updateBoneTransforms(float timeInSeconds, const aiScene* scene) {
    if (!scene || !scene->mAnimations || scene->mNumAnimations == 0) {
        std::cerr << "ERROR::ASSIMP:: No animations found in the model." << std::endl;
        return;
    }

    const aiAnimation* animation = scene->mAnimations[0];
    float ticksPerSecond = animation->mTicksPerSecond != 0 ? animation->mTicksPerSecond : 25.0f;
    float timeInTicks = timeInSeconds * ticksPerSecond;
    float animationTime = fmod(timeInTicks, animation->mDuration);

    glm::mat4 identity = glm::mat4(1.0f);
    readNodeHierarchy(animationTime, scene->mRootNode, identity);

    boneTransforms.resize(boneInfo.size());
    for (unsigned int i = 0; i < boneInfo.size(); i++) {
        boneTransforms[i] = boneInfo[i].FinalTransformation;
    }
}

void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform) {
    std::string nodeName(node->mName.data);

    const aiAnimation* animation = scene->mAnimations[0];
    glm::mat4 nodeTransformation = glm::transpose(glm::make_mat4(&node->mTransformation.a1));

    const aiNodeAnim* nodeAnim = findNodeAnim(animation, nodeName);

    if (nodeAnim) {
        aiVector3D scaling;
        calcInterpolatedScaling(scaling, animationTime, nodeAnim);
        glm::mat4 scalingM = glm::scale(glm::mat4(1.0f), glm::vec3(scaling.x, scaling.y, scaling.z));

        aiQuaternion rotationQ;
        calcInterpolatedRotation(rotationQ, animationTime, nodeAnim);
        glm::mat4 rotationM = glm::mat4_cast(glm::quat(rotationQ.w, rotationQ.x, rotationQ.y, rotationQ.z));

        aiVector3D translation;
        calcInterpolatedPosition(translation, animationTime, nodeAnim);
        glm::mat4 translationM = glm::translate(glm::mat4(1.0f), glm::vec3(translation.x, translation.y, translation.z));

        nodeTransformation = translationM * rotationM * scalingM;
    }

    glm::mat4 globalTransformation = parentTransform * nodeTransformation;

    if (boneMapping.find(nodeName) != boneMapping.end()) {
        int boneIndex = boneMapping[nodeName];
        boneInfo[boneIndex].FinalTransformation = globalTransformation * boneInfo[boneIndex].BoneOffset;
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        readNodeHierarchy(animationTime, node->mChildren[i], globalTransformation);
    }
}

const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName) {
    for (unsigned int i = 0; i < animation->mNumChannels; i++) {
        const aiNodeAnim* nodeAnim = animation->mChannels[i];
        if (std::string(nodeAnim->mNodeName.data) == nodeName) {
            return nodeAnim;
        }
    }
    return nullptr;
}

void calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumScalingKeys == 1) {
        out = nodeAnim->mScalingKeys[0].mValue;
        return;
    }

    unsigned int scalingIndex = findScaling(animationTime, nodeAnim);
    unsigned int nextScalingIndex = (scalingIndex + 1);
    assert(nextScalingIndex < nodeAnim->mNumScalingKeys);
    float deltaTime = (float)(nodeAnim->mScalingKeys[nextScalingIndex].mTime - nodeAnim->mScalingKeys[scalingIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mScalingKeys[scalingIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiVector3D& start = nodeAnim->mScalingKeys[scalingIndex].mValue;
    const aiVector3D& end = nodeAnim->mScalingKeys[nextScalingIndex].mValue;
    aiVector3D delta = end - start;
    out = start + factor * delta;
}

void calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumRotationKeys == 1) {
        out = nodeAnim->mRotationKeys[0].mValue;
        return;
    }

    unsigned int rotationIndex = findRotation(animationTime, nodeAnim);
    unsigned int nextRotationIndex = (rotationIndex + 1);
    assert(nextRotationIndex < nodeAnim->mNumRotationKeys);
    float deltaTime = (float)(nodeAnim->mRotationKeys[nextRotationIndex].mTime - nodeAnim->mRotationKeys[rotationIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mRotationKeys[rotationIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiQuaternion& startRotationQ = nodeAnim->mRotationKeys[rotationIndex].mValue;
    const aiQuaternion& endRotationQ = nodeAnim->mRotationKeys[nextRotationIndex].mValue;
    aiQuaternion::Interpolate(out, startRotationQ, endRotationQ, factor);
    out = out.Normalize();
}

void calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumPositionKeys == 1) {
        out = nodeAnim->mPositionKeys[0].mValue;
        return;
    }

    unsigned int positionIndex = findPosition(animationTime, nodeAnim);
    unsigned int nextPositionIndex = (positionIndex + 1);
    assert(nextPositionIndex < nodeAnim->mNumPositionKeys);
    float deltaTime = (float)(nodeAnim->mPositionKeys[nextPositionIndex].mTime - nodeAnim->mPositionKeys[positionIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mPositionKeys[positionIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiVector3D& start = nodeAnim->mPositionKeys[positionIndex].mValue;
    const aiVector3D& end = nodeAnim->mPositionKeys[nextPositionIndex].mValue;
    aiVector3D delta = end - start;
    out = start + factor * delta;
}
