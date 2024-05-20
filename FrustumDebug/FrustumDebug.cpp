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
    glm::vec2 gridPosition; // Add grid position
};

struct Vertex {
    glm::vec3 Position;
    glm::vec2 TexCoord; // Add texture coordinates
};

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

const glm::vec3 staticNodeRotationAxis(1.0f, 0.0f, 0.0f);
const float staticNodeRotationAngle = glm::radians(-90.0f);

std::vector<Vertex> aggregatedVertices; // Global vector to hold all vertices of the model
std::vector<Mesh> loadedMeshes;
AABB loadedModelAABB;

void initializeOpenGL(GLFWwindow* window);
void render(GLFWwindow* window);
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void initializeCubes();
void initializeShaders();
void loadModel(const std::string& path);
void processNode(aiNode* node, const aiScene* scene);
void processMesh(aiMesh* mesh, const aiScene* scene);
void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
AABB computeAABB(const std::vector<Vertex>& vertices);
AABB transformAABB(const AABB& aabb, const glm::mat4& transform);
unsigned int loadTexture(const char* path);
float randomFloat();

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

    initializeOpenGL(window);

    std::string staticModelPath = FileSystemUtils::getAssetFilePath("models/masterchief_no_lods.fbx");
    loadModel(staticModelPath);

    characterTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_D.tga").c_str());

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
}

void render(GLFWwindow* window) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
    viewMatrix = camera.getViewMatrix();

    frustum.update(viewMatrix, projectionMatrix);

    // Render static cubes
    glUseProgram(shaderProgram);
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    unsigned int timeLoc = glGetUniformLocation(shaderProgram, "time");
    unsigned int gridPositionLoc = glGetUniformLocation(shaderProgram, "gridPosition");

    int visibleObjectCount = 0;
    float currentTime = static_cast<float>(glfwGetTime());

    for (const Cube& cube : cubes) {
        if (frustum.isSphereInFrustum(cube.position, cube.size)) {
            visibleObjectCount++;

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cube.position);
            model = glm::scale(model, glm::vec3(cube.size));

            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
            glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
            glUniform1f(timeLoc, currentTime);
            glUniform2fv(gridPositionLoc, 1, glm::value_ptr(cube.gridPosition));

            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
    }

    // Render character model
    glUseProgram(characterShaderProgram);
    unsigned int modelLocChar = glGetUniformLocation(characterShaderProgram, "model");
    unsigned int viewLocChar = glGetUniformLocation(characterShaderProgram, "view");
    unsigned int projectionLocChar = glGetUniformLocation(characterShaderProgram, "projection");
    unsigned int textureLoc = glGetUniformLocation(characterShaderProgram, "texture_diffuse");

    glUniformMatrix4fv(viewLocChar, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionLocChar, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, characterTexture); // characterTexture is the texture ID loaded earlier
    glUniform1i(textureLoc, 0);

    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::rotate(modelMatrix, staticNodeRotationAngle, staticNodeRotationAxis);
    modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f));

    glUniformMatrix4fv(modelLocChar, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    for (const auto& mesh : loadedMeshes) {
        glBindVertexArray(mesh.VAO);
        glDrawElements(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0);
    }

    std::string windowTitle = "Frustum Culling - Object Count: " + std::to_string(visibleObjectCount);
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
    int gridSize = 5;
    float spacing = 2.0f;

    for (int i = -gridSize; i <= gridSize; i++) {
        for (int j = -gridSize; j <= gridSize; j++) {
            Cube cube;
            cube.position = glm::vec3(i * spacing, 0.0f, j * spacing);
            cube.size = 1.0f;
            cube.color = glm::vec3(1.0f, 1.0f, 1.0f); // Color is not used now, but we keep it for future use
            cube.gridPosition = glm::vec2(i + gridSize, j + gridSize); // Set grid position
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

void initializeShaders() {
    // Cube shader
    const char* cubeVertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float time;
    uniform vec2 gridPosition;

    void main() {
        float frequency = 2.0;
        float phaseShift = (gridPosition.x + gridPosition.y) * 0.5;
        float verticalDisplacement = sin(time * frequency + phaseShift) * 0.5;
        vec3 newPos = aPos + vec3(0.0, verticalDisplacement, 0.0);
        gl_Position = projection * view * model * vec4(newPos, 1.0);
    }
    )";

    const char* cubeFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    uniform float time;
    uniform vec2 gridPosition;

    void main() {
        float frequency = 2.0;
        float phaseShift = (gridPosition.x + gridPosition.y) * 0.5;
        float t = time * frequency + phaseShift;
        
        vec3 color;
        color.r = sin(t) * 0.5 + 0.5; // Red channel oscillates with sine wave
        color.g = sin(t + 2.0 * 3.14159 / 3.0) * 0.5 + 0.5; // Green channel shifted by 2?/3
        color.b = sin(t + 4.0 * 3.14159 / 3.0) * 0.5 + 0.5; // Blue channel shifted by 4?/3

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

    // Character shader
    const char* characterVertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
    )";

    const char* characterFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D texture_diffuse;
    void main() {
        FragColor = texture(texture_diffuse, TexCoord);
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
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return;
    }

    // Clear aggregated vertices before loading a new model
    aggregatedVertices.clear();

    // Process the root node recursively
    processNode(scene->mRootNode, scene);

    // Compute a single AABB for the entire model
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

    glBindVertexArray(0);

    mesh.indices = indices;
    loadedMeshes.push_back(mesh);
}

void processNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene);
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

void processMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 vector;

        // Process vertex positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.Position = vector;

        // Process vertex texture coordinates
        if (mesh->mTextureCoords[0]) {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.TexCoord = vec;
        }
        else {
            vertex.TexCoord = glm::vec2(0.0f, 0.0f);
        }

        vertices.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // Add vertices to the global aggregated vertices
    aggregatedVertices.insert(aggregatedVertices.end(), vertices.begin(), vertices.end());

    // Store the mesh data
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
