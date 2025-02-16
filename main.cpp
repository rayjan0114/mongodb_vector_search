#include "httplib.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <vector>
#include <queue>
#include <string>

namespace fs = std::filesystem;

// 🔹 Utility: Convert embedding path to image path
std::string embedding_path_to_image_path(const std::string& embedding_path) {
    std::string image_path = embedding_path;
    return image_path.replace(image_path.find("embedding"), 9, "raw-img").replace(image_path.find(".json"), 5, ".jpg");
}

std::string image_path_to_embedding_path(const std::string& image_path) {
    std::string embedding_path = image_path;
    return embedding_path.replace(embedding_path.find("raw-img"), 7, "embedding").replace(embedding_path.find(".jpg"), 4, ".json");
}

// 🔹 Function: Load all embeddings from JSON files
std::pair<Eigen::MatrixXf, std::vector<std::string>> load_embeddings(const std::string& directory) {
    std::vector<std::vector<float>> embeddings_list;
    std::vector<std::string> file_paths;
    std::size_t feature_size = -1;

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.path().extension() == ".json") {
            std::ifstream file(entry.path());
            if (!file.is_open()) {
                std::cerr << "Warning: Could not open file: " << entry.path() << "\n";
                continue;
            }

            try {
                nlohmann::json j;
                file >> j;
                std::vector<float> embedding = j.get<std::vector<float>>();

                if (feature_size == -1) {
                    feature_size = embedding.size();
                } else if (embedding.size() != feature_size) {
                    std::cerr << "Skipping " << entry.path() << " due to size mismatch.\n";
                    continue;
                }

                embeddings_list.push_back(embedding);
                file_paths.push_back(entry.path().string());
            } catch (const std::exception& e) {
                std::cerr << "Error parsing file " << entry.path() << ": " << e.what() << "\n";
            }
        }
    }

    Eigen::MatrixXf embeddings(embeddings_list.size(), feature_size);
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
        embeddings.row(i) = Eigen::VectorXf::Map(embeddings_list[i].data(), feature_size);
    }

    return {embeddings, file_paths};
}

// 🔹 QueryEngine: Efficient similarity search
class QueryEngine {
private:
    Eigen::MatrixXf embeddings;
    std::vector<std::string> file_paths;

public:
    QueryEngine(const Eigen::MatrixXf& embeddings, const std::vector<std::string>& file_paths)
        : embeddings(embeddings), file_paths(file_paths) {}

    std::vector<std::pair<std::string, float>> query(const Eigen::VectorXf& query_embedding, int topk, const std::string& mode = "cosine") {
        std::vector<std::pair<float, int>> topk_indices;

        if (mode == "cosine") {
            Eigen::VectorXf query_norm = query_embedding.normalized();
            Eigen::MatrixXf embeddings_norm = embeddings.rowwise().normalized();
            Eigen::VectorXf similarities = embeddings_norm * query_norm;

            auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first < b.first;  // Max heap for highest similarity
            };
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(compare)> pq(compare);

            for (int i = 0; i < similarities.size(); ++i) {
                pq.emplace(similarities(i), i);
            }
            
            for (int i = 0; i < topk && !pq.empty(); ++i) {
                topk_indices.push_back(pq.top());
                pq.pop();
            }

        } else if (mode == "euclidean") {
            Eigen::VectorXf distances = (embeddings.rowwise() - query_embedding.transpose()).rowwise().norm();

            auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;  // Min heap for smallest distance
            };
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(compare)> pq(compare);

            for (int i = 0; i < distances.size(); ++i) {
                pq.emplace(distances(i), i);
            }

            for (int i = 0; i < topk && !pq.empty(); ++i) {
                topk_indices.push_back(pq.top());
                pq.pop();
            }

        } else {
            throw std::invalid_argument("Invalid mode: " + mode);
        }

        std::vector<std::pair<std::string, float>> results;
        for (const auto& [value, idx] : topk_indices) {
            results.emplace_back(embedding_path_to_image_path(file_paths[idx]), value);
        }

        return results;
    }
};


void enable_cors(httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

std::unordered_map<std::string, std::string> read_image_name_to_path() {
    std::ifstream file("image_name_to_path.json");
    nlohmann::json j;
    file >> j;
    return j.get<std::unordered_map<std::string, std::string>>();
}

int main() {
    httplib::Server svr;

    std::cout << "Loading embeddings..." << std::endl;
    auto [embeddings, file_paths] = load_embeddings("animals10/embedding/");
    auto image_name_to_path = read_image_name_to_path();
    QueryEngine query_engine(embeddings, file_paths);
    std::cout << "Loaded " << embeddings.rows() << " embeddings.\n";

    // 🔹 Handle preflight requests (CORS)
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        enable_cors(res);
        res.status = 200; // Allow the preflight check to succeed
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        enable_cors(res);
        res.set_content("OK", "text/plain");
    });

    svr.Get("/get_image_info", [&](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        std::string file_name = req.get_param_value("file");
        std::string file_path = image_name_to_path[file_name];
        std::string embedding_path = image_path_to_embedding_path(file_path);
        std::cout << "file_path: " << file_path << std::endl;
        std::ifstream file(embedding_path);
        if (!file.is_open()) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }
        nlohmann::json embedding_json;
        file >> embedding_json;
    
        nlohmann::json response;
        response["embedding"] = embedding_json;  // The array from the .json file
        response["file_path"] = file_path;       // The original (raw image) path

        res.set_content(response.dump(), "application/json");
    });

    svr.Get("/get_image", [](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);
        if (!req.has_param("file")) {
            res.status = 400;
            res.set_content("Missing file parameter", "text/plain");
            return;
        }

        std::string file_path =  req.get_param_value("file");
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        res.set_content(buffer.str(), "image/jpeg");
    });


    svr.Post("/query", [&query_engine](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);
        try {
            auto json = nlohmann::json::parse(req.body);
            std::vector<float> embedding_vector = json["embedding"].get<std::vector<float>>();

            Eigen::VectorXf query_embedding = Eigen::Map<Eigen::VectorXf>(embedding_vector.data(), embedding_vector.size());
            int topk = json.value("topk", 5);
            std::string mode = json.value("mode", "cosine");

            auto results = query_engine.query(query_embedding, topk, mode);

            nlohmann::json response_json;
            for (const auto& [file, score] : results) {
                response_json["matches"].push_back({{"file", file}, {"score", score}});
            }

            res.set_content(response_json.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
        }
    });

    std::cout << "Server started on http://0.0.0.0:8765\n";
    svr.listen("0.0.0.0", 8765);

    return 0;
}
