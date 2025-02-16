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

// 🔹 Function: Load all embeddings from JSON files
std::pair<Eigen::MatrixXf, std::vector<std::string>> load_embeddings(const std::string& directory) {
    std::vector<std::vector<float>> embeddings_list;
    std::vector<std::string> file_paths;
    int feature_size = -1;

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

int main() {
    httplib::Server svr;

    // 🔹 Load embeddings before starting the server
    auto [embeddings, file_paths] = load_embeddings("animals10/embedding/");
    QueryEngine query_engine(embeddings, file_paths);
    std::cout << "Loaded " << embeddings.rows() << " embeddings.\n";

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK", "text/plain");
    });

    svr.Post("/query", [&query_engine](const httplib::Request& req, httplib::Response& res) {
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
