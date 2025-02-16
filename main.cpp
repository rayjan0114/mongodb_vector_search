#include "httplib.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <vector>


namespace fs = std::filesystem;

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
                }

                embeddings_list.push_back(embedding);
                file_paths.push_back(entry.path().string());
            } catch (const std::exception& e) {
                std::cerr << "Error parsing file " << entry.path() << ": " << e.what() << "\n";
            }
        }
    }

    int num_embeddings = embeddings_list.size();
    Eigen::MatrixXf embeddings(num_embeddings, feature_size);

    for (size_t i = 0; i < embeddings_list.size(); ++i) {
        embeddings.row(i) = Eigen::VectorXf::Map(embeddings_list[i].data(), feature_size);
    }

    return {embeddings, file_paths};
}

// Function to compute cosine similarity between a query and all stored embeddings
Eigen::VectorXf compute_cosine_similarity(const Eigen::VectorXf& query, const Eigen::MatrixXf& embeddings) {
    Eigen::VectorXf query_normalized = query.normalized();  // Normalize query
    Eigen::MatrixXf embeddings_normalized = embeddings.rowwise().normalized();  // Normalize all embeddings

    // Compute dot product between query and all embeddings (SIMD-optimized)
    return embeddings_normalized * query_normalized;
}

Eigen::VectorXf compute_euclidean_distance(const Eigen::VectorXf& query, const Eigen::MatrixXf& embeddings) {
    return (embeddings.rowwise() - query.transpose()).rowwise().norm();
}


int main() {
    Eigen::MatrixXf embeddings;
    std::vector<std::string> file_paths;
    std::tie(embeddings, file_paths) = load_embeddings("animals10/embedding/");
    std::cout << "Loaded " << embeddings.rows() << " embeddings.\n";
    for (size_t i = 0; i < std::min(static_cast<size_t>(embeddings.rows()), size_t(5)); ++i) {
        std::cout << "File: " << file_paths[i] << " | First 3 values: " << embeddings.row(i).head(3).transpose() << "\n";
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &req, httplib::Response &res) {
        res.set_content("OK", "text/plain");
    });

    svr.Post("/query", [&embeddings, &file_paths](const httplib::Request &req, httplib::Response &res) {
        try {
            auto json = nlohmann::json::parse(req.body);
            std::vector<float> embedding_vector = json["embedding"].get<std::vector<float>>();
            
            if (embedding_vector.size() != embeddings.cols()) {
                res.status = 400;
                res.set_content("Embedding size mismatch", "text/plain");
                return;
            }

            // Convert to Eigen::VectorXf
            Eigen::VectorXf query = Eigen::Map<Eigen::VectorXf>(embedding_vector.data(), embedding_vector.size());

            // Compute similarity (Choose cosine or Euclidean)
            Eigen::VectorXf similarities = compute_cosine_similarity(query, embeddings);

            // Find top 5 most similar
            std::vector<std::pair<float, std::string>> results;
            for (int i = 0; i < similarities.size(); ++i) {
                results.emplace_back(similarities(i), file_paths[i]);
            }

            // Sort descending
            std::sort(results.rbegin(), results.rend());

            // Create response JSON
            nlohmann::json response_json;
            for (int i = 0; i < std::min(5, static_cast<int>(results.size())); ++i) {
                response_json["matches"].push_back({{"similarity", results[i].first}, {"file", results[i].second}});
            }

            res.set_content(response_json.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
        }
    });


    svr.Post("/upsert", [](const httplib::Request &req, httplib::Response &res) {
        try {
            auto json = nlohmann::json::parse(req.body);
            // TODO: Implement upsert
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
        }
    });


    std::cout << "Server started on http://0.0.0.0:1234\n";
    svr.listen("0.0.0.0", 1234);

    return 0;
}
