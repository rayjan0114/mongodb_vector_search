#include "httplib.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

// Function to load embeddings into an Eigen::MatrixXf
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

int main() {
    httplib::Server svr;

    svr.Get("/hello", [](const httplib::Request &req, httplib::Response &res) {
        res.set_content("Hello route", "text/plain");
    });

    svr.Post("/query", [](const httplib::Request &req, httplib::Response &res) {
        try {
            auto json = nlohmann::json::parse(req.body);
            auto embedding = json["embedding"].get<std::vector<double>>();

            nlohmann::json response_json;
            response_json["embedding"] = embedding;

            res.set_content(response_json.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
        }
    });

    svr.Post("/upsert", [](const httplib::Request &req, httplib::Response &res) {
        try {
            auto json = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
        }
    });

    // Load all embeddings before starting the server
    auto [embeddings, file_paths] = load_embeddings("animals10/embedding/");
    std::cout << "Loaded " << embeddings.rows() << " embeddings.\n";
    // print the first 5 embeddings
    for (size_t i = 0; i < std::min(static_cast<size_t>(embeddings.rows()), size_t(5)); ++i) {
        std::cout << "File: " << file_paths[i] << " | First 3 values: " << embeddings.row(i).head(3).transpose() << "\n";
    }

    std::cout << "Server started on http://0.0.0.0:1234\n";
    svr.listen("0.0.0.0", 1234);

    return 0;
}
