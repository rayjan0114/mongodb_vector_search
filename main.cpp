// main.cpp
#include "httplib.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

std::vector<float> load_embedding(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    nlohmann::json j;
    file >> j;

    return j.get<std::vector<float>>();
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

    std::cout << "Server started on http://0.0.0.0:1234\n";
    svr.listen("0.0.0.0", 1234);

    return 0;
}
