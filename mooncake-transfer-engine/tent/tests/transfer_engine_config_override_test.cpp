// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "tent/common/config.h"
#include "tent/runtime/transfer_engine_impl.h"

namespace mooncake {
namespace tent {

namespace {

constexpr char kInvalidHostname[] = "256.256.256.256";

class EnvVarGuard {
   public:
    EnvVarGuard(const char* name, const std::string& value) : name_(name) {
        const char* old = std::getenv(name);
        if (old) {
            old_value_ = old;
            had_value_ = true;
        }
        setenv(name, value.c_str(), 1);
    }

    ~EnvVarGuard() {
        if (had_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

   private:
    std::string name_;
    std::string old_value_;
    bool had_value_ = false;
};

class TempConfigFile {
   public:
    explicit TempConfigFile(const std::string& content) {
        auto unique_name = "tent-config-" +
                           std::to_string(
                               std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count()) +
                           ".json";
        path_ = std::filesystem::temp_directory_path() / unique_name;
        std::ofstream ofs(path_);
        ofs << content;
    }

    ~TempConfigFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string path() const { return path_.string(); }

   private:
    std::filesystem::path path_;
};

TEST(TransferEngineConfigOverrideTest,
     ExplicitStringPortSurvivesMcTentConfReloadThroughConstructor) {
    TempConfigFile conf_file(R"({
        "metadata_type": "p2p",
        "metadata_servers": "127.0.0.1:2379",
        "rpc_server_hostname": "10.0.0.8",
        "rpc_server_port": 15011,
        "log_level": "warning",
        "merge_requests": false
    })");
    EnvVarGuard guard("MC_TENT_CONF", conf_file.path());

    auto config = std::make_shared<Config>();
    config->set("metadata_type", "http");
    config->set("metadata_servers", "127.0.0.1:18080/metadata");
    config->set("local_segment_name", "store-segment-A");
    config->set("rpc_server_hostname", kInvalidHostname);
    config->set("rpc_server_port", "26001");

    // Use an invalid hostname so construction stops after config merge.
    TransferEngineImpl engine(config);

    EXPECT_FALSE(engine.available());
    EXPECT_EQ(config->get("metadata_type", ""), "http");
    EXPECT_EQ(config->get("metadata_servers", ""),
              "127.0.0.1:18080/metadata");
    EXPECT_EQ(config->get("local_segment_name", ""), "store-segment-A");
    EXPECT_EQ(config->get("rpc_server_hostname", ""), kInvalidHostname);
    EXPECT_EQ(config->get("rpc_server_port", ""), "26001");
    EXPECT_EQ(engine.getRpcServerAddress(), kInvalidHostname);
    EXPECT_EQ(engine.getRpcServerPort(), 26001);

    EXPECT_EQ(config->get("log_level", ""), "warning");
    EXPECT_FALSE(config->get("merge_requests", true));
}

TEST(TransferEngineConfigOverrideTest,
     MissingExplicitKeysContinueUsingMcTentConfValuesThroughConstructor) {
    TempConfigFile conf_file(R"({
        "metadata_type": "p2p",
        "metadata_servers": "127.0.0.1:2379",
        "rpc_server_hostname": "256.256.256.256",
        "rpc_server_port": 15012
    })");
    EnvVarGuard guard("MC_TENT_CONF", conf_file.path());

    auto config = std::make_shared<Config>();
    config->set("local_segment_name", "store-segment-B");

    TransferEngineImpl engine(config);

    EXPECT_FALSE(engine.available());
    EXPECT_EQ(config->get("metadata_type", ""), "p2p");
    EXPECT_EQ(config->get("metadata_servers", ""), "127.0.0.1:2379");
    EXPECT_EQ(config->get("rpc_server_hostname", ""), kInvalidHostname);
    EXPECT_EQ(config->get("rpc_server_port", 0), 15012);
    EXPECT_EQ(config->get("local_segment_name", ""), "store-segment-B");
    EXPECT_EQ(engine.getRpcServerAddress(), kInvalidHostname);
    EXPECT_EQ(engine.getRpcServerPort(), 15012);
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
