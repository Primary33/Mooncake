// transfer_task_test.cpp
#include "transfer_task.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "types.h"
#include "pinned_buffer_pool.h"
#if defined(USE_CUDA) || defined(MOONCAKE_TEST_CUDA_H2D)
#include <cuda_runtime_api.h>
#endif

namespace mooncake {

// Test fixture for TransferTask tests
// TODO: Currently, this test does not cover TransferSubmitter and
// TransferEngine integration. Will add more tests in the future.
class ScopedEnvVar {
   public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old_value = std::getenv(name)) {
            had_old_value_ = true;
            old_value_ = old_value;
        }
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~ScopedEnvVar() {
        if (had_old_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

   private:
    std::string name_;
    bool had_old_value_ = false;
    std::string old_value_;
};

class TransferTaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize glog for logging
        google::InitGoogleLogging("TransferTaskTest");
        FLAGS_logtostderr = 1;  // Output logs to stderr
        unsetenv("MC_STORE_MEMCPY");
    }

    void TearDown() override {
        unsetenv("MC_STORE_MEMCPY");
        // Cleanup glog
        google::ShutdownGoogleLogging();
    }
};

// Test MemcpyOperationState functionality
TEST_F(TransferTaskTest, MemcpyOperationState) {
    auto state = std::make_shared<MemcpyOperationState>();

    // Initially not completed
    EXPECT_FALSE(state->is_completed());
    EXPECT_EQ(state->get_strategy(), TransferStrategy::LOCAL_MEMCPY);

    // Set completed with success
    state->set_completed(ErrorCode::OK);
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);
}

// Test MemcpyWorkerPool basic functionality
TEST_F(TransferTaskTest, MemcpyWorkerPoolBasic) {
    MemcpyWorkerPool pool;

    const size_t data_size = 512;
    std::vector<char> src_data(data_size, 'X');
    std::vector<char> dest_data(data_size, 'Y');

    auto state = std::make_shared<MemcpyOperationState>();

    // Create memcpy operations
    std::vector<MemcpyOperation> operations;
    operations.emplace_back(dest_data.data(), src_data.data(), data_size);

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify data was copied correctly
    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_EQ(dest_data[i], 'X');
    }
}

// Test multiple memcpy operations in one task
TEST_F(TransferTaskTest, MemcpyWorkerPoolMultipleOperations) {
    MemcpyWorkerPool pool;

    const size_t num_ops = 3;
    const size_t data_size = 256;

    std::vector<std::vector<char>> src_buffers(num_ops);
    std::vector<std::vector<char>> dest_buffers(num_ops);

    // Initialize source buffers with different patterns
    for (size_t i = 0; i < num_ops; ++i) {
        src_buffers[i].resize(data_size, 'A' + i);
        dest_buffers[i].resize(data_size, 'Z');
    }

    auto state = std::make_shared<MemcpyOperationState>();

    // Create multiple memcpy operations
    std::vector<MemcpyOperation> operations;
    for (size_t i = 0; i < num_ops; ++i) {
        operations.emplace_back(dest_buffers[i].data(), src_buffers[i].data(),
                                data_size);
    }

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify all data was copied correctly
    for (size_t i = 0; i < num_ops; ++i) {
        for (size_t j = 0; j < data_size; ++j) {
            EXPECT_EQ(dest_buffers[i][j], 'A' + i);
        }
    }
}

TEST_F(TransferTaskTest, TransferScatterHandlesFragmentedCpuBuffers) {
    constexpr size_t kBufferSize = 512;
    constexpr size_t kFragmentCount = 128;
    std::vector<char> source(kBufferSize), destination(kBufferSize, 0);
    std::iota(source.begin(), source.end(), 0);
    std::vector<size_t> completions(kFragmentCount, 0);
    std::vector<size_t> destination_offsets, source_offsets,
        lengths(kFragmentCount, 1);
    for (size_t i = 0; i < kFragmentCount; ++i) {
        destination_offsets.push_back(i * 2);
        source_offsets.push_back(i * 3);
    }

    TransferEngine engine(false);
    ASSERT_EQ(engine.init("P2PHANDSHAKE", "localhost:17931"), 0);
    if (!engine.isUsingTent()) {
        ASSERT_NE(engine.installTransport("tcp", nullptr), nullptr);
    }
    ASSERT_EQ(engine.registerLocalMemory(source.data(), source.size(), "cpu:0"),
              0);
    ASSERT_EQ(engine.registerLocalMemory(destination.data(), destination.size(),
                                         "cpu:0"),
              0);

    ASSERT_TRUE(engine
                    .transferScatter({{
                        .opcode = TransferRequest::READ,
                        .remote_segment = engine.getLocalIpAndPort(),
                        .remote_base_offset =
                            reinterpret_cast<uintptr_t>(source.data()),
                        .remote_size = source.size(),
                        .local_buffer = destination.data(),
                        .local_capacity = destination.size(),
                        .local_offsets = destination_offsets,
                        .remote_offsets = source_offsets,
                        .lengths = lengths,
                        .on_fragment_complete =
                            [&](size_t i, const Status& status) {
                                EXPECT_TRUE(status.ok());
                                ++completions[i];
                            },
                    }})
                    .ok());
    for (size_t i = 0; i < kFragmentCount; ++i) {
        EXPECT_EQ(destination[destination_offsets[i]],
                  source[source_offsets[i]]);
        EXPECT_EQ(completions[i], 1u);
    }
}

#ifdef USE_CUDA
TEST_F(TransferTaskTest, TransferScatterHandlesFragmentedGpuBuffers) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device is unavailable";
    }

    constexpr size_t kBufferSize = 512;
    constexpr size_t kFragmentCount = 128;
    std::vector<char> source(kBufferSize), cpu_destination(kBufferSize, 0);
    std::iota(source.begin(), source.end(), 0);
    void *gpu_source = nullptr, *gpu_destination = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_source, kBufferSize), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_destination, kBufferSize), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gpu_source, source.data(), kBufferSize,
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    TransferEngine engine(false);
    ASSERT_EQ(engine.init("P2PHANDSHAKE", "localhost:17932"), 0);
    if (!engine.isUsingTent()) {
        ASSERT_NE(engine.installTransport("tcp", nullptr), nullptr);
    }
    ASSERT_EQ(engine.registerLocalMemory(source.data(), source.size(), "cpu:0"),
              0);
    ASSERT_EQ(engine.registerLocalMemory(cpu_destination.data(),
                                         cpu_destination.size(), "cpu:0"),
              0);
    ASSERT_EQ(engine.registerLocalMemory(gpu_source, kBufferSize, "cuda:0"), 0);
    ASSERT_EQ(
        engine.registerLocalMemory(gpu_destination, kBufferSize, "cuda:0"), 0);

    std::vector<size_t> destination_offsets, source_offsets,
        lengths(kFragmentCount, 1);
    std::vector<char> expected(kBufferSize, 0), actual(kBufferSize);
    for (size_t i = 0; i < kFragmentCount; ++i) {
        destination_offsets.push_back(i * 2);
        source_offsets.push_back(i * 3);
        expected[i * 2] = source[i * 3];
    }
    auto make_transfer = [&](void* remote_source, void* local_destination) {
        return TransferEngine::ScatterTransferRange{
            .opcode = TransferRequest::READ,
            .remote_segment = engine.getLocalIpAndPort(),
            .remote_base_offset = reinterpret_cast<uintptr_t>(remote_source),
            .remote_size = kBufferSize,
            .local_buffer = local_destination,
            .local_capacity = kBufferSize,
            .local_offsets = destination_offsets,
            .remote_offsets = source_offsets,
            .lengths = lengths,
            .on_fragment_complete = {},
        };
    };

    auto invalid_transfer = make_transfer(source.data(), gpu_destination);
    invalid_transfer.remote_offsets = {};
    EXPECT_TRUE(engine.transferScatter({invalid_transfer}).IsInvalidArgument());

    ASSERT_EQ(cudaMemset(gpu_destination, 0, kBufferSize), cudaSuccess);
    ASSERT_TRUE(
        engine.transferScatter({make_transfer(source.data(), gpu_destination)})
            .ok());
    ASSERT_EQ(cudaMemcpy(actual.data(), gpu_destination, kBufferSize,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual, expected);

    ASSERT_TRUE(engine
                    .transferScatter(
                        {make_transfer(gpu_source, cpu_destination.data())})
                    .ok());
    EXPECT_EQ(cpu_destination, expected);

    ASSERT_EQ(cudaMemset(gpu_destination, 0, kBufferSize), cudaSuccess);
    auto operation =
        engine.submitScatter({make_transfer(gpu_source, gpu_destination)});
    ASSERT_EQ(engine.freeEngine(), 0);
    ASSERT_TRUE(operation.wait().ok());
    ASSERT_EQ(cudaMemcpy(actual.data(), gpu_destination, kBufferSize,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual, expected);

    EXPECT_EQ(cudaFree(gpu_source), cudaSuccess);
    EXPECT_EQ(cudaFree(gpu_destination), cudaSuccess);
}
#endif

// Same-host endpoints from different processes are not locally addressable.
TEST_F(TransferTaskTest, IsSameProcessEndpoint) {
    // Empty inputs -> not same-process (cannot prove locality).
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("", ""));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("", "192.168.1.10:12345"));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345", ""));

    // Identical ip:port -> same process.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345",
                                                         "192.168.1.10:12345"));

    // Same host, different port -> different process, NOT local.
    // This is the regression case fixed by this change.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.10:12346"));

    // Different hosts -> not local.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.11:12345"));

    // Hostname endpoints (non-P2P metadata mode) compare as full strings.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-a"));
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-b"));
}

TEST_F(TransferTaskTest, BatchGetOffloadObjectHonorsLocalMemcpySetting) {
    setenv("MC_STORE_MEMCPY", "1", 1);
    TransferEngine engine(false);
    ASSERT_EQ(engine.init("P2PHANDSHAKE", "localhost:17933"), 0);
    const std::string endpoint = engine.getLocalIpAndPort();

    std::vector<char> source(512, 'A');
    std::vector<char> destination(512, 0);
    const std::vector<std::string> keys{"key"};
    const std::vector<uint64_t> pointers{
        reinterpret_cast<uintptr_t>(source.data())};
    const std::unordered_map<std::string, std::vector<Slice>> slices{
        {"key", {{nullptr, 0}, {destination.data(), destination.size()}}}};

    {
        std::shared_ptr<StorageBackend> backend;
        TransferSubmitter submitter(engine, backend, endpoint);
        auto future = submitter.submit_batch_get_offload_object(
            endpoint, keys, pointers, slices,
            OffloadBufferAccess::kLocalAddress);
        ASSERT_TRUE(future);
        EXPECT_EQ(future->strategy(), TransferStrategy::LOCAL_MEMCPY);
        EXPECT_EQ(future->get(), ErrorCode::OK);
        EXPECT_FALSE(submitter.submit_batch_get_offload_object(
            endpoint, keys, {std::numeric_limits<uint64_t>::max() - 7},
            {{"key", {{destination.data(), 16}}}},
            OffloadBufferAccess::kLocalAddress));
    }
    EXPECT_EQ(destination, source);

    setenv("MC_STORE_MEMCPY", "0", 1);
    std::shared_ptr<StorageBackend> backend;
    TransferSubmitter submitter(engine, backend, endpoint);
    EXPECT_FALSE(submitter.submit_batch_get_offload_object(
        endpoint, keys, pointers, slices, OffloadBufferAccess::kLocalAddress));
    EXPECT_EQ(engine.freeEngine(), 0);
}

#ifdef MOONCAKE_TEST_CUDA_H2D
TEST_F(TransferTaskTest, BatchGetOffloadObjectCopiesPinnedHostToGpu) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device is unavailable";
    }

    setenv("MC_STORE_MEMCPY", "1", 1);
    constexpr size_t kSourceOffset = 128;
    constexpr size_t kSize = 4096;
    auto pinned_buffer =
        PinnedBufferPool::AllocatePinned(kSourceOffset + kSize);
    ASSERT_NE(pinned_buffer.pinned_host.addr, nullptr);
    void* pinned_source = pinned_buffer.data;
    void* gpu_destination = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_destination, kSize), cudaSuccess);
    std::memset(static_cast<char*>(pinned_source) + kSourceOffset, 0x5a, kSize);

    TransferEngine engine(false);
    ASSERT_EQ(engine.init("P2PHANDSHAKE", "localhost:17934"), 0);
    const std::string endpoint = engine.getLocalIpAndPort();
    {
        std::shared_ptr<StorageBackend> backend;
        TransferSubmitter submitter(engine, backend, endpoint);
        auto future = submitter.submit_batch_get_offload_object(
            endpoint, {"gpu"},
            {reinterpret_cast<uintptr_t>(static_cast<char*>(pinned_source) +
                                         kSourceOffset)},
            {{"gpu", {{gpu_destination, kSize}}}},
            OffloadBufferAccess::kLocalAddress);
        ASSERT_TRUE(future);
        EXPECT_EQ(future->get(), ErrorCode::OK);
    }

    std::vector<unsigned char> actual(kSize);
    EXPECT_EQ(cudaMemcpy(actual.data(), gpu_destination, kSize,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual, std::vector<unsigned char>(kSize, 0x5a));
    EXPECT_EQ(engine.freeEngine(), 0);
    EXPECT_EQ(cudaFree(gpu_destination), cudaSuccess);
}
#endif
TEST_F(TransferTaskTest, CanUseLocalMemcpyRequiresSameProcessEndpoint) {
    ScopedEnvVar memcpy_enabled("MC_STORE_MEMCPY", "1");

    TransferEngine engine(false);
    ASSERT_EQ(
        engine.init("P2PHANDSHAKE", "127.0.0.1:30991", "127.0.0.1", 30991), 0);
    ASSERT_NE(engine.installTransport("tcp", nullptr), nullptr);

    std::shared_ptr<StorageBackend> storage_backend;
    TransferSubmitter submitter(engine, storage_backend, "127.0.0.1:30991");

    const auto local_endpoint = engine.getLocalIpAndPort();
    ASSERT_FALSE(local_endpoint.empty());
    EXPECT_TRUE(submitter.canUseLocalMemcpy(local_endpoint));

    EXPECT_FALSE(submitter.canUseLocalMemcpy("127.0.0.1:30992"));
    EXPECT_FALSE(submitter.canUseLocalMemcpy(""));
}

TEST_F(TransferTaskTest, CanUseLocalMemcpyHonorsMemcpyEnv) {
    ScopedEnvVar memcpy_disabled("MC_STORE_MEMCPY", "0");

    TransferEngine engine(false);
    ASSERT_EQ(
        engine.init("P2PHANDSHAKE", "127.0.0.1:30993", "127.0.0.1", 30993), 0);
    ASSERT_NE(engine.installTransport("tcp", nullptr), nullptr);

    std::shared_ptr<StorageBackend> storage_backend;
    TransferSubmitter submitter(engine, storage_backend, "127.0.0.1:30993");

    EXPECT_FALSE(submitter.canUseLocalMemcpy(engine.getLocalIpAndPort()));
}

TEST_F(TransferTaskTest, BatchWriteHonorsLocalMemcpySetting) {
    ScopedEnvVar memcpy_enabled("MC_STORE_MEMCPY", "1");

    TransferEngine engine(false);
    ASSERT_EQ(
        engine.init("P2PHANDSHAKE", "127.0.0.1:30995", "127.0.0.1", 30995), 0);

    std::vector<char> source(512, 'A');
    std::vector<char> destination(source.size(), 0);
    MemoryDescriptor memory;
    memory.buffer_descriptor.buffer_address_ =
        reinterpret_cast<uintptr_t>(destination.data());
    memory.buffer_descriptor.size_ = destination.size();
    memory.buffer_descriptor.transport_endpoint_ = engine.getLocalIpAndPort();
    memory.buffer_descriptor.protocol_ = "tcp";
    Replica::Descriptor replica;
    replica.descriptor_variant = memory;
    replica.status = ReplicaStatus::PROCESSING;

    std::shared_ptr<StorageBackend> storage_backend;
    {
        TransferSubmitter submitter(engine, storage_backend,
                                    engine.getLocalIpAndPort());
        std::vector<std::vector<Slice>> slices{
            {{source.data(), source.size()}}};
        auto future =
            submitter.submit_batch({replica}, slices, TransferRequest::WRITE);

        ASSERT_TRUE(future);
        EXPECT_EQ(future->strategy(), TransferStrategy::LOCAL_MEMCPY);
        EXPECT_EQ(future->get(), ErrorCode::OK);
    }
    EXPECT_EQ(destination, source);
    EXPECT_EQ(engine.freeEngine(), 0);
}

TEST_F(TransferTaskTest, RemoteTransferWaitCompletesCorrectly) {
    ScopedEnvVar memcpy_disabled("MC_STORE_MEMCPY", "0");
    ScopedEnvVar tent_config("MC_TENT_CONF", nullptr);
    constexpr size_t kSize = 16 * 1024 * 1024;
    std::vector<char> local(kSize, 0), remote(kSize, 'A');
    TransferEngine client(false), server(false);
    ASSERT_EQ(client.init("P2PHANDSHAKE", "127.0.0.1:0", "", 0, "tcp"), 0);
    ASSERT_EQ(server.init("P2PHANDSHAKE", "127.0.0.1:0", "", 0, "tcp"), 0);
    if (!client.isUsingTent()) {
        ASSERT_NE(client.installTransport("tcp", nullptr), nullptr);
        ASSERT_NE(server.installTransport("tcp", nullptr), nullptr);
    }
    ASSERT_EQ(client.registerLocalMemory(local.data(), local.size(), "cpu:0"),
              0);
    ASSERT_EQ(server.registerLocalMemory(remote.data(), remote.size(), "cpu:0"),
              0);

    MemoryDescriptor memory;
    memory.buffer_descriptor.buffer_address_ =
        reinterpret_cast<uintptr_t>(remote.data());
    memory.buffer_descriptor.size_ = remote.size();
    memory.buffer_descriptor.transport_endpoint_ = server.getLocalIpAndPort();
    memory.buffer_descriptor.protocol_ = "tcp";
    Replica::Descriptor replica;
    replica.descriptor_variant = memory;
    replica.status = ReplicaStatus::COMPLETE;
    std::shared_ptr<StorageBackend> backend;
    TransferSubmitter submitter(client, backend, client.getLocalIpAndPort());
    std::vector<Slice> slices{{local.data(), local.size()}};

    for (auto opcode : {TransferRequest::READ, TransferRequest::WRITE}) {
        if (opcode == TransferRequest::WRITE) {
            std::fill(local.begin(), local.end(), 'B');
        }
        auto future = submitter.submit(replica, slices, opcode);
        ASSERT_TRUE(future);
        EXPECT_EQ(future->strategy(), TransferStrategy::TRANSFER_ENGINE);
        ASSERT_EQ(future->get(), ErrorCode::OK);
        EXPECT_EQ(local, remote);
    }
    EXPECT_EQ(client.unregisterLocalMemory(local.data()), 0);
    EXPECT_EQ(server.unregisterLocalMemory(remote.data()), 0);
}

class BatchReadTest : public TransferTaskTest {
   protected:
    static constexpr size_t kObjectSize = 256;
    static constexpr size_t kObjectCount = 4;

    void SetUp() override {
        TransferTaskTest::SetUp();
        source_.resize(kObjectSize * kObjectCount);
        destination_.resize(source_.size(), '?');
        for (size_t i = 0; i < kObjectCount; ++i)
            std::fill_n(source_.data() + i * kObjectSize, kObjectSize, 'A' + i);
        client_ = std::make_unique<TransferEngine>(false);
        server_ = std::make_unique<TransferEngine>(false);
        ASSERT_EQ(client_->init("P2PHANDSHAKE", "127.0.0.1:0", "", 0, "tcp"),
                  0);
        ASSERT_EQ(server_->init("P2PHANDSHAKE", "127.0.0.1:0", "", 0, "tcp"),
                  0);
        if (!client_->isUsingTent()) {
            ASSERT_NE(client_->installTransport("tcp", nullptr), nullptr);
            ASSERT_NE(server_->installTransport("tcp", nullptr), nullptr);
        }
        ASSERT_EQ(client_->registerLocalMemory(destination_.data(),
                                               destination_.size()),
                  0);
        destination_registered_ = true;
        ASSERT_EQ(server_->registerLocalMemory(source_.data(), source_.size()),
                  0);
        source_registered_ = true;
        submitter_ = std::make_unique<TransferSubmitter>(
            *client_, backend_, client_->getLocalIpAndPort());
    }

    void TearDown() override {
        submitter_.reset();
        if (destination_registered_) {
            EXPECT_EQ(client_->unregisterLocalMemory(destination_.data()), 0);
        }
        if (source_registered_) {
            EXPECT_EQ(server_->unregisterLocalMemory(source_.data()), 0);
        }
        client_.reset();
        server_.reset();
        TransferTaskTest::TearDown();
    }

    std::vector<Replica::Descriptor> Replicas() const {
        std::vector<Replica::Descriptor> replicas;
        for (size_t i = 0; i < kObjectCount; ++i) {
            MemoryDescriptor memory;
            memory.buffer_descriptor.buffer_address_ =
                reinterpret_cast<uintptr_t>(source_.data() + i * kObjectSize);
            memory.buffer_descriptor.size_ = kObjectSize;
            memory.buffer_descriptor.transport_endpoint_ =
                server_->getLocalIpAndPort();
            Replica::Descriptor replica;
            replica.descriptor_variant = memory;
            replica.status = ReplicaStatus::COMPLETE;
            replicas.push_back(std::move(replica));
        }
        return replicas;
    }

    std::vector<std::vector<Slice>> Destinations() {
        std::vector<std::vector<Slice>> slices;
        for (size_t i = 0; i < kObjectCount; ++i)
            slices.push_back(
                {{destination_.data() + i * kObjectSize, kObjectSize}});
        return slices;
    }

    ScopedEnvVar tent_conf_{"MC_TENT_CONF", nullptr};
    std::vector<char> source_;
    std::vector<char> destination_;
    std::unique_ptr<TransferEngine> client_;
    std::unique_ptr<TransferEngine> server_;
    std::shared_ptr<StorageBackend> backend_;
    std::unique_ptr<TransferSubmitter> submitter_;
    bool destination_registered_ = false;
    bool source_registered_ = false;
};

TEST_F(BatchReadTest, InvalidObjectsDoNotPreventValidReads) {
    auto slices = Destinations();
    slices[0][0].size -= 1;
    slices[1][0].ptr = nullptr;
    auto operation = submitter_->submitBatchRead(Replicas(), slices);
    EXPECT_EQ(operation.wait(),
              (std::vector<ErrorCode>{ErrorCode::TRANSFER_FAIL,
                                      ErrorCode::TRANSFER_FAIL, ErrorCode::OK,
                                      ErrorCode::OK}));
    EXPECT_TRUE(std::all_of(destination_.begin(),
                            destination_.begin() + 2 * kObjectSize,
                            [](char value) { return value == '?'; }));
    EXPECT_TRUE(std::equal(destination_.begin() + 2 * kObjectSize,
                           destination_.end(),
                           source_.begin() + 2 * kObjectSize));
}

TEST_F(BatchReadTest, MultipleSlicesAndZeroLengthFragments) {
    auto slices = Destinations();
    for (size_t i = 0; i < slices.size(); ++i) {
        auto* buffer = destination_.data() + i * kObjectSize;
        slices[i] = {
            {buffer, 17}, {nullptr, 0}, {buffer + 17, kObjectSize - 17}};
    }
    auto operation = submitter_->submitBatchRead(Replicas(), slices);
    auto moved = std::move(operation);
    const std::vector<ErrorCode> expected(kObjectCount, ErrorCode::OK);
    EXPECT_EQ(moved.wait(), expected);
    EXPECT_EQ(moved.wait(), expected);
    EXPECT_EQ(destination_, source_);
}

TEST_F(BatchReadTest, DestructionDrainsOutstandingReads) {
    {
        auto operation =
            submitter_->submitBatchRead(Replicas(), Destinations());
        // The caller is allowed to abandon the result, but not the transfer's
        // ownership of its buffers. Destruction must finish before we inspect.
    }
    EXPECT_EQ(destination_, source_);
}

TEST_F(BatchReadTest, EmptyAndMismatchedInputs) {
    EXPECT_TRUE(submitter_->submitBatchRead({}, {}).wait().empty());
    EXPECT_EQ(submitter_->submitBatchRead(Replicas(), {}).wait(),
              std::vector<ErrorCode>(kObjectCount, ErrorCode::TRANSFER_FAIL));
    EXPECT_TRUE(std::all_of(destination_.begin(), destination_.end(),
                            [](char value) { return value == '?'; }));
}

TEST_F(BatchReadTest, CompletionPreservesSharedRemoteHandle) {
    const auto endpoint = server_->getLocalIpAndPort();
    const auto handle = client_->openSegment(endpoint);
    ASSERT_NE(handle, static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT));
    std::vector<SegmentBufferInfo> buffers;
    ASSERT_EQ(client_->getSegmentBuffers(handle, buffers), 0);

    for (int iteration = 0; iteration < 3; ++iteration) {
        auto slices = Destinations();
        auto expected = std::vector<ErrorCode>(kObjectCount, ErrorCode::OK);
        if (iteration == 1) {
            slices[0][0].size -= 1;
            expected[0] = ErrorCode::TRANSFER_FAIL;
        }
        {
            auto operation = submitter_->submitBatchRead(Replicas(), slices);
            EXPECT_EQ(operation.wait(), expected);
        }
        // Another caller may still be using this handle after the batch is
        // complete, including when one object failed validation.
        ASSERT_EQ(client_->getSegmentBuffers(handle, buffers), 0);
        EXPECT_EQ(client_->openSegment(endpoint), handle);
    }
}

TEST_F(BatchReadTest, ConcurrentBatchesAndSingleReadsShareRemoteHandle) {
    constexpr size_t kReaders = 4;
    constexpr size_t kIterations = 32;
    const auto replicas = Replicas();
    std::vector<char> destinations(source_.size() * kReaders, '?');
    ASSERT_EQ(
        client_->registerLocalMemory(destinations.data(), destinations.size()),
        0);
    ScopedEnvVar memcpy_config("MC_STORE_MEMCPY", "0");
    TransferSubmitter submitter(*client_, backend_,
                                client_->getLocalIpAndPort());
    std::atomic<size_t> failures{0};
    std::barrier start(kReaders);
    std::vector<std::thread> readers;
    for (size_t reader = 0; reader < kReaders; ++reader) {
        readers.emplace_back([&, reader] {
            auto* destination = destinations.data() + reader * source_.size();
            std::vector<std::vector<Slice>> slices;
            for (size_t i = 0; i < kObjectCount; ++i)
                slices.push_back(
                    {{destination + i * kObjectSize, kObjectSize}});
            start.arrive_and_wait();
            for (size_t iteration = 0;
                 iteration < kIterations && failures.load() == 0; ++iteration) {
                std::fill_n(destination, source_.size(), '?');
                if (reader == 0) {
                    for (size_t i = 0; i < kObjectCount; ++i) {
                        auto future = submitter.submit(replicas[i], slices[i],
                                                       TransferRequest::READ);
                        if (!future ||
                            future->strategy() !=
                                TransferStrategy::TRANSFER_ENGINE ||
                            future->get() != ErrorCode::OK) {
                            ++failures;
                            return;
                        }
                    }
                } else {
                    auto operation =
                        submitter.submitBatchRead(replicas, slices);
                    const auto& results = operation.wait();
                    if (results.size() != kObjectCount ||
                        !std::all_of(results.begin(), results.end(),
                                     [](ErrorCode result) {
                                         return result == ErrorCode::OK;
                                     })) {
                        ++failures;
                        return;
                    }
                }
                if (!std::equal(source_.begin(), source_.end(), destination)) {
                    ++failures;
                    return;
                }
            }
        });
    }
    for (auto& reader : readers) reader.join();
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(client_->unregisterLocalMemory(destinations.data()), 0);
}

// Test TransferStrategy enum and stream operator
TEST_F(TransferTaskTest, TransferStrategyEnum) {
    // Test enum values
    EXPECT_EQ(static_cast<int>(TransferStrategy::LOCAL_MEMCPY), 0);
    EXPECT_EQ(static_cast<int>(TransferStrategy::TRANSFER_ENGINE), 1);
    EXPECT_EQ(static_cast<int>(TransferStrategy::FILE_READ), 2);
    EXPECT_EQ(static_cast<int>(TransferStrategy::EMPTY), 3);
    EXPECT_EQ(static_cast<int>(TransferStrategy::SPDK_NVMF), 4);

    // Test stream operator
    std::ostringstream oss;
    oss << TransferStrategy::LOCAL_MEMCPY;
    EXPECT_EQ(oss.str(), "LOCAL_MEMCPY");

    oss.str("");
    oss << TransferStrategy::TRANSFER_ENGINE;
    EXPECT_EQ(oss.str(), "TRANSFER_ENGINE");

    oss.str("");
    oss << TransferStrategy::SPDK_NVMF;
    EXPECT_EQ(oss.str(), "SPDK_NVMF");

    oss.str("");
    oss << TransferStrategy::FILE_READ;
    EXPECT_EQ(oss.str(), "FILE_READ");

    oss.str("");
    oss << TransferStrategy::EMPTY;
    EXPECT_EQ(oss.str(), "EMPTY");
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
