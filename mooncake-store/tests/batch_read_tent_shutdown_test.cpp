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

// A standalone executable so no earlier test can initialize Slab<Batch>.
// Return from main with an unfinished TENT read: do not use _exit or a death
// test's implicit _exit, which would bypass the static destructors under test.
#include <glog/logging.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "transfer_task.h"
#include "tent/runtime/slab.h"
#include "tent/runtime/transfer_engine_impl.h"
#include "tent/runtime/transport.h"
#include "tent/transfer_engine.h"

namespace mooncake {

class TransferEngineImplTestPeer {
   public:
    static void installTransport(TransferEngine& engine,
                                 std::shared_ptr<tent::Transport> transport) {
        CHECK(engine.isUsingTent());
        engine.impl_tent_->impl_->swapTransportForTest(tent::TCP,
                                                       std::move(transport));
    }

    static size_t batchCount(TransferEngine& engine) {
        return engine.impl_tent_->impl_->aliveBatchCountForTest();
    }
};

namespace {

using namespace std::chrono_literals;

struct ExitProbe {
    std::mutex mutex;
    std::condition_variable cv;
    const std::thread::id caller = std::this_thread::get_id();
    bool poll_blocked = false;
    bool teardown_started = false;
    size_t teardown_polls = 0;
};

// The synchronization objects and caller buffers must outlive static teardown,
// too. This test isolates the pool's lifetime from caller-buffer ownership.
ExitProbe* const probe = new ExitProbe;
std::array<char, 64> source{};
std::array<char, 64> destination{};

void PollDuringStaticTeardown() {
    std::unique_lock<std::mutex> lock(probe->mutex);
    probe->teardown_started = true;
    probe->cv.notify_all();
    // The second poll enters TENT again, reading its actual Batch as well as
    // our sub-batch after the old Slab destructors would have freed them.
    CHECK(probe->cv.wait_for(lock, 5s,
                             [] { return probe->teardown_polls >= 2; }));
}

class PendingSubBatch : public tent::Transport::SubBatch {
   public:
    size_t size() const override { return count; }
    size_t count = 0;
};

class PendingTransport : public tent::Transport {
   public:
    explicit PendingTransport(bool slab_subbatch)
        : slab_subbatch_(slab_subbatch) {
        caps.dram_to_dram = true;
    }

    tent::Status allocateSubBatch(SubBatchRef& batch, size_t) override {
        batch = slab_subbatch_ ? tent::Slab<PendingSubBatch>::Get().allocate()
                               : new PendingSubBatch;
        return tent::Status::OK();
    }

    tent::Status freeSubBatch(SubBatchRef&) override {
        // Nothing completes in this test, so reclamation must not reach here.
        LOG(FATAL) << "Unexpected reclamation of a pending sub-batch";
        return tent::Status::InternalError("pending sub-batch reclaimed");
    }

    tent::Status submitTransferTasks(
        SubBatchRef batch,
        const std::vector<tent::Request>& requests) override {
        static_cast<PendingSubBatch*>(batch)->count += requests.size();
        return tent::Status::OK();
    }

    tent::Status getTransferStatus(SubBatchRef batch, int task_id,
                                   tent::TransferStatus& status) override {
        const bool background = std::this_thread::get_id() != probe->caller;
        std::unique_lock<std::mutex> lock(probe->mutex);
        if (background) {
            probe->poll_blocked = true;
            probe->cv.notify_all();
            probe->cv.wait(lock, [] { return probe->teardown_started; });
        }
        // This access intentionally happens AFTER the teardown barrier. ASan
        // catches a freed sub-batch here, or a freed Batch in TENT's next poll.
        CHECK_GE(task_id, 0);
        CHECK_LT(static_cast<size_t>(task_id),
                 static_cast<PendingSubBatch*>(batch)->count);
        status.s = tent::PENDING;
        status.transferred_bytes = 0;
        if (background) {
            ++probe->teardown_polls;
            probe->cv.notify_all();
        }
        return tent::Status::OK();
    }

    tent::Status addMemoryBuffer(tent::BufferDesc& desc,
                                 const tent::MemoryOptions&) override {
        desc.transports.push_back(tent::TCP);
        return tent::Status::OK();
    }

    tent::Status removeMemoryBuffer(tent::BufferDesc&) override {
        return tent::Status::OK();
    }

   private:
    const bool slab_subbatch_;
};

void Run(bool slab_subbatch) {
    TransferEngine engine(false);
    CHECK(engine.isUsingTent());
    CHECK_EQ(engine.init("P2PHANDSHAKE", "127.0.0.1:0", "", 0, "tcp"), 0);
    TransferEngineImplTestPeer::installTransport(
        engine, std::make_shared<PendingTransport>(slab_subbatch));
    CHECK_EQ(engine.registerLocalMemory(source.data(), source.size()), 0);
    CHECK_EQ(engine.registerLocalMemory(destination.data(), destination.size()),
             0);
    std::shared_ptr<StorageBackend> backend;
    TransferSubmitter submitter(engine, backend, engine.getLocalIpAndPort());

    // Initialize only Reclaimer, without creating any TENT batch. Register the
    // barrier next, then submit the first real batch. With the old Get(), exit
    // order is Slab destructors -> barrier -> Reclaimer destructor. This makes
    // the bad interleaving deterministic without a production failpoint.
    CHECK(submitter.submitBatchRead({}, {}).wait().empty());
    CHECK_EQ(TransferEngineImplTestPeer::batchCount(engine), 0);
    CHECK_EQ(std::atexit(PollDuringStaticTeardown), 0);

    MemoryDescriptor memory;
    memory.buffer_descriptor.buffer_address_ =
        reinterpret_cast<uintptr_t>(source.data());
    memory.buffer_descriptor.size_ = source.size();
    memory.buffer_descriptor.transport_endpoint_ = engine.getLocalIpAndPort();
    Replica::Descriptor replica;
    replica.descriptor_variant = memory;
    replica.status = ReplicaStatus::COMPLETE;
    auto operation = submitter.submitBatchRead(
        {replica}, {{{destination.data(), destination.size()}}}, 0ms);
    const auto& results = operation.wait();
    CHECK_EQ(results.size(), 1);
    CHECK_EQ(results.front(), ErrorCode::TRANSFER_FAIL);
    // The transport never completes, so this task will remain live until exit.
    std::unique_lock<std::mutex> lock(probe->mutex);
    CHECK(probe->cv.wait_for(lock, 5s, [] { return probe->poll_blocked; }));
}

}  // namespace
}  // namespace mooncake

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    CHECK_EQ(setenv("MC_USE_TENT", "1", 1), 0);
    CHECK_EQ(setenv("MC_FORCE_TCP", "1", 1), 0);
    CHECK_EQ(unsetenv("MC_TENT_CONF"), 0);
    mooncake::Run(argc == 2 && std::string_view(argv[1]) == "--slab-subbatch");
    // Keep logging available for the exit probe and Reclaimer. Returning runs
    // real static teardown; an external CTest timeout also detects deadlocks.
    return 0;
}
