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

// Regression tests for RdmaTransport::submitTransferTask()'s
// "memory region not registered" (!found_device) error path. A slice that
// already succeeded device selection and was queued into the function-local
// slices_to_post accumulator remains owned by TransferTask::slice_list. It
// must not be deallocated twice, and it must reach a terminal FAILED state if
// a later slice causes submitTransferTask() to return an error.

#include <gtest/gtest.h>

#include <memory>
#include <array>
#include <string>

#include "config.h"
#include "multi_transport.h"
#include "rdma_test_peers.h"
#include "transfer_metadata.h"
#include "transfer_engine.h"
#include "transfer_engine_impl.h"
#include "transport/rdma_transport/rdma_context.h"
#include "transport/rdma_transport/rdma_transport.h"

using namespace mooncake;

#ifdef MOONCAKE_RDMA_SUBMIT_TEST_HOOKS
namespace mooncake {
class TransferEngineImplTestPeer {
   public:
    static void bind(TransferEngine &engine,
                     const std::shared_ptr<TransferMetadata> &metadata,
                     const std::shared_ptr<RdmaTransport> &transport) {
        auto &impl = *engine.impl_;
        impl.metadata_ = metadata;
        impl.local_server_name_ = "unit-test-server:1234";
        impl.multi_transports_ =
            std::make_shared<MultiTransport>(metadata, impl.local_server_name_);
        impl.multi_transports_->transport_map_.emplace("rdma", transport);
    }
};
}  // namespace mooncake

static std::vector<void *> posted_sources;

// Exercise actual RDMA slicing, device selection and submission cleanup with
// synthetic registered ranges. Only posting/completion is replaced; no device
// is opened and these synthetic addresses must never be dereferenced.
extern "C" int
__wrap__ZN8mooncake11RdmaContext14submitPostSendERKSt6vectorIPNS_9Transport5SliceESaIS4_EE(
    RdmaContext *, const std::vector<Transport::Slice *> &slices) {
    for (auto *slice : slices) {
        posted_sources.push_back(slice->source_addr);
        slice->markSuccess();
    }
    return 0;
}
#endif

namespace {

using SegmentDesc = TransferMetadata::SegmentDesc;
using BufferDesc = TransferMetadata::BufferDesc;

class SubmitTransferTaskTest : public ::testing::Test {
   protected:
    static constexpr uint64_t kBufferAddr = 0x10000;

    std::shared_ptr<TransferMetadata> metadata_;
    std::shared_ptr<RdmaTransport> transport_;
    std::shared_ptr<RdmaContext> context_;
    uint64_t block_size_ = 0;

    void SetUp() override {
        block_size_ = globalConfig().slice_size;

        metadata_ = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
        transport_ = std::make_shared<RdmaTransport>();
        RdmaTransportTestPeer::bindMetadata(*transport_, metadata_,
                                            "unit-test-server:1234");

        // construct() is never called: no real device is opened. active()
        // defaults to true, which is all submitTransferTask() checks.
        context_ = std::make_shared<RdmaContext>(*transport_, "mlx5_unit_test");
        RdmaTransportTestPeer::addContext(*transport_, context_);

        auto desc = std::make_shared<SegmentDesc>();
        desc->name = "unit-test-server:1234";
        desc->protocol = "rdma";
        BufferDesc buffer;
        buffer.name = "cpu:0";
        buffer.addr = kBufferAddr;
        buffer.length = block_size_;
        buffer.lkey = {1};
        buffer.rkey = {1};
        desc->buffers.push_back(buffer);
        ASSERT_EQ(
            desc->topology.parse(R"({"cpu:0": [["mlx5_unit_test"], []]})"), 0);
        metadata_->addLocalSegment(LOCAL_SEGMENT_ID, desc->name,
                                   std::move(desc));
    }

    void triggerError(Transport::TransferRequest &req,
                      Transport::TransferTask &task) {
        req.opcode = Transport::TransferRequest::WRITE;
        req.source = reinterpret_cast<void *>(kBufferAddr);
        req.length = 2 * block_size_;
        req.target_id = LOCAL_SEGMENT_ID;
        req.target_offset = 0;
        task.request = &req;

        // markFailed() needs a valid BatchDesc in event-driven builds. The
        // direct task is deliberately not inserted into that BatchDesc; the ID
        // is only used by Slice::check_batch_completion().
        task.batch_id = transport_->allocateBatchID(1);
        auto status = transport_->submitTransferTask({&task});
        EXPECT_FALSE(status.ok());
        EXPECT_TRUE(status.IsAddressNotRegistered());
        ASSERT_EQ(task.slice_list.size(), 2u);
        EXPECT_EQ(transport_->freeBatchID(task.batch_id), Status::OK());
    }
};

TEST_F(SubmitTransferTaskTest, NoDuplicateSlice) {
    Transport::Slice *original = nullptr;
    {
        Transport::TransferRequest req;
        Transport::TransferTask task;
        triggerError(req, task);
        original = task.slice_list[0];
    }

    Transport::TransferRequest req;
    Transport::TransferTask task;
    triggerError(req, task);

    EXPECT_EQ(task.slice_list[0], original)
        << "the cache should legitimately reuse the first released slice";
    EXPECT_NE(task.slice_list[0], task.slice_list[1])
        << "two independent slices must never share the same Slice object";
}

TEST_F(SubmitTransferTaskTest, PartialSubmitFailsBatch) {
    auto batch_id = transport_->allocateBatchID(1);
    Transport::TransferRequest request;
    request.opcode = Transport::TransferRequest::WRITE;
    request.source = reinterpret_cast<void *>(kBufferAddr);
    request.length = 2 * block_size_;
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = 0;

    auto submit_status = transport_->submitTransfer(batch_id, {request});
    ASSERT_FALSE(submit_status.ok());
    ASSERT_TRUE(submit_status.IsAddressNotRegistered());

    Transport::TransferStatus transfer_status;
    ASSERT_EQ(transport_->getTransferStatus(batch_id, 0, transfer_status),
              Status::OK());
    EXPECT_EQ(transfer_status.s, Transport::TransferStatusEnum::FAILED);
    EXPECT_EQ(transport_->freeBatchID(batch_id), Status::OK());
}

TEST_F(SubmitTransferTaskTest, PartialSubmitFailsAllTasks) {
    auto batch_id = transport_->allocateBatchID(2);
    Transport::TransferRequest failing_request;
    failing_request.opcode = Transport::TransferRequest::WRITE;
    failing_request.source = reinterpret_cast<void *>(kBufferAddr);
    failing_request.length = 2 * block_size_;
    failing_request.target_id = LOCAL_SEGMENT_ID;
    failing_request.target_offset = 0;

    Transport::TransferRequest unstarted_request;
    unstarted_request.opcode = Transport::TransferRequest::WRITE;
    unstarted_request.source = reinterpret_cast<void *>(kBufferAddr);
    unstarted_request.length = block_size_;
    unstarted_request.target_id = LOCAL_SEGMENT_ID;
    unstarted_request.target_offset = kBufferAddr;

    auto submit_status = transport_->submitTransfer(
        batch_id, {failing_request, unstarted_request});
    ASSERT_FALSE(submit_status.ok());
    ASSERT_TRUE(submit_status.IsAddressNotRegistered());

    std::vector<Transport::TransferStatus> transfer_status;
    ASSERT_EQ(transport_->getTransferStatus(batch_id, transfer_status),
              Status::OK());
    ASSERT_EQ(transfer_status.size(), 2u);
    EXPECT_EQ(transfer_status[0].s, Transport::TransferStatusEnum::FAILED);
    EXPECT_EQ(transfer_status[1].s, Transport::TransferStatusEnum::FAILED);
    EXPECT_EQ(transport_->freeBatchID(batch_id), Status::OK());
}

TEST_F(SubmitTransferTaskTest, GroupedRequestsReportFailurePerRequest) {
    std::vector<Transport::TransferRequest> requests(2);
    for (auto &request : requests) {
        request.opcode = Transport::TransferRequest::WRITE;
        request.source = reinterpret_cast<void *>(kBufferAddr);
        request.length = block_size_;
        request.target_id = LOCAL_SEGMENT_ID;
        request.target_offset = 0;
    }
    requests[1].length = 2 * block_size_;
    std::string server_name = "unit-test-server:1234";
    MultiTransport multi_transport(metadata_, server_name);
    auto batch_id = multi_transport.allocateBatchID(1);
    auto &task = reinterpret_cast<Transport::BatchDesc *>(batch_id)
                     ->task_list.emplace_back();
    task.batch_id = batch_id;
    task.request = requests.data();
    task.request_count = requests.size();
    EXPECT_TRUE(
        transport_->submitTransferTask({&task}).IsAddressNotRegistered());
    Transport::TransferStatus status;
    ASSERT_EQ(transport_->getTransferStatus(batch_id, 0, status), Status::OK());
    EXPECT_EQ(status.s, Transport::TransferStatusEnum::FAILED);
    std::vector<Transport::TransferStatusEnum> request_statuses;
    ASSERT_EQ(multi_transport.getScatterRequestStatuses(batch_id, 0,
                                                        request_statuses),
              Status::OK());
    EXPECT_EQ(request_statuses, std::vector(2, status.s));
    EXPECT_EQ(multi_transport.freeBatchID(batch_id), Status::OK());
}

#ifdef MOONCAKE_RDMA_SUBMIT_TEST_HOOKS
class IndependentScatterRdmaTest : public SubmitTransferTaskTest {
   protected:
    void checkFailureIsolation(bool partially_registered) {
        TransferEngine engine(false);
        TransferEngineImplTestPeer::bind(engine, metadata_, transport_);
        for (size_t bad_index = 0; bad_index < 3; ++bad_index) {
            SCOPED_TRACE(bad_index);
            posted_sources.clear();
            std::array<size_t, 3> lengths{4, 4, 4};
            std::array<size_t, 1> offsets{0};
            std::array<size_t, 3> callbacks{};
            std::array<bool, 3> succeeded{};
            std::vector<TransferEngine::ScatterTransferRange> ranges;
            for (size_t i = 0; i < lengths.size(); ++i) {
                uintptr_t address = kBufferAddr + i * 4;
                if (i == bad_index) {
                    address = partially_registered ? kBufferAddr
                                                   : kBufferAddr + block_size_;
                    lengths[i] = partially_registered ? 2 * block_size_ : 4;
                }
                ranges.push_back({
                    .opcode = TransferRequest::READ,
                    .remote_segment = "unit-test-server:1234",
                    .remote_base_offset = kBufferAddr,
                    .remote_size = 2 * block_size_,
                    .local_buffer = reinterpret_cast<void *>(address),
                    .local_capacity = lengths[i],
                    .local_offsets = offsets,
                    .remote_offsets = offsets,
                    .lengths = std::span<const size_t>(&lengths[i], 1),
                    .on_fragment_complete =
                        [&, i](size_t, const Status &status) {
                            ++callbacks[i];
                            succeeded[i] = status.ok();
                        },
                });
            }
            auto operation = engine.submitScatter(
                ranges, {.cancel_on_error = false, .busy_poll = true});
            EXPECT_FALSE(operation.wait().ok());
            EXPECT_EQ(posted_sources.size(), 2u);
            for (size_t i = 0; i < lengths.size(); ++i) {
                EXPECT_EQ(callbacks[i], 1u);
                EXPECT_EQ(succeeded[i], i != bad_index);
            }
        }
    }
};

TEST_F(IndependentScatterRdmaTest, UnregisteredDestinationDoesNotFailPeers) {
    checkFailureIsolation(false);
}

TEST_F(IndependentScatterRdmaTest, PartialRegistrationDoesNotFailPeers) {
    checkFailureIsolation(true);
}
#endif
}  // namespace
