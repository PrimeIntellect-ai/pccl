#include "ccoip_client.hpp"

#include <ccoip_client_handler.hpp>

ccoip::CCoIPClient::CCoIPClient(const ccoip_socket_address_t &master_socket_address,
                                const uint32_t peer_group) : client(
    new CCoIPClientHandler(master_socket_address, peer_group)) {
}

bool ccoip::CCoIPClient::connect() const {
    return client->connect();
}

bool ccoip::CCoIPClient::acceptNewPeers() const {
    return client->requestAndEstablishP2PConnections(true);
}

bool ccoip::CCoIPClient::arePeersPending(bool &pending_out) const {
    return client->arePeersPending(pending_out);
}

bool ccoip::CCoIPClient::syncSharedState(ccoip_shared_state_t &shared_state,
                                         ccoip_shared_state_sync_info_t &info_out) const {
    return client->syncSharedState(shared_state, info_out);
}

bool ccoip::CCoIPClient::interrupt() const {
    return client->interrupt();
}

bool ccoip::CCoIPClient::optimizeTopology() const {
    return client->optimizeTopology();
}

bool ccoip::CCoIPClient::allReduceAsync(const void *sendbuff, void *recvbuff, const size_t count,
                                        const ccoip_data_type_t datatype, const ccoip_data_type_t quantized_data_type,
                                        const ccoip_quantization_algorithm_t quantization_algorithm,
                                        const ccoip_reduce_op_t op, const uint64_t tag) const {
    return client->allReduceAsync(sendbuff, recvbuff, count, datatype, quantized_data_type, quantization_algorithm, op,
                                  tag);
}

bool ccoip::CCoIPClient::joinAsyncReduce(const uint64_t tag) const {
    return client->joinAsyncReduce(tag);
}

bool ccoip::CCoIPClient::getAsyncReduceInfo(const uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out) const {
    return client->getAsyncReduceInfo(tag, info_out);
}

bool ccoip::CCoIPClient::join() const {
    return client->join();
}

bool ccoip::CCoIPClient::isInterrupted() const {
    return client->isInterrupted();
}

bool ccoip::CCoIPClient::isAnyCollectiveComsOpRunning() const {
    return client->isAnyCollectiveComsOpRunning();
}

size_t ccoip::CCoIPClient::getGlobalWorldSize() const { return client->getGlobalWorldSize(); }

size_t ccoip::CCoIPClient::getNumDistinctPeerGroups() const {
    return client->getNumDistinctPeerGroups();
}

size_t ccoip::CCoIPClient::getLargestPeerGroupWorldSize() const {
    return client->getLargestPeerGroupWorldSize();
}

size_t ccoip::CCoIPClient::getLocalWorldSize() const {
    return client->getLocalWorldSize();
}


void ccoip::CCoIPClient::setMainThread(const std::thread::id main_thread_id) {
    this->main_thread_id = main_thread_id;
    this->client->setMainThread(main_thread_id);
}

ccoip::CCoIPClient::~CCoIPClient() {
    delete client;
}
