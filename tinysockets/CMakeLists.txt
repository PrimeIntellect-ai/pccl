add_library(tinysockets STATIC src/server_socket.cpp src/blocking_client_socket.cpp src/queued_client_socket.cpp src/blocking_server_socket.cpp src/ccoip_utils.cpp src/poll.cpp src/multiplexed_socket.cpp)

add_subdirectory(mpsc)
add_subdirectory(spsc)
add_subdirectory(lockfree_map)

target_include_directories(tinysockets PUBLIC include)
target_include_directories(tinysockets PRIVATE internal)
target_link_libraries(tinysockets PUBLIC common_utils)
target_link_libraries(tinysockets PUBLIC ccoip_private)
target_link_libraries(tinysockets PUBLIC ccoip_public)
target_link_libraries(tinysockets PUBLIC pccl_log)
target_link_libraries(tinysockets PRIVATE libuv::libuv)
target_link_libraries(tinysockets PUBLIC threadpark)
target_link_libraries(tinysockets PRIVATE mpsc)
target_link_libraries(tinysockets PRIVATE spsc)
target_link_libraries(tinysockets PRIVATE lockfree_map)

add_subdirectory(tests)
