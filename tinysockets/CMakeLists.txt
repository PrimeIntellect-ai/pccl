add_library(tinysockets STATIC src/server_socket.cpp src/blocking_client_socket.cpp src/queued_client_socket.cpp src/blocking_server_socket.cpp src/ccoip_utils.cpp)

target_include_directories(tinysockets PUBLIC include)
target_include_directories(tinysockets PRIVATE internal)
target_link_libraries(tinysockets PUBLIC common_utils)
target_link_libraries(tinysockets PUBLIC ccoip_private)
target_link_libraries(tinysockets PUBLIC ccoip_public)
target_link_libraries(tinysockets PUBLIC pccl_log)
target_link_libraries(tinysockets PRIVATE libuv::libuv)

add_subdirectory(tests)