add_sanitized_gtest(test_server_socket test_server_socket.cpp)
target_link_libraries(test_server_socket PRIVATE tinysockets)
target_include_directories(test_server_socket PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_sanitized_gtest(test_blocking_io_socket test_blocking_io_socket.cpp)
target_link_libraries(test_blocking_io_socket PRIVATE tinysockets)
target_include_directories(test_blocking_io_socket PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_sanitized_gtest(test_queued_socket test_queued_socket.cpp)
target_link_libraries(test_queued_socket PRIVATE tinysockets)
target_include_directories(test_queued_socket PRIVATE ${CMAKE_CURRENT_LIST_DIR})