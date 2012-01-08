add_executable(auto_enc auto_enc.cpp)
target_link_libraries(auto_enc ${Boost_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuvnet_ops pthread)
