add_executable(test_associating_b1 test_associating_b1.cpp)
target_include_directories(test_associating_b1 PRIVATE ${LAMMPS_SOURCE_DIR} ${LAMMPS_SOURCE_DIR}/ASSOCIATING ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_associating_b1 PRIVATE lammps GTest::GMockMain)
add_test(NAME AssociatingB1 COMMAND test_associating_b1)
