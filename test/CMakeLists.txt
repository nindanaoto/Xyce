if (Xyce_REGRESSION AND DEFINED Xyce_REGRESSION_DIR AND EXISTS "${Xyce_REGRESSION_DIR}/CMakeLists.txt")
  add_subdirectory (${Xyce_REGRESSION_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Xyce_Regression)
endif ()

# unit tests
if(Xyce_GTEST_UNIT_TESTS)
  if(NOT Xyce_PARALLEL_MPI)
    add_subdirectory(CircuitPKG)
  endif()

  add_subdirectory(IOInterfacePKG)
  add_subdirectory(UtilityPKG)
endif()

find_package(Python3 3.7 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME DiodeSidewallCharge
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/DeviceModelPKG/diode_charge.py $<TARGET_FILE:Xyce>)
  set_tests_properties(DiodeSidewallCharge PROPERTIES LABELS "nightly;diode" TIMEOUT 300)
endif()
