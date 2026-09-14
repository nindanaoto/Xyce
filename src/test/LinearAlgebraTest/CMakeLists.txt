add_executable( testBlockLinearSystems testBlockLinearSystems.C )
target_link_libraries( testBlockLinearSystems XyceLib )

if(Xyce_KLS AND BUILD_TESTING)
     add_executable(testKLSAdapter testKLSAdapter.C)
     target_link_libraries(testKLSAdapter XyceLib KLS::kls)
     add_test(NAME testKLSAdapter COMMAND testKLSAdapter)
     set_tests_properties(testKLSAdapter PROPERTIES LABELS "unittest" TIMEOUT 60)
     if(Xyce_PARALLEL_MPI)
          add_test(NAME testKLSAdapterMPI COMMAND mpiexec -np 4 $<TARGET_FILE:testKLSAdapter>)
          set_tests_properties(testKLSAdapterMPI PROPERTIES LABELS "parallel;unittest" TIMEOUT 60)
     endif()
endif()

if( BUILD_TESTING )
     add_test( testBlockLinearSystems testBlockLinearSystems )
     set_tests_properties( testBlockLinearSystems PROPERTIES
          REQUIRED_FILES "hb_matrix1.mm"
          REQUIRED_FILES "hb_rhs1.mm" )
     get_target_property(XyceLibDir XyceLib BINARY_DIR )
     set_tests_properties( testBlockLinearSystems PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${XyceLibDir}")
     set_tests_properties( testBlockLinearSystems PROPERTIES LABELS "nightly;unittest" )
     file( COPY hb_matrix1.mm hb_rhs1.mm DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
