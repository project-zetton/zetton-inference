macro(add_simple_excutables dirname)
    file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/${dirname}/*.cc")
    foreach (file ${files})
        get_filename_component(name ${file} NAME_WE)
        add_simple_excutable(${dirname} ${name})
    endforeach ()
endmacro()

macro(add_simple_excutable dirname name)
    add_executable(${dirname}_${name} ${CMAKE_CURRENT_SOURCE_DIR}/${dirname}/${name}.cc)
    target_link_libraries(${dirname}_${name} ${PROJECT_NAME})
    install(TARGETS ${dirname}_${name}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
endmacro()

macro(add_simple_apps)
    add_simple_excutables("app")
endmacro()

macro(add_simple_examples)
    add_simple_excutables("example")
endmacro()