# Common Logger Library
add_library(nanodeploy_logger SHARED logging.cpp)

# Install Logger
install(
    TARGETS nanodeploy_logger
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)

add_subdirectory(metrics)
add_subdirectory(scheduler)
add_subdirectory(sequence)
add_subdirectory(worker)
