BUILD_DIR=${PWD}/../../build_jaxlib
TEST_DIR=${PWD}
cd ${BUILD_DIR} && ./build_script
cd dist/ && pip install -e .
cd ${TEST_DIR}
