TARGET_DIR="/scratch_aisg/SPEC-SF-AISG/yuli/ARF-Training/repos/KataGo/assets/analysis/inputs"
SOURCE_URL="http://katagoarchive.org/kata1/traininggames/2025-01-20sgfs.tar.bz2"
# download to the target dir
wget $SOURCE_URL -O ${TARGET_DIR}/2025-01-20sgfs.tar.bz2
tar -xjf ${TARGET_DIR}/2025-01-20sgfs.tar.bz2 -C ${TARGET_DIR}