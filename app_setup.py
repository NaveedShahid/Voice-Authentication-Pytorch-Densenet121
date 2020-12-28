import os


WORKING_FOLDER = os.path.dirname(os.path.realpath(__file__))
SAVED_AUDIO_FOLDER = os.path.join(WORKING_FOLDER, "registered users")
if not os.path.exists(SAVED_AUDIO_FOLDER): os.makedirs(SAVED_AUDIO_FOLDER)
print(WORKING_FOLDER)
TEMP_FOLDER = os.path.join(WORKING_FOLDER, "temp")
if not os.path.exists(TEMP_FOLDER): os.makedirs(TEMP_FOLDER)
TEST_FOLDER = os.path.join(WORKING_FOLDER, "test")
if not os.path.exists(TEST_FOLDER): os.makedirs(TEST_FOLDER)