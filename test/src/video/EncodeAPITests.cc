#include <gtest/gtest.h>
#include "EncodeAPI.h"
#include "VideoEncoder.h"

class EncodeAPITestFixture : public testing::Test {
public:
    EncodeAPITestFixture()
      : context(0), encodeAPI(context.get())
    { }

protected:
    GPUContext context;
    EncodeAPI encodeAPI;
    char* preset = "hq";
};

TEST_F(EncodeAPITestFixture, testConstructor) {
}

TEST_F(EncodeAPITestFixture, testPresetGUIDs) {
  auto preset_guid = encodeAPI.GetPresetGUID(preset, NV_ENC_HEVC);

  ASSERT_EQ(preset_guid, NV_ENC_PRESET_HQ_GUID);
}

TEST_F(EncodeAPITestFixture, testCreateEncoder) {
  EncodeConfig configuration("/dev/null", "/dev/null", 1080, 1920, 2, 2,
                             NV_ENC_HEVC, preset, 30, 30, 1024*1024, 0, 0);

  ASSERT_EQ(encodeAPI.CreateEncoder(&configuration), NV_ENC_SUCCESS);
}

TEST_F(EncodeAPITestFixture, testEncodeFrame) {
  EncodeConfig configuration("/dev/null", "/dev/null", 1080, 1920, 2, 2,
                             NV_ENC_HEVC, preset, 30, 30, 1024*1024, 0, 0);
  EncodeBuffer encodeBuffer(encodeAPI, configuration);

  EXPECT_EQ(encodeAPI.NvEncEncodeFrame(&encodeBuffer, nullptr, NV_ENC_PIC_STRUCT_FRAME), NV_ENC_SUCCESS);
}

TEST_F(EncodeAPITestFixture, testProcessOutput) {
  EncodeConfig configuration("/dev/null", "/dev/null", 1080, 1920, 2, 2,
                             NV_ENC_HEVC, preset, 30, 30, 1024*1024, 0, 0);
  EncodeBuffer encodeBuffer(encodeAPI, configuration);

  configuration.fOutput = fopen(configuration.outputFileName, "wb");

  EXPECT_NE(configuration.fOutput, nullptr);

  EXPECT_EQ(encodeAPI.NvEncEncodeFrame(&encodeBuffer, nullptr, NV_ENC_PIC_STRUCT_FRAME), NV_ENC_SUCCESS);

  EXPECT_EQ(fclose(configuration.fOutput), 0);
}