/* Single translation unit that instantiates the stb implementations, so the
 * strict warning flags used for the engine and CLI do not apply to them.   */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
