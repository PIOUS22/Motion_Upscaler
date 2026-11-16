#!/usr/bin/env python3
"""ABI check from OUTSIDE C: loads the built libupscaler.so with ctypes (the same mechanism class
as Dart FFI: dlopen + dlsym + raw struct layout) and runs an upscale through the exported symbols.
Usage: test_abi.py path/to/libupscaler.so"""
import ctypes as C, subprocess, sys

lib = C.CDLL(sys.argv[1])

class ImageBuffer(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("width", C.c_int32), ("height", C.c_int32),
                ("channels", C.c_int32), ("stride", C.c_int32)]

class UpscaleOptions(C.Structure):
    _fields_ = [("scale", C.c_int32), ("kernel", C.c_int32), ("flags", C.c_int32),
                ("sharpen_amount", C.c_float), ("sharpen_radius", C.c_float),
                ("sharpen_threshold", C.c_int32)]

assert C.sizeof(ImageBuffer) == 24 and C.sizeof(UpscaleOptions) == 24, "struct size drift"
P = C.POINTER
lib.create_image_buffer.argtypes = [C.c_int32] * 3 + [P(ImageBuffer)]
lib.create_image_buffer.restype = C.c_int32
lib.free_image_buffer.argtypes = [P(ImageBuffer)]; lib.free_image_buffer.restype = None
lib.upscale_image_into.argtypes = [P(ImageBuffer), P(UpscaleOptions), P(ImageBuffer)]
lib.upscale_image_into.restype = C.c_int32
lib.upscaler_options_default.argtypes = [P(UpscaleOptions)]
lib.upscaler_error_string.argtypes = [C.c_int32]; lib.upscaler_error_string.restype = C.c_char_p

src, dst, opt = ImageBuffer(), ImageBuffer(), UpscaleOptions()
assert lib.create_image_buffer(13, 9, 4, C.byref(src)) == 0
assert lib.create_image_buffer(26, 18, 4, C.byref(dst)) == 0
assert C.addressof(src.data.contents) % 64 == 0 and src.stride == 64   # 13*4=52 -> 64
for y in range(9):                                       # fill via the raw pointer, honouring stride
    for x in range(13):
        for c in range(4):
            src.data[y * src.stride + x * 4 + c] = 200 if c == 3 else (x * 19 + y * 7) & 255
lib.upscaler_options_default(C.byref(opt))
assert (opt.scale, opt.kernel) == (2, 1)
opt.flags = 1
assert lib.upscale_image_into(C.byref(src), C.byref(opt), C.byref(dst)) == 0
assert dst.data[3] == 200, "alpha plane must be preserved (flat 200 stays 200)"
opt.scale = 3
assert lib.upscale_image_into(C.byref(src), C.byref(opt), C.byref(dst)) == -3
assert lib.upscaler_error_string(-3) == b"unsupported scale, kernel or channel count"
lib.free_image_buffer(C.byref(src)); lib.free_image_buffer(C.byref(dst))
assert not src.data and src.width == 0                   # struct zeroed => double free is a no-op
lib.free_image_buffer(C.byref(src))

syms = subprocess.check_output(["nm", "-D", "--defined-only", sys.argv[1]], text=True).split("\n")
exported = sorted(l.split()[-1] for l in syms if " T " in l)
print("exported:", ", ".join(exported))
assert all(n.startswith(("create_image", "free_image", "upscal")) for n in exported), exported
needed = subprocess.check_output(["readelf", "-d", sys.argv[1]], text=True)
libs = [l.split("[")[1].rstrip("]") for l in needed.split("\n") if "(NEEDED)" in l]
print("runtime deps (DT_NEEDED):", libs)
assert all(l.startswith(("libc.so", "libm.so", "ld-linux")) for l in libs), libs
print("ABI OK")
