import math
#
import openeo.processes as oep
from fontTools.misc.cython import returns

from openeo.processes import clip as oe_clip, array_element, array_create, array_create_labeled

maxR = 3.0 # max reflectance
midR = 0.13
sat = 1.2
gamma = 1.8

gOff = 0.01
gOffPow = math.pow(gOff, gamma)
gOffRange = math.pow(1 + gOff, gamma) - gOffPow

def evaluatePixel(smp):
    rgbLin = satEnh(sAdj(smp["B04"]), sAdj(smp["B03"]), sAdj(smp["B02"]))
    # return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2]), smp.dataMask]
    return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2])]

def evaluatePixelArray(smp, depth_in=32760, depth_out=256):
    rgbLin = satEnh([sAdj(smp[0] / depth_in), sAdj(smp[1] / depth_in), sAdj(smp[2] / depth_in) ])
    # return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2]), smp.dataMask]
    return [sRGB(rgbLin[0]) * depth_out, sRGB(rgbLin[1])* depth_out, sRGB(rgbLin[2]) * depth_out]

def clip(s):
    if s > 1:
        return 1
    elif s < 0:
        return 0
    else:
        return s

def adj(a, tx, ty, maxC):
    ar = clip(a / maxC)
    return ar * (ar * (tx / maxC + ty - 1.0) - ty) / (ar * (2.0 * tx / maxC - 1) - tx / maxC)

def adjGamma(b):
    return (math.pow(b + gOff, gamma) - gOffPow) / gOffRange

def sAdj(a):
    return adjGamma(adj(a, midR, 1, maxR))

def sRGB(c):
    if c <=  0.0031308:
        return 12.92 * c
    else:
        return 1.055 * math.pow(c, 0.41666666666) - 0.055

def satEnh(r, g, b):
  avgS = (r + g + b) / 3.0 * (1 - sat)
  return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]

def satEnh2(p):
    r, g, b = p[0], p[1], p[2]
    avgS = (r + g + b) / 3.0 * (1 - sat)
    return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]

def scale_function(x: oep.ProcessBuilder):
    return x.linear_scale_range(0, 6000, 0, 255)

def pixel_tci(x: oep.ProcessBuilder):
    return x.linear_scale_range(0, 6000, 0, 255)


def tci_function(data: oep.ProcessBuilder):
    red_B04 = oep.array_element(data, label="B04")
    blue_B02 = oep.array_element(data, label="B02")
    green_B03 = oep.array_element(data, label="B03")

    rgbLin = satEnh(sAdj(red_B04), sAdj(blue_B02), sAdj(green_B03))

    # return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2])]

    return array_create_labeled([sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2])], ["r, b, g"])
    # return sRGB(rgbLin[0])
